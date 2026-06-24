"""
Phase 1 Naive RAG 베이스라인 평가 스크립트

사용법:
    cd backend
    python ../eval/baseline_eval.py \
        --questions ../eval/questions_v1.json \
        --output ../eval/results/baseline_YYYYMMDD.json \
        --top-k 5
        [--no-judge]  # LLM-as-Judge 생략 시
"""
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from openai import OpenAI
from config import settings
from db.chroma_client import ChromaClient
from pipeline.embedder.openai_embedder import OpenAIEmbedder
from pipeline.retriever.naive_retriever import NaiveRetriever
from pipeline.generator.openai_generator import OpenAIGenerator

# ── Judge 프롬프트 ────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """당신은 RAG 시스템의 답변 품질을 평가하는 전문가입니다.
주어진 질문, 검색된 컨텍스트, 생성된 답변을 보고 아래 기준으로 0~3점을 부여하세요.

[평가 기준]
3점: 컨텍스트 기반으로 질문에 정확하게 답변. 핵심 정보 포함.
2점: 대체로 정확하나 일부 정보 누락 또는 불명확.
1점: 질문과 관련 있으나 핵심 정보 오류 또는 중요 내용 누락.
0점: 완전히 잘못된 답변이거나 "찾을 수 없습니다"류 무응답.

반드시 아래 JSON 형식으로만 응답하세요:
{"score": <0~3 정수>, "reason": "<한 문장 판단 근거>"}"""

_JUDGE_USER = """[질문]
{question}

[검색된 컨텍스트]
{context}

[생성된 답변]
{answer}"""


def llm_judge(
    client: OpenAI,
    question: str,
    context_chunks: list[str],
    answer: str,
) -> dict:
    """LLM-as-Judge: 답변 품질을 0~3점으로 평가. {"score": int, "reason": str} 반환."""
    context_text = "\n---\n".join(f"[{i+1}] {c[:300]}" for i, c in enumerate(context_chunks))
    user_msg = _JUDGE_USER.format(
        question=question,
        context=context_text,
        answer=answer,
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    parsed = json.loads(raw)
    score = max(0, min(3, int(parsed.get("score", 0))))
    return {"score": score, "reason": parsed.get("reason", "")}


# ── 기존 키워드 적중률 (레거시 참고용) ───────────────────────────────────────

def keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    return sum(1 for kw in keywords if kw in answer) / len(keywords)


# ── 평가 실행 ─────────────────────────────────────────────────────────────────

def run_evaluation(
    questions: list[dict],
    embedder: OpenAIEmbedder,
    retriever: NaiveRetriever,
    generator: OpenAIGenerator,
    top_k: int,
    judge_client: OpenAI | None = None,
) -> list[dict]:
    results = []
    for q in questions:
        qid = q["id"]
        question = q["question"]
        expected_kw = q.get("expected_keywords", [])

        print(f"  [{qid}] {question[:45]}...")
        t0 = time.time()

        query_vec = embedder.embed(question)
        retrieved = retriever.retrieve(query_vec, top_k=top_k)
        context_chunks = [r.text for r in retrieved]
        context_metadata = [r.metadata for r in retrieved]
        gen_result = generator.generate(
            query=question,
            context_chunks=context_chunks,
            context_metadata=context_metadata,
        )
        elapsed = time.time() - t0

        hit_rate = keyword_hit_rate(gen_result.answer, expected_kw)
        retrieval_scores = [r.score for r in retrieved]

        row: dict = {
            "id": qid,
            "category": q.get("category", ""),
            "question": question,
            "answer": gen_result.answer,
            "keyword_hit_rate": round(hit_rate, 4),
            "top_retrieved_score": round(retrieval_scores[0], 4) if retrieval_scores else 0.0,
            "avg_retrieved_score": round(
                sum(retrieval_scores) / len(retrieval_scores), 4
            ) if retrieval_scores else 0.0,
            "retrieved_sources": [r.metadata.get("source", "") for r in retrieved],
            "latency_sec": round(elapsed, 3),
        }

        if judge_client is not None:
            judge_result = llm_judge(judge_client, question, context_chunks, gen_result.answer)
            row["judge_score"] = round(judge_result["score"] / 3, 4)   # 0-1 정규화
            row["judge_score_raw"] = judge_result["score"]              # 0-3 원점수
            row["judge_reason"] = judge_result["reason"]
            print(
                f"       judge={judge_result['score']}/3"
                f"  hit_rate={hit_rate:.2f}"
                f"  top_score={retrieval_scores[0]:.3f}"
                f"  {elapsed:.1f}s"
            )
        else:
            print(f"       hit_rate={hit_rate:.2f}  top_score={retrieval_scores[0]:.3f}  {elapsed:.1f}s")

        results.append(row)

    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    base = {
        "total_questions": n,
        "avg_keyword_hit_rate": round(sum(r["keyword_hit_rate"] for r in results) / n, 4),
        "avg_top_retrieved_score": round(sum(r["top_retrieved_score"] for r in results) / n, 4),
        "avg_latency_sec": round(sum(r["latency_sec"] for r in results) / n, 3),
        "perfect_keyword_hit_count": sum(1 for r in results if r["keyword_hit_rate"] == 1.0),
    }
    if "judge_score" in results[0]:
        base["avg_judge_score"] = round(sum(r["judge_score"] for r in results) / n, 4)
        base["perfect_judge_count"] = sum(1 for r in results if r["judge_score_raw"] == 3)
    return base


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Naive RAG 베이스라인 평가")
    parser.add_argument("--data-dir", type=Path, default=Path("../data"))
    parser.add_argument("--questions", type=Path, default=Path("../eval/questions_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("../eval/results/baseline.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--collection", type=str, default=settings.chroma_collection_name)
    parser.add_argument("--questions-version", type=str, default="")
    parser.add_argument("--pipeline-version", type=str, default="phase1-naive")
    parser.add_argument("--no-judge", action="store_true", help="LLM-as-Judge 평가 생략")
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    print(f"평가 질문 수: {len(questions)}")

    chroma = ChromaClient(host=None, collection_name=args.collection)
    total_chunks = chroma.count()
    print(f"벡터 DB 청크 수: {total_chunks}")
    if total_chunks == 0:
        print("❌ 벡터 DB가 비어 있습니다. 먼저 데이터를 인제스트하세요.")
        sys.exit(1)

    embedder = OpenAIEmbedder(api_key=settings.openai_api_key)
    retriever = NaiveRetriever(chroma_client=chroma)
    generator = OpenAIGenerator(api_key=settings.openai_api_key)
    judge_client = None if args.no_judge else OpenAI(api_key=settings.openai_api_key)

    mode = "키워드 적중률만" if args.no_judge else "LLM-as-Judge 포함"
    print(f"\n평가 시작 (top_k={args.top_k}, {mode})")
    print("-" * 60)
    results = run_evaluation(questions, embedder, retriever, generator, args.top_k, judge_client)

    summary = summarize(results)
    print("\n" + "=" * 60)
    print("평가 요약")
    if "avg_judge_score" in summary:
        print(f"  평균 Judge 점수:     {summary['avg_judge_score']:.2%}  (완전정답: {summary['perfect_judge_count']}/{summary['total_questions']})")
    print(f"  평균 키워드 적중률:  {summary['avg_keyword_hit_rate']:.2%}  (레거시)")
    print(f"  평균 검색 유사도:    {summary['avg_top_retrieved_score']:.4f}")
    print(f"  평균 응답 시간:      {summary['avg_latency_sec']:.2f}s")
    print("=" * 60)

    output = {
        "eval_date": datetime.now().isoformat(),
        "pipeline_version": args.pipeline_version,
        "questions_version": args.questions_version or args.questions.stem,
        "questions_file": str(args.questions),
        "top_k": args.top_k,
        "total_chunks_in_db": total_chunks,
        "judge_enabled": not args.no_judge,
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    main()
