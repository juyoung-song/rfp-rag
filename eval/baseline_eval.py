"""
Phase 1 Naive RAG 베이스라인 평가 스크립트

사용법:
    cd backend
    python ../eval/baseline_eval.py \
        --data-dir ../data \
        --questions ../eval/questions.json \
        --output ../eval/results/baseline_YYYYMMDD.json \
        --top-k 5
"""
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from config import settings
from db.chroma_client import ChromaClient
from pipeline.embedder.openai_embedder import OpenAIEmbedder
from pipeline.retriever.naive_retriever import NaiveRetriever
from pipeline.generator.openai_generator import OpenAIGenerator


def keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    """답변에 기대 키워드가 몇 개 포함됐는지 비율로 반환."""
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw in answer)
    return hits / len(keywords)


def run_evaluation(
    questions: list[dict],
    embedder: OpenAIEmbedder,
    retriever: NaiveRetriever,
    generator: OpenAIGenerator,
    top_k: int,
) -> list[dict]:
    results = []
    for q in questions:
        qid = q["id"]
        question = q["question"]
        expected_kw = q.get("expected_keywords", [])

        print(f"  [{qid}] {question[:40]}...")
        t0 = time.time()

        query_vec = embedder.embed(question)
        retrieved = retriever.retrieve(query_vec, top_k=top_k)
        gen_result = generator.generate(
            query=question,
            context_chunks=[r.text for r in retrieved],
        )
        elapsed = time.time() - t0

        hit_rate = keyword_hit_rate(gen_result.answer, expected_kw)
        retrieval_scores = [r.score for r in retrieved]

        results.append({
            "id": qid,
            "category": q.get("category", ""),
            "question": question,
            "answer": gen_result.answer,
            "keyword_hit_rate": round(hit_rate, 4),
            "top_retrieved_score": round(retrieval_scores[0], 4) if retrieval_scores else 0.0,
            "avg_retrieved_score": round(sum(retrieval_scores) / len(retrieval_scores), 4) if retrieval_scores else 0.0,
            "retrieved_sources": [r.metadata.get("source", "") for r in retrieved],
            "latency_sec": round(elapsed, 3),
        })
        print(f"       hit_rate={hit_rate:.2f}  top_score={retrieval_scores[0]:.3f}  {elapsed:.1f}s")

    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    return {
        "total_questions": n,
        "avg_keyword_hit_rate": round(sum(r["keyword_hit_rate"] for r in results) / n, 4),
        "avg_top_retrieved_score": round(sum(r["top_retrieved_score"] for r in results) / n, 4),
        "avg_latency_sec": round(sum(r["latency_sec"] for r in results) / n, 3),
        "perfect_hit_count": sum(1 for r in results if r["keyword_hit_rate"] == 1.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Naive RAG 베이스라인 평가")
    parser.add_argument("--data-dir", type=Path, default=Path("../data"))
    parser.add_argument("--questions", type=Path, default=Path("../eval/questions.json"))
    parser.add_argument("--output", type=Path, default=Path("../eval/results/baseline.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--collection", type=str, default=settings.chroma_collection_name)
    parser.add_argument("--questions-version", type=str, default="")
    parser.add_argument("--pipeline-version", type=str, default="phase1-naive")
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

    print(f"\n평가 시작 (top_k={args.top_k})")
    print("-" * 60)
    results = run_evaluation(questions, embedder, retriever, generator, args.top_k)

    summary = summarize(results)
    print("\n" + "=" * 60)
    print("평가 요약")
    print(f"  총 질문 수:          {summary['total_questions']}")
    print(f"  평균 키워드 적중률:  {summary['avg_keyword_hit_rate']:.2%}")
    print(f"  평균 검색 유사도:    {summary['avg_top_retrieved_score']:.4f}")
    print(f"  평균 응답 시간:      {summary['avg_latency_sec']:.2f}s")
    print(f"  완전 적중 질문 수:   {summary['perfect_hit_count']}/{summary['total_questions']}")
    print("=" * 60)

    output = {
        "eval_date": datetime.now().isoformat(),
        "pipeline_version": args.pipeline_version,
        "questions_version": args.questions_version or args.questions.stem,
        "questions_file": str(args.questions),
        "top_k": args.top_k,
        "total_chunks_in_db": total_chunks,
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    main()
