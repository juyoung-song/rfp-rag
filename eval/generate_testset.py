"""
RAGAS v0.2+ 기반 평가 데이터셋 생성 스크립트.

ADR-007 설계에 따라:
- 100개 전체 RFP 문서에서 Knowledge Graph 기반으로 한국어 QA 쌍 자동 생성
- testset_size=60 (Negative case 5개는 수작업으로 별도 추가)
- 결과: eval/testset_v1.json (RAGAS v0.2+ 필드명)
"""

import csv
import json
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from pipeline.loader.opendataloader_loader import OpenDataLoaderLoader  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_LIST_CSV = DATA_DIR / "data_list.csv"
OUTPUT_PATH = Path(__file__).parent / "testset_v1.json"
ODL_OUTPUT_DIR = Path(__file__).parent / "odl_output"
TESTSET_SIZE = 60


def load_metadata_map() -> dict[str, dict]:
    """data_list.csv에서 파일명 → {사업명, 발주기관} 매핑 반환."""
    meta_map: dict[str, dict] = {}
    with open(DATA_LIST_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            filename = row.get("파일명", "").strip()
            if filename:
                meta_map[filename] = {
                    "사업명": row.get("사업명", "").strip(),
                    "발주기관": row.get("발주 기관", "").strip(),
                }
    return meta_map


def load_documents(meta_map: dict[str, dict]) -> list[Document]:
    """ODL 로더로 PDF 텍스트 추출 후 LangChain Document로 변환."""
    loader = OpenDataLoaderLoader(output_dir=ODL_OUTPUT_DIR)
    docs: list[Document] = []

    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    print(f"PDF 파일 수: {len(pdf_paths)}개")

    for i, pdf_path in enumerate(pdf_paths, 1):
        try:
            text = loader.load(pdf_path)
            if not text.strip():
                print(f"  [{i}/{len(pdf_paths)}] SKIP (빈 텍스트): {pdf_path.name}")
                continue

            meta = meta_map.get(pdf_path.name, {})
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "사업명": meta.get("사업명", ""),
                    "발주기관": meta.get("발주기관", ""),
                },
            ))
            print(f"  [{i}/{len(pdf_paths)}] OK: {pdf_path.name}")
        except Exception as e:
            print(f"  [{i}/{len(pdf_paths)}] ERROR: {pdf_path.name} — {e}")

    print(f"\n로드 완료: {len(docs)}개 문서")
    return docs


def build_query_distribution(generator_llm: ChatOpenAI) -> list:
    """ADR-007 설계에 따른 Synthesizer 분포: SingleHop 70% / MultiHop 30%."""
    return [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="headlines"), 0.4),
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="keyphrases"), 0.3),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.3),
    ]


def save_testset(testset, output_path: Path) -> None:
    """RAGAS EvaluationDataset → testset_v1.json 저장 (v0.2+ 필드명)."""
    df = testset.to_pandas()
    records = df.to_dict(orient="records")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {output_path} ({len(records)}개)")


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR.parent / ".env")

    generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    meta_map = load_metadata_map()
    docs = load_documents(meta_map)

    if not docs:
        print("로드된 문서가 없습니다. 종료.")
        return

    query_distribution = build_query_distribution(generator_llm)

    generator = TestsetGenerator.from_langchain(
        llm=generator_llm,
        embedding_model=embeddings,
    )

    print(f"\nRAGAS 테스트셋 생성 시작 (testset_size={TESTSET_SIZE})...")
    testset = generator.generate_with_langchain_docs(
        docs,
        testset_size=TESTSET_SIZE,
        query_distribution=query_distribution,
        raise_exceptions=False,
    )

    save_testset(testset, OUTPUT_PATH)
    print("\n완료. Negative case 5개는 수작업으로 testset_v1.json에 추가하세요.")


if __name__ == "__main__":
    main()
