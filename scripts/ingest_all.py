"""
전체 PDF 인제스트 스크립트
사용법: cd backend && python ../scripts/ingest_all.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from config import settings
from db.chroma_client import ChromaClient
from pipeline.loader.opendataloader_loader import OpenDataLoaderLoader
from pipeline.chunker.markdown_header_chunker import MarkdownHeaderChunker
from pipeline.embedder.openai_embedder import OpenAIEmbedder
from pipeline.ingest_pipeline import IngestPipeline
from pipeline.metadata_loader import MetadataLoader


def main():
    data_dir = Path(__file__).parent.parent / "data"
    pdf_files = sorted(data_dir.glob("*.pdf"))
    print(f"총 PDF 수: {len(pdf_files)}")

    chroma = ChromaClient(host=None, collection_name=settings.chroma_collection_name)
    existing = chroma.count()
    if existing > 0:
        print(f"기존 DB에 {existing}개 청크 존재 → 초기화 후 재인제스트")
        chroma.reset()

    meta_csv = data_dir / "metadata.csv"
    meta_loader = MetadataLoader(meta_csv) if meta_csv.exists() else None

    pipeline = IngestPipeline(
        loader=OpenDataLoaderLoader(output_dir=data_dir / "odl_output"),
        chunker=MarkdownHeaderChunker(chunk_size=1000, chunk_overlap=100),
        embedder=OpenAIEmbedder(api_key=settings.openai_api_key),
        chroma_client=chroma,
        metadata_loader=meta_loader,
    )

    total_chunks = 0
    errors = []
    t_start = time.time()

    for i, pdf in enumerate(pdf_files, 1):
        t0 = time.time()
        try:
            count = pipeline.ingest_file(pdf)
            elapsed = time.time() - t0
            total_chunks += count
            print(f"[{i:3d}/{len(pdf_files)}] {pdf.name[:50]:<50} {count:4d} 청크  {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            errors.append((pdf.name, str(e)))
            print(f"[{i:3d}/{len(pdf_files)}] ERROR: {pdf.name[:40]} — {e}  {elapsed:.1f}s")

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"인제스트 완료: {len(pdf_files) - len(errors)}/{len(pdf_files)} 성공")
    print(f"총 저장 청크: {total_chunks}")
    print(f"총 소요 시간: {total_elapsed/60:.1f}분")
    if errors:
        print(f"\n실패 파일 ({len(errors)}개):")
        for name, err in errors:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
