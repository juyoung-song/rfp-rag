"""
청킹 전략 비교 실험 (ODL markdown 텍스트 기준)
- Fixed(200/20) vs Fixed(500/50) vs Recursive(500/50)
- 측정: 청크 수, 평균/최소/최대 청크 길이
"""
import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.loader.opendataloader_loader import OpenDataLoaderLoader
from pipeline.chunker.fixed_chunker import FixedChunker
from pipeline.chunker.recursive_chunker import RecursiveChunker
from pipeline.chunker.markdown_header_chunker import MarkdownHeaderChunker

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SAMPLE_PDFS = list(DATA_DIR.glob("*.pdf"))[:5]


def analyze(chunks: list[str]) -> dict:
    lengths = [len(c) for c in chunks]
    return {
        "count": len(chunks),
        "avg": int(sum(lengths) / len(lengths)),
        "min": min(lengths),
        "max": max(lengths),
    }


def main():
    print(f"\n{'='*65}")
    print(f"청킹 전략 비교 실험 (ODL markdown) — 샘플 {len(SAMPLE_PDFS)}개")
    print(f"{'='*65}\n")

    with tempfile.TemporaryDirectory() as tmp:
        loader = OpenDataLoaderLoader(output_dir=Path(tmp))
        print("▶ ODL markdown 변환 중 (배치)...")
        texts = loader.load_batch(SAMPLE_PDFS)
        print(f"  완료: {len(texts)}개 파일\n")

    strategies = [
        ("Fixed(200/20)", FixedChunker(200, 20)),
        ("Fixed(500/50)", FixedChunker(500, 50)),
        ("Recursive(500/50)", RecursiveChunker(500, 50)),
        ("Recursive(1000/100)", RecursiveChunker(1000, 100)),
        ("MarkdownHeader(1000/100)", MarkdownHeaderChunker(1000, 100)),
    ]

    results = []
    for name, chunker in strategies:
        all_stats = []
        for text in texts.values():
            if text:
                chunks = chunker.split(text)
                all_stats.append(analyze(chunks))

        avg_count = int(sum(s["count"] for s in all_stats) / len(all_stats))
        avg_len = int(sum(s["avg"] for s in all_stats) / len(all_stats))
        avg_min = int(sum(s["min"] for s in all_stats) / len(all_stats))
        avg_max = int(sum(s["max"] for s in all_stats) / len(all_stats))
        results.append((name, avg_count, avg_len, avg_min, avg_max))
        print(f"▶ {name}")
        print(f"  평균 청크 수: {avg_count} | 평균 길이: {avg_len}자 | 최소: {avg_min}자 | 최대: {avg_max}자")

    print(f"\n{'='*65}")
    print(f"{'전략':<22} {'청크 수':<10} {'평균 길이':<12} {'최소':<10} {'최대'}")
    print("-" * 65)
    for r in results:
        print(f"{r[0]:<22} {r[1]:<10} {r[2]:<12} {r[3]:<10} {r[4]}")
    print()


if __name__ == "__main__":
    main()
