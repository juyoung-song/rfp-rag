"""
PDF 파서 비교 실험
- 측정 항목: 텍스트 추출 길이, 처리 속도, 한국어 문자 비율
- 대상: PyMuPDF, pdfplumber, OpenDataLoader
- 사용 PDF: data/ 폴더 내 5개 샘플
"""
import time
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.loader.pymupdf_loader import PyMuPDFLoader
from pipeline.loader.pdfplumber_loader import PdfplumberLoader
from pipeline.loader.opendataloader_loader import OpenDataLoaderLoader

DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "odl_output"

SAMPLE_PDFS = list(DATA_DIR.glob("*.pdf"))[:5]


def korean_ratio(text: str) -> float:
    if not text:
        return 0.0
    korean_chars = sum(1 for ch in text if '가' <= ch <= '힣')
    return korean_chars / len(text)


def run_experiment(loader_name: str, loader, pdfs: list) -> dict:
    results = []
    total_start = time.time()
    for pdf in pdfs:
        start = time.time()
        try:
            text = loader.load(pdf)
            elapsed = time.time() - start
            results.append({
                "file": pdf.name[:35],
                "length": len(text),
                "korean_ratio": round(korean_ratio(text), 3),
                "elapsed_sec": round(elapsed, 2),
                "error": None,
            })
        except Exception as e:
            results.append({
                "file": pdf.name[:35],
                "length": 0,
                "korean_ratio": 0,
                "elapsed_sec": round(time.time() - start, 2),
                "error": str(e)[:80],
            })
    total_elapsed = round(time.time() - total_start, 2)
    valid = [r for r in results if r["error"] is None]
    avg_length = sum(r["length"] for r in valid) / len(valid) if valid else 0
    avg_korean = sum(r["korean_ratio"] for r in valid) / len(valid) if valid else 0
    return {
        "loader": loader_name,
        "total_elapsed_sec": total_elapsed,
        "avg_text_length": int(avg_length),
        "avg_korean_ratio": round(avg_korean, 3),
        "details": results,
    }


def main():
    print(f"\n{'='*65}")
    print(f"PDF 파서 비교 실험 — 샘플 {len(SAMPLE_PDFS)}개")
    print(f"{'='*65}\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        loaders = [
            ("PyMuPDF", PyMuPDFLoader()),
            ("pdfplumber", PdfplumberLoader()),
            ("OpenDataLoader", OpenDataLoaderLoader(output_dir=Path(tmpdir))),
        ]

        summaries = []
        for name, loader in loaders:
            print(f"▶ {name} 실행 중...")
            result = run_experiment(name, loader, SAMPLE_PDFS)
            summaries.append(result)
            print(f"  총 {result['total_elapsed_sec']}초 | 평균 길이: {result['avg_text_length']:,}자 | 한국어 비율: {result['avg_korean_ratio']:.1%}")
            for d in result["details"]:
                status = f"✅ {d['length']:,}자" if not d["error"] else f"❌ {d['error']}"
                print(f"    {d['file']}: {status} ({d['elapsed_sec']}s)")
            print()

    print(f"\n{'='*65}")
    print("📊 종합 비교")
    print(f"{'로더':<20} {'총 시간(s)':<12} {'평균 길이':<14} {'한국어 비율'}")
    print("-" * 65)
    for s in summaries:
        print(f"{s['loader']:<20} {s['total_elapsed_sec']:<12} {s['avg_text_length']:<14,} {s['avg_korean_ratio']:.1%}")
    print()


if __name__ == "__main__":
    main()
