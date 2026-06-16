from pathlib import Path
import opendataloader_pdf
from .base import BaseLoader


class OpenDataLoaderLoader(BaseLoader):
    def __init__(self, output_dir: Path = Path("./odl_output")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load(self, file_path: Path) -> str:
        opendataloader_pdf.convert(
            input_path=str(file_path),
            output_dir=str(self.output_dir),
            format="markdown",
            quiet=True,
        )
        md_path = self.output_dir / (file_path.stem + ".md")
        if not md_path.exists():
            raise FileNotFoundError(f"OpenDataLoader 변환 실패: {md_path}")
        return md_path.read_text(encoding="utf-8")

    def load_batch(self, file_paths: list[Path]) -> dict[str, str]:
        """여러 PDF를 배치로 처리 (JVM 1회 기동). {파일명: 텍스트} 반환."""
        opendataloader_pdf.convert(
            input_path=[str(p) for p in file_paths],
            output_dir=str(self.output_dir),
            format="markdown",
            quiet=True,
        )
        results = {}
        for file_path in file_paths:
            md_path = self.output_dir / (file_path.stem + ".md")
            if md_path.exists():
                results[file_path.name] = md_path.read_text(encoding="utf-8")
            else:
                results[file_path.name] = ""
        return results
