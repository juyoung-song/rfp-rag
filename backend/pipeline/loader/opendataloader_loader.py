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
            format="text",
            quiet=True,
        )
        txt_path = self.output_dir / (file_path.stem + ".txt")
        if not txt_path.exists():
            raise FileNotFoundError(f"OpenDataLoader 변환 실패: {txt_path}")
        return txt_path.read_text(encoding="utf-8")
