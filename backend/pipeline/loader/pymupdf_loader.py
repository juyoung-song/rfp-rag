from pathlib import Path
import fitz  # PyMuPDF
from .base import BaseLoader


class PyMuPDFLoader(BaseLoader):
    def load(self, file_path: Path) -> str:
        doc = fitz.open(str(file_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
