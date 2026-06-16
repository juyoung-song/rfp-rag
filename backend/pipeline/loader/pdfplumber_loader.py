from pathlib import Path
import pdfplumber
from .base import BaseLoader


class PdfplumberLoader(BaseLoader):
    def load(self, file_path: Path) -> str:
        pages = []
        with pdfplumber.open(str(file_path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)
