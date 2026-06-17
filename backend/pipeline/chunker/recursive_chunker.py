from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import BaseChunker


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def split(self, text: str) -> list[str]:
        return self._splitter.split_text(text)
