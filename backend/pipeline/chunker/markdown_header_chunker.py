from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from .base import BaseChunker

# 분할 기준 헤딩 레벨
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


class MarkdownHeaderChunker(BaseChunker):
    """
    마크다운 헤딩(#, ##, ###) 기준 1차 분할 후
    여전히 chunk_size를 초과하는 청크는 RecursiveCharacterTextSplitter로 2차 분할.
    헤딩 메타데이터는 청크 텍스트 앞에 접두어로 포함시켜 임베딩 품질 유지.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS,
            strip_headers=False,
        )
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def split(self, text: str) -> list[str]:
        header_chunks = self._header_splitter.split_text(text)
        chunks = []
        for doc in header_chunks:
            content = doc.page_content.strip()
            if not content:
                continue
            if len(content) <= self._fallback_splitter._chunk_size:
                chunks.append(content)
            else:
                # 너무 긴 섹션은 Recursive로 2차 분할
                sub = self._fallback_splitter.split_text(content)
                chunks.extend(sub)
        return [c for c in chunks if len(c) >= 20]
