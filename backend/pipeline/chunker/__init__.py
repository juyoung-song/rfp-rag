# ADR-006: MarkdownHeaderChunker 선택 (헤딩 기준 분할, 최소 93자, 평균 859자)
from .markdown_header_chunker import MarkdownHeaderChunker

DEFAULT_CHUNKER = MarkdownHeaderChunker
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
