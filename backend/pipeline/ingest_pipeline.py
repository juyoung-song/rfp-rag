import hashlib
from pathlib import Path

from pipeline.loader.base import BaseLoader
from pipeline.chunker.base import BaseChunker
from pipeline.embedder.base import BaseEmbedder
from pipeline.metadata_loader import MetadataLoader
from db.chroma_client import ChromaClient


def _chunk_id(file_name: str, index: int) -> str:
    raw = f"{file_name}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()


class IngestPipeline:
    def __init__(
        self,
        loader: BaseLoader,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        chroma_client: ChromaClient,
        metadata_loader: MetadataLoader | None = None,
    ):
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._db = chroma_client
        self._metadata_loader = metadata_loader

    def ingest_file(self, file_path: Path) -> int:
        """단일 PDF를 파싱→청크→임베딩→저장. 저장된 청크 수 반환."""
        text = self._loader.load(file_path)
        chunks = self._chunker.split(text)
        if not chunks:
            return 0

        embeddings = self._embedder.embed_batch(chunks)

        base_meta = {"source": file_path.name}
        if self._metadata_loader:
            doc_meta = self._metadata_loader.get_by_filename(file_path.name)
            if doc_meta:
                base_meta.update({k: str(v) for k, v in doc_meta.items()})

        ids = [_chunk_id(file_path.name, i) for i in range(len(chunks))]
        metadatas = [{**base_meta, "chunk_index": i} for i in range(len(chunks))]

        self._db.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        return len(chunks)

    def ingest_batch(self, file_paths: list[Path]) -> dict[str, int]:
        """여러 PDF를 순차 처리. {파일명: 청크수} 반환."""
        return {fp.name: self.ingest_file(fp) for fp in file_paths}
