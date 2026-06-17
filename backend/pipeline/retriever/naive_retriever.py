from db.chroma_client import ChromaClient
from .base import BaseRetriever, RetrievedChunk


class NaiveRetriever(BaseRetriever):
    def __init__(self, chroma_client: ChromaClient):
        self._db = chroma_client

    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        results = self._db.query(query_embedding=query_embedding, n_results=top_k)

        chunks = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for text, meta, dist in zip(docs, metas, distances):
            # Chroma cosine distance → similarity: score = 1 - distance
            chunks.append(RetrievedChunk(text=text, metadata=meta, score=1.0 - dist))

        return chunks
