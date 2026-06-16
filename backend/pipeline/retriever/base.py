from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    score: float


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        """쿼리 임베딩으로 유사 청크를 반환한다."""
