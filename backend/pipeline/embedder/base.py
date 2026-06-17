from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """단일 텍스트를 임베딩 벡터로 변환한다."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """복수 텍스트를 임베딩 벡터 리스트로 변환한다."""
