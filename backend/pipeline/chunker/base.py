from abc import ABC, abstractmethod


class BaseChunker(ABC):
    @abstractmethod
    def split(self, text: str) -> list[str]:
        """텍스트를 청크 리스트로 분할한다."""
