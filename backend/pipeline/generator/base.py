from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class GeneratorResponse:
    answer: str
    history: list[Message] = field(default_factory=list)


class BaseGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        query: str,
        context_chunks: list[str],
        history: list[Message] | None = None,
    ) -> GeneratorResponse:
        """검색된 컨텍스트와 대화 히스토리를 기반으로 답변을 생성한다."""
