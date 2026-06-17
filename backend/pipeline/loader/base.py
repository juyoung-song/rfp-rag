from abc import ABC, abstractmethod
from pathlib import Path


class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_path: Path) -> str:
        """PDF 파일을 읽어 텍스트를 반환한다."""
