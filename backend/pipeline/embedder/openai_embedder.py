from openai import OpenAI
from .base import BaseEmbedder

MODEL = "text-embedding-3-small"


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, api_key: str):
        self._client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=MODEL, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=MODEL, input=texts)
        return [item.embedding for item in response.data]
