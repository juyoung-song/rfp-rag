import pytest
from unittest.mock import patch, MagicMock
from pipeline.embedder.openai_embedder import OpenAIEmbedder


def _make_mock_client(embeddings: list[list[float]]) -> MagicMock:
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=e) for e in embeddings]
    )
    return mock_client


@pytest.fixture
def mock_openai_single():
    with patch("pipeline.embedder.openai_embedder.OpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_client([[0.1] * 1536])
        yield mock_cls


@pytest.fixture
def mock_openai_batch():
    with patch("pipeline.embedder.openai_embedder.OpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_client([[0.1] * 1536, [0.2] * 1536])
        yield mock_cls


def test_openai_embedder_returns_vector(mock_openai_single):
    embedder = OpenAIEmbedder(api_key="test-key")
    vector = embedder.embed("국민연금공단 이러닝시스템")
    assert isinstance(vector, list)
    assert len(vector) == 1536


def test_openai_embedder_calls_api(mock_openai_single):
    embedder = OpenAIEmbedder(api_key="test-key")
    embedder.embed("테스트 텍스트")
    embedder._client.embeddings.create.assert_called_once()


def test_openai_embedder_batch(mock_openai_batch):
    embedder = OpenAIEmbedder(api_key="test-key")
    vectors = embedder.embed_batch(["텍스트 1", "텍스트 2"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 1536
    assert len(vectors[1]) == 1536
