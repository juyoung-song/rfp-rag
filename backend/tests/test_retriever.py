import pytest
from unittest.mock import MagicMock
from pipeline.retriever.naive_retriever import NaiveRetriever
from pipeline.retriever.base import RetrievedChunk


@pytest.fixture
def mock_chroma():
    client = MagicMock()
    client.query.return_value = {
        "documents": [["국민연금공단 이러닝시스템 관련 내용", "한국원자력연구원 선량평가 내용"]],
        "metadatas": [[{"file": "a.pdf"}, {"file": "b.pdf"}]],
        "distances": [[0.1, 0.4]],
    }
    return client


def test_retrieve_returns_chunks(mock_chroma):
    retriever = NaiveRetriever(chroma_client=mock_chroma)
    results = retriever.retrieve(query_embedding=[0.1] * 1536, top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_retrieve_score_conversion(mock_chroma):
    retriever = NaiveRetriever(chroma_client=mock_chroma)
    results = retriever.retrieve(query_embedding=[0.1] * 1536, top_k=2)

    # distance 0.1 → score 0.9, distance 0.4 → score 0.6
    assert abs(results[0].score - 0.9) < 1e-6
    assert abs(results[1].score - 0.6) < 1e-6


def test_retrieve_ranking_order(mock_chroma):
    retriever = NaiveRetriever(chroma_client=mock_chroma)
    results = retriever.retrieve(query_embedding=[0.1] * 1536, top_k=2)

    # Chroma가 이미 distance 오름차순 정렬 → score 내림차순
    assert results[0].score > results[1].score


def test_retrieve_metadata_preserved(mock_chroma):
    retriever = NaiveRetriever(chroma_client=mock_chroma)
    results = retriever.retrieve(query_embedding=[0.1] * 1536, top_k=2)

    assert results[0].metadata["file"] == "a.pdf"
    assert results[1].metadata["file"] == "b.pdf"


def test_retrieve_calls_chroma_with_correct_args(mock_chroma):
    retriever = NaiveRetriever(chroma_client=mock_chroma)
    embedding = [0.5] * 1536
    retriever.retrieve(query_embedding=embedding, top_k=3)

    mock_chroma.query.assert_called_once_with(query_embedding=embedding, n_results=3)
