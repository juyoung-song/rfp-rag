import pytest
from db.chroma_client import ChromaClient


@pytest.fixture
def chroma_client(tmp_path):
    client = ChromaClient(
        host=None,
        collection_name="test_collection",
        persist_dir=str(tmp_path / "chroma"),
    )
    yield client
    client.reset()


def test_chroma_add_and_query(chroma_client):
    docs = ["국민연금공단 이러닝시스템", "한국원자력연구원 선량평가"]
    embeddings = [[0.1] * 1536, [0.9] * 1536]
    metadatas = [{"file": "a.pdf"}, {"file": "b.pdf"}]
    ids = ["id1", "id2"]

    chroma_client.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)
    results = chroma_client.query(query_embedding=[0.1] * 1536, n_results=1)

    assert len(results["documents"][0]) == 1
    assert results["documents"][0][0] == "국민연금공단 이러닝시스템"


def test_chroma_count(chroma_client):
    chroma_client.add(
        documents=["test doc"],
        embeddings=[[0.5] * 1536],
        metadatas=[{"file": "x.pdf"}],
        ids=["x1"],
    )
    assert chroma_client.count() == 1


def test_chroma_reset(chroma_client):
    chroma_client.add(
        documents=["doc"],
        embeddings=[[0.1] * 1536],
        metadatas=[{"file": "a.pdf"}],
        ids=["a1"],
    )
    chroma_client.reset()
    # reset 후 새 컬렉션으로 재생성되어 count 0
    assert chroma_client.count() == 0
