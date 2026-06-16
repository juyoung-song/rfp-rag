import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from main import app
from api.dependencies import get_embedder, get_retriever, get_generator, get_chroma_client, get_ingest_pipeline
from pipeline.retriever.base import RetrievedChunk
from pipeline.generator.base import GeneratorResponse, Message


# ── 공통 mock 객체 ──────────────────────────────────────────

def _mock_embedder():
    m = MagicMock()
    m.embed.return_value = [0.1] * 1536
    return m


def _mock_retriever():
    m = MagicMock()
    m.retrieve.return_value = [
        RetrievedChunk(text="관련 내용입니다.", metadata={"source": "a.pdf"}, score=0.9)
    ]
    return m


def _mock_generator():
    m = MagicMock()
    m.generate.return_value = GeneratorResponse(
        answer="테스트 답변",
        history=[
            Message(role="user", content="질문"),
            Message(role="assistant", content="테스트 답변"),
        ],
    )
    return m


def _mock_chroma():
    m = MagicMock()
    m.count.return_value = 42
    return m


# ── /health ──────────────────────────────────────────────────

def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


# ── POST /chat ────────────────────────────────────────────────

@pytest.fixture
def chat_client():
    app.dependency_overrides[get_embedder] = _mock_embedder
    app.dependency_overrides[get_retriever] = _mock_retriever
    app.dependency_overrides[get_generator] = _mock_generator
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_chat_success(chat_client):
    res = chat_client.post("/chat", json={"query": "사업 예산은?"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "테스트 답변"
    assert len(body["sources"]) == 1
    assert body["sources"][0]["score"] == 0.9


def test_chat_empty_query(chat_client):
    res = chat_client.post("/chat", json={"query": "  "})
    assert res.status_code == 400


def test_chat_returns_history(chat_client):
    res = chat_client.post("/chat", json={"query": "질문"})
    history = res.json()["history"]
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


# ── GET /documents/status ─────────────────────────────────────

@pytest.fixture
def docs_client():
    app.dependency_overrides[get_chroma_client] = _mock_chroma
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_documents_status(docs_client):
    res = docs_client.get("/documents/status")
    assert res.status_code == 200
    assert res.json()["total_chunks"] == 42


def test_documents_reset(docs_client):
    res = docs_client.delete("/documents/reset")
    assert res.status_code == 200
    assert "초기화" in res.json()["message"]


# ── POST /ingest ──────────────────────────────────────────────

@pytest.fixture
def ingest_client(tmp_path):
    mock_pipeline = MagicMock()
    mock_pipeline.ingest_file.return_value = 15
    app.dependency_overrides[get_ingest_pipeline] = lambda: mock_pipeline
    with TestClient(app) as client:
        yield client, tmp_path
    app.dependency_overrides.clear()


def test_ingest_pdf(ingest_client):
    client, tmp_path = ingest_client
    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy")
    with dummy_pdf.open("rb") as f:
        res = client.post("/ingest", files={"file": ("test.pdf", f, "application/pdf")})
    assert res.status_code == 200
    assert res.json()["chunks_stored"] == 15


def test_ingest_non_pdf_rejected(ingest_client):
    client, tmp_path = ingest_client
    dummy = tmp_path / "doc.txt"
    dummy.write_text("hello")
    with dummy.open("rb") as f:
        res = client.post("/ingest", files={"file": ("doc.txt", f, "text/plain")})
    assert res.status_code == 400
