import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from pipeline.ingest_pipeline import IngestPipeline, _chunk_id


@pytest.fixture
def mock_loader():
    loader = MagicMock()
    loader.load.return_value = "# 제목\n\n본문 내용입니다."
    return loader


@pytest.fixture
def mock_chunker():
    chunker = MagicMock()
    chunker.split.return_value = ["청크1 텍스트 내용", "청크2 텍스트 내용"]
    return chunker


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_batch.return_value = [[0.1] * 1536, [0.2] * 1536]
    return embedder


@pytest.fixture
def mock_chroma():
    return MagicMock()


@pytest.fixture
def pipeline(mock_loader, mock_chunker, mock_embedder, mock_chroma):
    return IngestPipeline(
        loader=mock_loader,
        chunker=mock_chunker,
        embedder=mock_embedder,
        chroma_client=mock_chroma,
    )


def test_ingest_file_returns_chunk_count(pipeline, mock_chroma):
    count = pipeline.ingest_file(Path("test.pdf"))
    assert count == 2


def test_ingest_file_calls_add(pipeline, mock_chroma):
    pipeline.ingest_file(Path("test.pdf"))
    mock_chroma.add.assert_called_once()
    call_kwargs = mock_chroma.add.call_args.kwargs
    assert len(call_kwargs["documents"]) == 2
    assert len(call_kwargs["embeddings"]) == 2
    assert len(call_kwargs["ids"]) == 2


def test_ingest_file_metadata_has_source(pipeline, mock_chroma):
    pipeline.ingest_file(Path("test.pdf"))
    metadatas = mock_chroma.add.call_args.kwargs["metadatas"]
    assert all(m["source"] == "test.pdf" for m in metadatas)


def test_ingest_file_with_metadata_loader(mock_loader, mock_chunker, mock_embedder, mock_chroma):
    meta_loader = MagicMock()
    meta_loader.get_by_filename.return_value = {"발주 기관": "국민연금공단", "사업명": "이러닝"}
    pipe = IngestPipeline(
        loader=mock_loader,
        chunker=mock_chunker,
        embedder=mock_embedder,
        chroma_client=mock_chroma,
        metadata_loader=meta_loader,
    )
    pipe.ingest_file(Path("test.pdf"))
    metadatas = mock_chroma.add.call_args.kwargs["metadatas"]
    assert metadatas[0]["발주 기관"] == "국민연금공단"


def test_ingest_file_empty_chunks(mock_loader, mock_embedder, mock_chroma):
    chunker = MagicMock()
    chunker.split.return_value = []
    pipe = IngestPipeline(
        loader=mock_loader,
        chunker=chunker,
        embedder=mock_embedder,
        chroma_client=mock_chroma,
    )
    count = pipe.ingest_file(Path("empty.pdf"))
    assert count == 0
    mock_chroma.add.assert_not_called()


def test_ingest_batch(pipeline):
    results = pipeline.ingest_batch([Path("a.pdf"), Path("b.pdf")])
    assert set(results.keys()) == {"a.pdf", "b.pdf"}
    assert all(v == 2 for v in results.values())


def test_chunk_id_deterministic():
    id1 = _chunk_id("file.pdf", 0)
    id2 = _chunk_id("file.pdf", 0)
    assert id1 == id2


def test_chunk_id_unique_per_index():
    id0 = _chunk_id("file.pdf", 0)
    id1 = _chunk_id("file.pdf", 1)
    assert id0 != id1
