import pytest
from pipeline.loader.pymupdf_loader import PyMuPDFLoader


def test_pymupdf_loader_returns_text(sample_pdf_path):
    loader = PyMuPDFLoader()
    text = loader.load(sample_pdf_path)
    assert isinstance(text, str)
    assert len(text) > 100


def test_pymupdf_loader_text_contains_korean(sample_pdf_path):
    loader = PyMuPDFLoader()
    text = loader.load(sample_pdf_path)
    assert any('가' <= ch <= '힣' for ch in text)
