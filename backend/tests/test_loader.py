import subprocess
import pytest
from pipeline.loader.pymupdf_loader import PyMuPDFLoader
from pipeline.loader.pdfplumber_loader import PdfplumberLoader
from pipeline.loader.opendataloader_loader import OpenDataLoaderLoader


def _java_works() -> bool:
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


requires_java = pytest.mark.skipif(not _java_works(), reason="Java Runtime이 없어 ODL 실행 불가")


def test_pymupdf_loader_returns_text(sample_pdf_path):
    loader = PyMuPDFLoader()
    text = loader.load(sample_pdf_path)
    assert isinstance(text, str)
    assert len(text) > 100


def test_pymupdf_loader_text_contains_korean(sample_pdf_path):
    loader = PyMuPDFLoader()
    text = loader.load(sample_pdf_path)
    assert any('가' <= ch <= '힣' for ch in text)


def test_pdfplumber_loader_returns_text(sample_pdf_path):
    loader = PdfplumberLoader()
    text = loader.load(sample_pdf_path)
    assert isinstance(text, str)
    assert len(text) > 100


def test_pdfplumber_loader_text_contains_korean(sample_pdf_path):
    loader = PdfplumberLoader()
    text = loader.load(sample_pdf_path)
    assert any('가' <= ch <= '힣' for ch in text)


@requires_java
def test_opendataloader_returns_markdown(sample_pdf_path, tmp_path):
    loader = OpenDataLoaderLoader(output_dir=tmp_path)
    text = loader.load(sample_pdf_path)
    assert isinstance(text, str)
    assert len(text) > 100
    # markdown 포맷 확인: 헤딩(#) 또는 표(|) 포함
    assert '#' in text or '|' in text


@requires_java
def test_opendataloader_text_contains_korean(sample_pdf_path, tmp_path):
    loader = OpenDataLoaderLoader(output_dir=tmp_path)
    text = loader.load(sample_pdf_path)
    assert any('가' <= ch <= '힣' for ch in text)
