import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SAMPLE_PDF = DATA_DIR / "(사)벤처기업협회_2024년 벤처확인종합관리시스템 기능 고도화 용역사업 .pdf"


@pytest.fixture
def sample_pdf_path():
    assert SAMPLE_PDF.exists(), f"샘플 PDF가 없습니다: {SAMPLE_PDF}"
    return SAMPLE_PDF


@pytest.fixture
def data_dir():
    assert DATA_DIR.exists()
    return DATA_DIR
