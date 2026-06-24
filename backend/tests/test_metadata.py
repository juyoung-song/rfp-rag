import pytest
from pipeline.metadata_loader import MetadataLoader


def test_metadata_loader_returns_dataframe(data_dir):
    loader = MetadataLoader(data_dir / "data_list.csv")
    df = loader.load()
    assert len(df) > 0
    assert "사업명" in df.columns
    assert "발주 기관" in df.columns
    assert "파일명" in df.columns


def test_metadata_loader_get_by_filename(data_dir):
    loader = MetadataLoader(data_dir / "data_list.csv")
    df = loader.load()
    first_filename = df.iloc[0]["파일명"]
    meta = loader.get_by_filename(first_filename)
    assert meta is not None
    assert meta["파일명"] == first_filename


def test_metadata_loader_returns_none_for_unknown(data_dir):
    loader = MetadataLoader(data_dir / "data_list.csv")
    assert loader.get_by_filename("존재하지않는파일.pdf") is None


def test_metadata_loader_matches_across_extensions(data_dir):
    """CSV가 .hwp, 실제 파일이 .pdf여도 stem이 같으면 매칭되어야 함."""
    loader = MetadataLoader(data_dir / "data_list.csv")
    df = loader.load()
    hwp_file = df[df["파일명"].str.endswith(".hwp")].iloc[0]["파일명"]
    pdf_file = hwp_file.replace(".hwp", ".pdf")
    meta = loader.get_by_filename(pdf_file)
    assert meta is not None, f".hwp→.pdf 변환 후 매칭 실패: {pdf_file}"


def test_metadata_loader_matches_nfd_filename(data_dir):
    """macOS 파일시스템(NFD)으로 읽은 파일명도 CSV(NFC)와 매칭되어야 함."""
    import unicodedata
    from pathlib import Path as P
    loader = MetadataLoader(data_dir / "data_list.csv")
    df = loader.load()
    hwp_file = df[df["파일명"].str.endswith(".hwp")].iloc[0]["파일명"]
    # macOS glob이 돌려주는 NFD 형식으로 파일명 변환
    nfd_pdf = unicodedata.normalize("NFD", hwp_file.replace(".hwp", ".pdf"))
    assert not unicodedata.is_normalized("NFC", nfd_pdf)  # 전제: NFD임을 확인
    meta = loader.get_by_filename(nfd_pdf)
    assert meta is not None, f"NFD 파일명 매칭 실패: {nfd_pdf[:30]}"


def test_metadata_loader_matches_filename_with_trailing_space(data_dir):
    """CSV 파일명에 공백이 있어도 PDF 파일명과 매칭되어야 함 (예: '시스템 .hwp' ↔ '시스템.pdf')."""
    loader = MetadataLoader(data_dir / "data_list.csv")
    # 케빈랩 파일: CSV에 '...시스템 .hwp' (공백 포함), PDF는 '...시스템.pdf' (공백 없음)
    meta = loader.get_by_filename(
        "케빈랩 주식회사_평택시 강소형 스마트시티 AI 기반의 영상감시 시스템.pdf"
    )
    assert meta is not None, "trailing space 포함 CSV 파일명 매칭 실패"
    assert "사업명" in meta
