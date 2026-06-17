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
