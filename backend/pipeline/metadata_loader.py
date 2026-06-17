from pathlib import Path
import unicodedata
import pandas as pd


def _nfc_stem(filename: str) -> str:
    return unicodedata.normalize("NFC", Path(filename).stem).strip()


class MetadataLoader:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.csv_path, encoding="utf-8-sig")
        return self._df

    def get_by_filename(self, filename: str) -> dict | None:
        df = self.load()
        target = _nfc_stem(filename)
        matches = df[df["파일명"].apply(_nfc_stem) == target]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()
