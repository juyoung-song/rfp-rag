import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from pydantic import BaseModel
from api.dependencies import get_ingest_pipeline
from pipeline.ingest_pipeline import IngestPipeline

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestResponse(BaseModel):
    filename: str
    chunks_stored: int


@router.post("", response_model=IngestResponse)
def ingest(
    file: UploadFile = File(...),
    pipeline: IngestPipeline = Depends(get_ingest_pipeline),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / file.filename
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        count = pipeline.ingest_file(tmp_path)

    return IngestResponse(filename=file.filename, chunks_stored=count)
