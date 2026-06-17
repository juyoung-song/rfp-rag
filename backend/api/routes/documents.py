from fastapi import APIRouter, Depends
from pydantic import BaseModel
from api.dependencies import get_chroma_client
from db.chroma_client import ChromaClient

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentsStatusResponse(BaseModel):
    total_chunks: int


@router.get("/status", response_model=DocumentsStatusResponse)
def documents_status(client: ChromaClient = Depends(get_chroma_client)):
    return DocumentsStatusResponse(total_chunks=client.count())


@router.delete("/reset")
def documents_reset(client: ChromaClient = Depends(get_chroma_client)):
    client.reset()
    return {"message": "벡터 DB가 초기화되었습니다."}
