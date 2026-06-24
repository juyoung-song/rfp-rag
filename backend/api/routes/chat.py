from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from api.dependencies import get_embedder, get_retriever, get_generator
from pipeline.embedder.base import BaseEmbedder
from pipeline.retriever.base import BaseRetriever
from pipeline.generator.base import BaseGenerator, Message
from config import settings

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    history: list[dict] | None = None  # [{"role": "user"|"assistant", "content": "..."}]
    top_k: int = settings.top_k


class SourceChunk(BaseModel):
    text: str
    metadata: dict
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    history: list[dict]


@router.post("", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    embedder: BaseEmbedder = Depends(get_embedder),
    retriever: BaseRetriever = Depends(get_retriever),
    generator: BaseGenerator = Depends(get_generator),
):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query가 비어 있습니다.")

    query_vec = embedder.embed(req.query)
    retrieved = retriever.retrieve(query_vec, top_k=req.top_k)

    history: list[Message] = []
    if req.history:
        history = [Message(role=h["role"], content=h["content"]) for h in req.history]

    result = generator.generate(
        query=req.query,
        context_chunks=[r.text for r in retrieved],
        history=history,
        context_metadata=[r.metadata for r in retrieved],
    )

    return ChatResponse(
        answer=result.answer,
        sources=[SourceChunk(text=r.text, metadata=r.metadata, score=r.score) for r in retrieved],
        history=[{"role": m.role, "content": m.content} for m in result.history],
    )
