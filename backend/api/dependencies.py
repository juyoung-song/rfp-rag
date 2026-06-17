from functools import lru_cache
from config import settings
from db.chroma_client import ChromaClient
from pipeline.loader.opendataloader_loader import OpenDataLoaderLoader
from pipeline.chunker.markdown_header_chunker import MarkdownHeaderChunker
from pipeline.embedder.openai_embedder import OpenAIEmbedder
from pipeline.retriever.naive_retriever import NaiveRetriever
from pipeline.generator.openai_generator import OpenAIGenerator
from pipeline.metadata_loader import MetadataLoader
from pipeline.ingest_pipeline import IngestPipeline


@lru_cache
def get_chroma_client() -> ChromaClient:
    host = settings.chroma_host if settings.chroma_host != "localhost" else None
    return ChromaClient(
        host=host,
        collection_name=settings.chroma_collection_name,
    )


@lru_cache
def get_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder(api_key=settings.openai_api_key)


@lru_cache
def get_retriever() -> NaiveRetriever:
    return NaiveRetriever(chroma_client=get_chroma_client())


@lru_cache
def get_generator() -> OpenAIGenerator:
    return OpenAIGenerator(api_key=settings.openai_api_key)


@lru_cache
def get_ingest_pipeline() -> IngestPipeline:
    metadata_csv = settings.data_dir / "metadata.csv"
    meta_loader = MetadataLoader(metadata_csv) if metadata_csv.exists() else None
    return IngestPipeline(
        loader=OpenDataLoaderLoader(),
        chunker=MarkdownHeaderChunker(),
        embedder=get_embedder(),
        chroma_client=get_chroma_client(),
        metadata_loader=meta_loader,
    )
