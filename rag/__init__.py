"""Minimal from-scratch RAG package for coding-assessment practice.

No LangChain / LlamaIndex. Pipeline:

    documents -> loader -> preprocessor -> chunker
             -> embedder -> vector_store -> retriever
"""

from rag.models.document import Chunk, Document, RetrievalResult
from rag.ingestion.loader import load_document, load_documents, preprocess_text
from rag.preprocessing.chunker import chunk_document, chunk_documents
from rag.utils.similarity import cosine_similarity
from rag.retrieval.vector_store import InMemoryVectorStore

__all__ = [
    "Chunk",
    "Document",
    "RetrievalResult",
    "load_document",
    "load_documents",
    "preprocess_text",
    "chunk_document",
    "chunk_documents",
    "cosine_similarity",
    "InMemoryVectorStore",
]
