"""Composition root for the text ingestion and retrieval workflow."""

from dataclasses import dataclass
from pathlib import Path

from .document_chunker import Chunk, DocumentChunker
from .embeddings import EmbeddingModel
from .keyword_index import BM25Index
from .reranker import HybridReranker
from .text_loader import load_text_documents
from .vector_index import InMemoryVectorIndex
from .vector_index import SearchResult


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    metadata: dict[str, object]


class RAGPipeline:
    """Load, chunk, embed, index, and retrieve text documents."""

    def __init__(
        self,
        embedder: EmbeddingModel,
        *,
        chunker: DocumentChunker | None = None,
    ) -> None:
        self.chunker = chunker or DocumentChunker()
        self.index = InMemoryVectorIndex(embedder)
        self.keyword_index = BM25Index()
        self.chunks: list[Chunk] = []

    def ingest_directory(self, directory: str | Path) -> int:
        """Ingest all text files and return the number of chunks added."""
        new_chunks: list[Chunk] = []
        for document in load_text_documents(directory):
            new_chunks.extend(
                self.chunker.chunk_document(document.doc_id, document.text, document.metadata)
            )
        self.index.add(new_chunks)
        self.keyword_index.add(new_chunks)
        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(result.chunk.text, result.score, result.chunk.metadata)
            for result in self.index.search(query, top_k=top_k)
        ]

    def retrieve_keyword(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(result.chunk.text, result.score, result.chunk.metadata)
            for result in self.keyword_index.search(query, top_k=top_k)
        ]

    def retrieve_reranked(self, query: str, top_k: int = 5, candidate_k: int = 20) -> list[RetrievedChunk]:
        """Retrieve vector candidates, then apply a transparent lexical reranker."""
        candidates: list[SearchResult] = self.index.search(query, top_k=candidate_k)
        reranked = HybridReranker().rerank(query, candidates, top_k=top_k)
        return [
            RetrievedChunk(result.chunk.text, result.score, result.chunk.metadata)
            for result in reranked
        ]
