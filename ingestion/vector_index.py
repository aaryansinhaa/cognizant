"""A tiny in-memory cosine-similarity vector index."""

from dataclasses import dataclass
import math
from typing import Sequence

from .document_chunker import Chunk
from .embeddings import EmbeddingModel


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and its similarity score."""

    chunk: Chunk
    score: float


class InMemoryVectorIndex:
    """Store vectors in Python lists and scan them for top-k cosine matches."""

    def __init__(self, embedder: EmbeddingModel) -> None:
        self.embedder = embedder
        self._items: list[tuple[Chunk, list[float]]] = []
        self._dimensions: int | None = None

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            raise ValueError("vectors must have the same dimension")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        dot_product = sum(a * b for a, b in zip(left, right))
        return dot_product / (left_norm * right_norm)

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Embed and append chunks to the index."""
        if not chunks:
            return
        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedder returned the wrong number of vectors")
        if not vectors:
            raise ValueError("embedder returned no vectors")
        dimensions = len(vectors[0])
        if not dimensions or any(len(vector) != dimensions for vector in vectors):
            raise ValueError("embedder returned inconsistent vector dimensions")
        if self._dimensions is not None and dimensions != self._dimensions:
            raise ValueError("new vectors do not match the index dimension")
        self._dimensions = dimensions
        self._items.extend(zip(chunks, vectors))

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return at most ``top_k`` results, highest score first."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip() or not self._items:
            return []
        query_vector = self.embedder.embed([query])[0]
        scored = [
            SearchResult(chunk, self._cosine_similarity(query_vector, vector))
            for chunk, vector in self._items
        ]
        scored.sort(key=lambda result: (-result.score, result.chunk.chunk_id))
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._items)
