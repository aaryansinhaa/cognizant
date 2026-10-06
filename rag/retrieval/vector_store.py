from __future__ import annotations

import heapq
from typing import Literal

import numpy as np

from rag.models.document import Chunk, RetrievalResult
from rag.utils.similarity import cosine_similarity_matrix


class InMemoryVectorStore:
    """Brute-force cosine index. Complexity of search: O(n * d + n log k)."""

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._embeddings: list[np.ndarray] = []
        self._ids: set[str] = set()
        self._matrix: np.ndarray | None = None
        self._dirty = True

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunk: Chunk, embedding) -> None:
        embedding = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if embedding.size == 0:
            raise ValueError("embedding must be non-empty")
        if not np.isfinite(embedding).all():
            raise ValueError(f"invalid values in embedding for {chunk.id}")
        if self._embeddings and embedding.shape != self._embeddings[0].shape:
            raise ValueError(
                f"dimension mismatch for {chunk.id}: "
                f"{embedding.shape} vs {self._embeddings[0].shape}"
            )
        if chunk.id in self._ids:
            raise ValueError(f"duplicate chunk id: {chunk.id}")
        self._chunks.append(chunk)
        self._embeddings.append(embedding)
        self._ids.add(chunk.id)
        self._dirty = True

    def add_many(self, chunks: list[Chunk], embeddings) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2-D array")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"alignment error: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
            )
        for chunk, vector in zip(chunks, embeddings):
            self.add(chunk, vector)

    def _materialize(self) -> np.ndarray:
        if self._dirty:
            if not self._embeddings:
                self._matrix = np.empty((0, 0), dtype=np.float64)
            else:
                self._matrix = np.vstack(self._embeddings)
            self._dirty = False
        return self._matrix

    def search(
        self,
        query_embedding,
        *,
        top_k: int = 5,
        method: Literal["heap", "sort"] = "heap",
    ) -> list[RetrievalResult]:
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if top_k == 0 or not self._chunks:
            return []

        matrix = self._materialize()
        scores = cosine_similarity_matrix(query_embedding, matrix)
        k = min(top_k, len(self._chunks))

        if method == "sort":
            order = sorted(
                range(len(scores)),
                key=lambda i: (-scores[i], self._chunks[i].id),
            )
            chosen = order[:k]
        elif method == "heap":
            # Smallest heap of (-score, chunk_id, index) inverted via nlargest
            # Tie-break: higher score wins, then lexicographically smaller id.
            chosen = heapq.nsmallest(
                k,
                range(len(scores)),
                key=lambda i: (-scores[i], self._chunks[i].id),
            )
        else:
            raise ValueError("method must be 'heap' or 'sort'")

        return [
            RetrievalResult(chunk=self._chunks[i], score=float(scores[i]))
            for i in chosen
        ]

    def top_k(self, query_embedding, k: int = 5) -> list[RetrievalResult]:
        return self.search(query_embedding, top_k=k)
