"""Embedding interfaces and adapters.

FastEmbed is optional so the rest of the repository can be studied and tested
without downloading a model. The tests use ``HashEmbedder`` only as a
repeatable stand-in; it is not a semantic embedding model.
"""

from hashlib import sha256
import math
from typing import Protocol, Sequence


class EmbeddingModel(Protocol):
    """Minimal dependency-injection contract used by the vector index."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class FastEmbedder:
    """Adapter around FastEmbed's ``TextEmbedding`` model."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise ImportError(
                "FastEmbed is optional. Install it with `pip install fastembed`."
            ) from error
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(list(texts))]


class HashEmbedder:
    """Small deterministic embedder for unit tests and offline demonstrations."""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in text.lower().split():
                digest = sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                vector[index] += 1.0 if digest[4] % 2 else -1.0
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors
