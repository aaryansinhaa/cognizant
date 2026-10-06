from __future__ import annotations

from typing import Iterable

import numpy as np

from rag.models.document import Chunk


class Embedder:
    """Thin wrapper around FastEmbed.

    Current FastEmbed API (verified against qdrant/fastembed):

        TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        model.embed(texts)          -> generator of np.ndarray
        model.passage_embed(texts)  -> generator, document/passage prefix
        model.query_embed(query)    -> generator, query prefix
        model.embedding_size        -> int

    Load the model ONCE. Reuse it for every batch.
    Always materialize the generator with list() / np.vstack.
    Use passage_embed for chunks and query_embed for queries so prefixes match
    the model's retrieval training (BGE / E5 families).
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str = DEFAULT_MODEL, **kwargs):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name, **kwargs)
        self.dim = self.model.embedding_size

    def embed_documents(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 256,
    ) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = list(self.model.passage_embed(texts, batch_size=batch_size))
        matrix = np.vstack(vectors).astype(np.float32, copy=False)
        self._assert_dims(matrix)
        return matrix

    def embed_query(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError("query must be a string")
        vector = next(iter(self.model.query_embed(text)))
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim:
            raise ValueError(
                f"query embedding dim {vector.shape[0]} != model dim {self.dim}"
            )
        return vector

    def embed_chunks(self, chunks: list[Chunk], *, batch_size: int = 256) -> np.ndarray:
        return self.embed_documents((c.text for c in chunks), batch_size=batch_size)

    def _assert_dims(self, matrix: np.ndarray) -> None:
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(
                f"expected shape (n, {self.dim}), got {matrix.shape}"
            )
