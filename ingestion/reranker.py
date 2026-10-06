"""Small reranking strategies that operate on an initial candidate set."""

from typing import Protocol, Sequence

from .document_chunker import Chunk
from .vector_index import SearchResult
from .keyword_index import tokenize


class Reranker(Protocol):
    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int = 5) -> list[SearchResult]:
        ...


class HybridReranker:
    """Combine an existing retrieval score with normalized keyword overlap.

    This is intentionally transparent for interview practice. Production
    rerankers may call a cross-encoder or another learned scoring model.
    """

    def __init__(self, lexical_weight: float = 0.35) -> None:
        if not 0 <= lexical_weight <= 1:
            raise ValueError("lexical_weight must be between 0 and 1")
        self.lexical_weight = lexical_weight

    @staticmethod
    def _overlap(query: str, chunk: Chunk) -> float:
        query_terms = set(tokenize(query))
        chunk_terms = set(tokenize(chunk.text))
        if not query_terms:
            return 0.0
        return len(query_terms & chunk_terms) / len(query_terms)

    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not results:
            return []
        max_score = max(result.score for result in results)
        min_score = min(result.score for result in results)
        score_range = max_score - min_score
        ranked: list[tuple[float, SearchResult]] = []
        for result in results:
            normalized_vector_score = (
                (result.score - min_score) / score_range if score_range else 1.0
            )
            combined_score = (
                (1 - self.lexical_weight) * normalized_vector_score
                + self.lexical_weight * self._overlap(query, result.chunk)
            )
            ranked.append((combined_score, result))
        ranked.sort(key=lambda item: (-item[0], item[1].chunk.chunk_id))
        return [result for _, result in ranked[:top_k]]
