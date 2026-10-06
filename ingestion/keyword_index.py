"""Dependency-free BM25 keyword retrieval for small in-memory corpora."""

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Sequence

from .document_chunker import Chunk


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Return lowercase word-like tokens used by BM25."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class KeywordResult:
    chunk: Chunk
    score: float


class BM25Index:
    """Rank chunks with BM25, a strong lexical retrieval baseline."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self._items: list[tuple[Chunk, Counter[str]]] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 0.0

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Add chunks and update corpus statistics."""
        for chunk in chunks:
            frequencies = Counter(tokenize(chunk.text))
            self._items.append((chunk, frequencies))
            self._document_frequency.update(frequencies.keys())
        if self._items:
            self._average_length = sum(sum(frequencies.values()) for _, frequencies in self._items) / len(self._items)

    def search(self, query: str, top_k: int = 5) -> list[KeywordResult]:
        """Return lexical matches sorted by descending BM25 score."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_terms = set(tokenize(query))
        if not query_terms or not self._items or not self._average_length:
            return []

        document_count = len(self._items)
        results: list[KeywordResult] = []
        for chunk, frequencies in self._items:
            document_length = sum(frequencies.values())
            score = 0.0
            for term in query_terms:
                term_frequency = frequencies[term]
                if not term_frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_document_frequency = math.log(
                    1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * document_length / self._average_length
                )
                score += inverse_document_frequency * (term_frequency * (self.k1 + 1)) / denominator
            if score > 0.0:
                results.append(KeywordResult(chunk, score))

        results.sort(key=lambda result: (-result.score, result.chunk.chunk_id))
        return results[:top_k]

    def __len__(self) -> int:
        return len(self._items)
