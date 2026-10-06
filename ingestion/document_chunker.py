from dataclasses import dataclass
import re
from typing import Any, Dict, List


@dataclass
class Chunk:
    chunk_id: str
    text: str
    token_count: int
    metadata: Dict[str, Any]

class DocumentChunker:
    def __init__(self, max_tokens: int = 256, chunk_overlap: int = 32):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if chunk_overlap < 0 or chunk_overlap >= max_tokens:
            raise ValueError("chunk_overlap must be between 0 and max_tokens - 1")

        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " "]

    def _count_tokens(self, text: str) -> int:
        return len(text.strip().split())

    def _split_once(self, text: str, start: int, separator: str) -> List[tuple[str, int]]:
        """Split while retaining the source offset of every piece."""
        pieces: List[tuple[str, int]] = []
        cursor = 0
        while True:
            separator_start = text.find(separator, cursor)
            if separator_start == -1:
                if cursor < len(text):
                    pieces.append((text[cursor:], start + cursor))
                break
            piece_end = separator_start + len(separator)
            pieces.append((text[cursor:piece_end], start + cursor))
            cursor = piece_end
        return pieces

    def _recursive_split(self, text: str, start: int, separator_index: int = 0) -> List[tuple[str, int]]:
        if self._count_tokens(text) <= self.max_tokens:
            return [(text, start)]

        if separator_index >= len(self.separators):
            tokens = list(re.finditer(r"\S+", text))
            return [
                (text[token_start:token_end], start + token_start)
                for token_start, token_end in ((token.start(), token.end()) for token in tokens)
            ]

        separator = self.separators[separator_index]
        pieces = self._split_once(text, start, separator)
        if len(pieces) == 1:
            return self._recursive_split(text, start, separator_index + 1)

        result: List[tuple[str, int]] = []
        for piece, piece_start in pieces:
            if piece.strip():
                result.extend(self._recursive_split(piece, piece_start, separator_index + 1))
        return result

    def _pack_with_overlap(self, pieces: List[tuple[str, int]]) -> List[tuple[str, int, int]]:
        tokens: List[tuple[str, int, int]] = []
        for piece, piece_start in pieces:
            tokens.extend(
                (match.group(), piece_start + match.start(), piece_start + match.end())
                for match in re.finditer(r"\S+", piece)
            )

        packed: List[tuple[str, int, int]] = []
        index = 0
        while index < len(tokens):
            end = min(index + self.max_tokens, len(tokens))
            selected = tokens[index:end]
            packed.append((" ".join(token[0] for token in selected), selected[0][1], selected[-1][2]))
            if end == len(tokens):
                break
            index = end - self.chunk_overlap
        return packed

    def chunk_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Splits `text` into chunks respecting `self.max_tokens` and `self.chunk_overlap`,
        recording character start/end offsets and maintaining semantic hierarchy.
        """
        if not doc_id:
            raise ValueError("doc_id must not be empty")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            return []

        pieces = self._recursive_split(text, 0)
        chunks: List[Chunk] = []
        for index, (chunk_text, start_char, end_char) in enumerate(
            self._pack_with_overlap(pieces)
        ):
            chunk_metadata = dict(metadata)
            chunk_metadata.update(
                chunk_index=index,
                start_char=start_char,
                end_char=end_char,
            )
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_{index}",
                    text=chunk_text,
                    token_count=self._count_tokens(chunk_text),
                    metadata=chunk_metadata,
                )
            )
        return chunks

        

