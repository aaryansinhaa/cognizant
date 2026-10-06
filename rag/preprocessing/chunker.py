from __future__ import annotations

from rag.models.document import Chunk, Document


def _validate_sizes(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be < chunk_size ({chunk_size}); "
            "otherwise the window never advances"
        )


def _windows(items: list, chunk_size: int, overlap: int) -> list[list]:
    if not items:
        return []
    if len(items) <= chunk_size:
        return [items]
    step = chunk_size - overlap
    windows = []
    start = 0
    n = len(items)
    while start < n:
        end = min(start + chunk_size, n)
        windows.append(items[start:end])
        if end == n:
            break
        start += step
    return windows


def chunk_text(
    text: str,
    *,
    chunk_size: int = 100,
    overlap: int = 20,
    unit: str = "word",
) -> list[str]:
    """Split text into overlapping windows.

    unit:
        'word' — split on whitespace (assessment default)
        'char' — raw characters
    """
    _validate_sizes(chunk_size, overlap)
    if text is None:
        raise TypeError("text must be a string")
    if not text:
        return []

    if unit == "word":
        tokens = text.split()
        joiner = " "
    elif unit == "char":
        tokens = list(text)
        joiner = ""
    else:
        raise ValueError(f"unit must be 'word' or 'char', got {unit!r}")

    return [joiner.join(window) for window in _windows(tokens, chunk_size, overlap)]


def chunk_document(
    document: Document,
    *,
    chunk_size: int = 100,
    overlap: int = 20,
    unit: str = "word",
) -> list[Chunk]:
    texts = chunk_text(
        document.text, chunk_size=chunk_size, overlap=overlap, unit=unit
    )
    chunks: list[Chunk] = []
    for index, text in enumerate(texts):
        chunks.append(
            Chunk(
                id=f"{document.id}_chunk_{index}",
                text=text,
                source=document.source,
                chunk_index=index,
                document_id=document.id,
                metadata=dict(document.metadata),
            )
        )
    return chunks


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = 100,
    overlap: int = 20,
    unit: str = "word",
) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for document in documents:
        for chunk in chunk_document(
            document, chunk_size=chunk_size, overlap=overlap, unit=unit
        ):
            if chunk.id in seen:
                raise ValueError(f"duplicate chunk id: {chunk.id}")
            seen.add(chunk.id)
            chunks.append(chunk)
    return chunks
