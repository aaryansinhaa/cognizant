from __future__ import annotations

import json
import re
from pathlib import Path

from rag.models.document import Document

SUPPORTED_EXTENSIONS = {".txt", ".json"}
_WHITESPACE_RE = re.compile(r"\s+")


def preprocess_text(
    text: str,
    *,
    lowercase: bool = False,
    collapse_whitespace: bool = True,
    strip: bool = True,
) -> str:
    """Light cleanup that preserves meaning.

    Assessment note: do NOT strip punctuation by default. Punctuation and
    casing carry semantics ("US" vs "us", "C++", emails, code).
    """
    if text is None:
        raise TypeError("text must be a string, got None")
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

    if strip:
        text = text.strip()
    if collapse_whitespace:
        text = _WHITESPACE_RE.sub(" ", text)
    if lowercase:
        text = text.lower()
    return text


def _read_text(path: Path, encoding: str = "utf-8") -> str:
    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError as exc:
        # Fallback: replace undecodable bytes instead of silently dropping the file.
        try:
            return path.read_text(encoding=encoding, errors="replace")
        except OSError:
            raise exc


def load_document(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    preprocess: bool = True,
) -> Document:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"expected a file, got directory: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported extension {path.suffix!r} for {path}")

    if path.suffix.lower() == ".txt":
        raw = _read_text(path, encoding=encoding)
        text = preprocess_text(raw) if preprocess else raw
        return Document(id=path.stem, text=text, source=str(path))

    raw = _read_text(path, encoding=encoding)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {path}: {exc}") from exc

    if isinstance(payload, dict):
        text = payload.get("text")
        if not isinstance(text, str):
            raise ValueError(f"JSON object in {path} must contain string field 'text'")
        doc_id = str(payload.get("id", path.stem))
        source = str(payload.get("source", path))
        metadata = {k: v for k, v in payload.items() if k not in {"id", "text", "source"}}
        if preprocess:
            text = preprocess_text(text)
        return Document(id=doc_id, text=text, source=source, metadata=metadata)

    raise ValueError(f"JSON in {path} must be an object with a 'text' field")


def load_documents(
    source: str | Path,
    *,
    encoding: str = "utf-8",
    preprocess: bool = True,
    recursive: bool = False,
) -> list[Document]:
    """Load one file or every supported file in a directory.

    Ordering is deterministic: sorted by file path.
    Duplicate file stems are allowed as long as full paths differ; IDs use
    the file stem, so colliding stems get a numeric suffix.
    """
    source = Path(source)
    if source.is_file():
        return [load_document(source, encoding=encoding, preprocess=preprocess)]
    if not source.is_dir():
        raise FileNotFoundError(f"missing path: {source}")

    pattern = "**/*" if recursive else "*"
    files = sorted(
        p
        for p in source.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        return []

    documents: list[Document] = []
    seen_ids: dict[str, int] = {}
    for path in files:
        doc = load_document(path, encoding=encoding, preprocess=preprocess)
        count = seen_ids.get(doc.id, 0)
        seen_ids[doc.id] = count + 1
        if count:
            doc = Document(
                id=f"{doc.id}_{count}",
                text=doc.text,
                source=doc.source,
                metadata=doc.metadata,
            )
        documents.append(doc)
    return documents
