"""Load and normalize plain-text documents.

The loader deliberately supports only ``.txt`` files. Keeping file I/O separate
from chunking makes the pipeline easy to test and easy to adapt in an interview.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator


@dataclass(frozen=True)
class TextDocument:
    """The text and source metadata needed by the next pipeline stage."""

    doc_id: str
    text: str
    metadata: dict[str, str]


def normalize_text(text: str) -> str:
    """Normalize whitespace without changing the order of the words."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _iter_text_files(directory: Path) -> Iterator[Path]:
    yield from sorted(path for path in directory.rglob("*.txt") if path.is_file())


def load_text_documents(directory: str | Path, *, encoding: str = "utf-8") -> list[TextDocument]:
    """Read every ``.txt`` file recursively, returning deterministic ordering."""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    documents: list[TextDocument] = []
    for path in _iter_text_files(root):
        documents.append(
            TextDocument(
                doc_id=path.relative_to(root).as_posix(),
                text=normalize_text(path.read_text(encoding=encoding)),
                metadata={"source": path.name, "path": path.relative_to(root).as_posix()},
            )
        )
    return documents
