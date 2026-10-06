# Text loader lesson

## Job

`text_loader.py` owns file discovery and text normalization. It supports `.txt` recursively and returns `TextDocument` objects. A loader should not chunk or embed because those concerns need different tests and may change independently.

## Important decisions

- `Path.rglob("*.txt")` finds nested text files.
- Sorting paths makes ingestion order deterministic.
- `relative_to(root).as_posix()` gives a portable document ID.
- `normalize_text` converts Windows line endings, collapses repeated spaces, trims lines, and limits large blank-line runs.
- Metadata retains both the visible filename and relative path.

## Interview questions

**Why normalize before chunking?** It removes accidental formatting differences and makes approximate token counts more predictable.

**What can go wrong?** Aggressive normalization can remove meaningful formatting. In a production system, preserve raw text and store normalized text separately when layout matters.

**Why only `.txt`?** This study module has a deliberate boundary. PDF parsing is a separate concern and is not part of this implementation.
