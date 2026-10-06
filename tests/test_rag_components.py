import json
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:  # allow running tests/run_without_pytest.py
    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return None

        def __exit__(self, t, e, tb):
            if t is None:
                raise AssertionError(f"expected {self.exc}")
            return issubclass(t, self.exc)

    class _Approx:
        def __init__(self, value, abs=1e-9):
            self.value = value
            self.abs = abs

        def __eq__(self, other):
            return abs(other - self.value) <= self.abs

    class pytest:  # type: ignore
        raises = staticmethod(lambda exc: _Raises(exc))
        approx = staticmethod(lambda value, **kw: _Approx(value, abs=kw.get("abs", 1e-9)))

from rag.ingestion.loader import load_document, load_documents, preprocess_text
from rag.models.document import Chunk, Document
from rag.preprocessing.chunker import chunk_document, chunk_text
from rag.retrieval.vector_store import InMemoryVectorStore
from rag.utils.similarity import cosine_similarity, cosine_similarity_python


def test_preprocess_collapses_whitespace():
    assert preprocess_text("  Hello\n\n  world\t") == "Hello world"


def test_preprocess_optional_lowercase():
    assert preprocess_text("US Navy", lowercase=True) == "us navy"


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "nope.txt")


def test_load_unsupported_extension(tmp_path):
    path = tmp_path / "notes.pdf"
    path.write_bytes(b"%PDF")
    with pytest.raises(ValueError):
        load_document(path)


def test_load_empty_txt(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    doc = load_document(path)
    assert doc.text == ""
    assert doc.id == "empty"


def test_load_json_and_directory(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.json").write_text(
        json.dumps({"id": "beta", "text": "beta text", "source": "x"}),
        encoding="utf-8",
    )
    docs = load_documents(tmp_path)
    assert [d.id for d in docs] == ["a", "beta"]


def test_chunk_empty_and_short():
    assert chunk_text("", chunk_size=5, overlap=1) == []
    assert chunk_text("one two", chunk_size=10, overlap=2) == ["one two"]


def test_chunk_overlap_and_ids():
    doc = Document(id="doc1", text="a b c d e f", source="s")
    chunks = chunk_document(doc, chunk_size=3, overlap=1, unit="word")
    assert [c.id for c in chunks] == ["doc1_chunk_0", "doc1_chunk_1", "doc1_chunk_2"]
    assert chunks[0].text == "a b c"
    assert chunks[1].text == "c d e"
    assert chunks[2].text == "e f"
    assert all(c.document_id == "doc1" and c.source == "s" for c in chunks)


def test_overlap_too_large():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=3, overlap=3)


def test_cosine_cases():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([0, 0], [1, 0]) == 0.0
    assert cosine_similarity_python([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cosine_similarity([1, 0], [1, 0, 0])
    with pytest.raises(ValueError):
        cosine_similarity([np.nan, 1], [1, 1])


def _chunk(i: int, text: str = "t") -> Chunk:
    return Chunk(id=f"c{i}", text=text, source="s", chunk_index=i, document_id="d")


def test_vector_store_topk_and_ties():
    store = InMemoryVectorStore()
    store.add(_chunk(0, "a"), [1.0, 0.0])
    store.add(_chunk(1, "b"), [0.9, 0.1])
    store.add(_chunk(2, "c"), [0.0, 1.0])
    results = store.search([1.0, 0.0], top_k=2)
    assert [r.chunk.id for r in results] == ["c0", "c1"]
    assert results[0].score == pytest.approx(1.0)


def test_vector_store_empty_and_k_bounds():
    store = InMemoryVectorStore()
    assert store.search([1.0, 0.0], top_k=3) == []
    store.add(_chunk(0), [1.0, 0.0])
    assert store.search([1.0, 0.0], top_k=0) == []
    assert len(store.search([1.0, 0.0], top_k=9)) == 1
    with pytest.raises(ValueError):
        store.search([1.0, 0.0], top_k=-1)


def test_alignment_and_dimension_errors():
    store = InMemoryVectorStore()
    store.add(_chunk(0), [1.0, 0.0])
    with pytest.raises(ValueError):
        store.add(_chunk(1), [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        store.add_many([_chunk(2)], np.array([[1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        store.add(_chunk(0), [0.0, 1.0])


def test_sample_docs_exist():
    root = Path(__file__).resolve().parents[1] / "data" / "sample_docs"
    docs = load_documents(root)
    assert len(docs) == 3
    assert {d.id for d in docs} >= {"embeddings", "chunking", "ingestion_notes"}
