from common import md, code


def cells():
    return [
        md(
            """
# Part 11 — Module separation

Assessment signal: they want to see that `main.py` is 20 lines of wiring, not 400 lines of mixed concerns.

```
rag/
├── main.py
├── ingestion/loader.py
├── preprocessing/chunker.py
├── embeddings/embedder.py
├── retrieval/vector_store.py
├── models/document.py
└── utils/similarity.py
```

| Module | Responsibility | Knows about |
|--------|----------------|-------------|
| `models` | records only | nothing above |
| `utils.similarity` | math | NumPy |
| `ingestion` | files → Document | models, pathlib, json |
| `preprocessing` | Document → Chunk | models |
| `embeddings` | text → vectors | FastEmbed, models.Chunk |
| `retrieval` | store + search | models, similarity |
| `main` | wire the graph | all of the above |

**Dependency flow (one direction):**

```
main → ingestion → models
     → preprocessing → models
     → embeddings → models, fastembed
     → retrieval → models, utils
     → utils
```

`chunker` must not import `Embedder`. `vector_store` must not import `TextEmbedding`. `loader` must not import the store.

## Imports

```python
# good
from rag.preprocessing.chunker import chunk_documents
from rag.models.document import Document

# bad
from rag.main import MiniRAG   # lower layer importing app
from rag.embeddings.embedder import Embedder  # inside chunker.py
```

Circular import typical cause: `document.py` imports `vector_store` for a type hint and `vector_store` imports `Chunk`. Fix: keep types in `models/`.

## Classes vs functions

| Use a function | Use a class |
|----------------|-------------|
| `preprocess_text` | `Embedder` (heavy model) |
| `chunk_text` | `InMemoryVectorStore` (growing state) |
| `cosine_similarity` | `MiniRAG` (facade) |

## BAD architecture

```python
# monolith.py
model = TextEmbedding()
for path in glob("*.txt"):
    text = open(path).read().lower()
    for i in range(0, len(text), 50):
        v = list(TextEmbedding().embed([text[i:i+50]]))[0]  # reload + no overlap + no id
        db.append((path, v))
```

Problems: model in a loop, no metadata, no overlap, locale encoding, lowercase, list-of-tuples you cannot test.

## GOOD architecture

Each box has tests. `main` only calls `ingest` and `retrieve`.

The repo already contains this package under `rag/`. Read those files after you can rewrite them from scratch.
"""
        ),
        md(
            """
# Part 12 — Experienced RAG engineer mistakes

1. **Document / chunk / embedding misalignment** — You sorted chunks by length but not vectors. Retrieval cites the wrong paragraph. *Why:* silent `zip` of unequal lists. **Fix:** `strict=True` + length assert.

2. **Recreating the embedding model** — 2–10s × n files. *Why:* constructor is the expensive part. **Fix:** one `Embedder` in `main`.

3. **Different models for queries and documents** — dimensions may match by luck; geometry will not. *Why:* cosine in two incompatible spaces is meaningless.

4. **Wrong vector dimensions** — 384 vs 768. Crash if you check; silent garbage if you truncate.

5. **Incorrect normalization** — double-normalizing or normalizing only queries. Cosine is scale-invariant, so extra L2 usually OK; **inconsistent** extra scaling plus dot-product-as-score (forgetting to divide) is not.

6. **Zero-vector cosine** — `0/0 → NaN`. NaNs sort to the top or bottom arbitrarily. **Fix:** return 0 or skip the row; reject NaN on insert.

7. **Arbitrary tie results** — hidden tests expect a stable order. **Fix:** `(-score, chunk_id)`.

8. **Losing source during chunking** — you cannot cite. Automatic fail on "return metadata" rubrics.

9. **Duplicate chunk IDs** — later `add` overwrites or you have two rows with one key. **Fix:** reject duplicates.

10. **Incorrect overlap** — `start += overlap` or `start += chunk_size` only. Adjacent sentences vanish or infinite loop.

11. **`overlap >= chunk_size`** — window never advances. **Fix:** validate.

12. **Embedding empty chunks** — wasted slots that match oddly. **Fix:** drop empty text before embed.

13. **Swallowing file errors** — `except: continue` drops half the corpus. Tests seed a bad file on purpose.

14. **Assuming UTF-8 without saying so** — works on your machine. **Fix:** `encoding="utf-8"` always.

15. **Loading huge files entirely** — mention memory; for OA still read_text unless they say stream.

16. **Unnecessary normalization** — lowercasing legal/medical/code text.

17. **Changing meaning in preprocess** — deleting punctuation turns "don't" and "dont" into a policy decision you did not intend.

18. **Recomputing embeddings for identical text** — optional cache keyed by `(model_name, text, role)`. Do not cache across models.

19. **Sorting the entire index for top-k=3** — fine for n=500; say heap for n=1e6.

20. **Not validating `top_k`** — `top_k=-1` → `lst[:-1]` surprise, or `heapq` errors.

21. **Returning embeddings without metadata** — the caller prints floats.

22. **Mixing query/document preprocessing** — stemming the query but not docs (or the reverse).

23. **Model init inside a loop** — see (2). Same for `TextEmbedding()` in `embed_query`.

24. **Non-deterministic results** — `set` iteration order, unsorted `glob`, `dict` over ids you then take first. **Fix:** sort paths, explicit tie-break.

25. **Hidden module coupling** — chunker calls embedder "for convenience". You cannot unit-test chunking offline.

Each of these is a 30-second verbal follow-up. Prepare a one-line "why" and a one-line "fix".
"""
        ),
        md(
            """
# Part 13 — Testing / hidden cases

Write tests **before** you trust a chunker. Assessment often includes a hidden harness that looks like this.
"""
        ),
        code(
            """
# pytest-style examples (pip install pytest to run as a suite)
try:
    import pytest
except ImportError:
    class pytest:  # demo stand-in
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

        @staticmethod
        def raises(exc):
            class _C:
                def __enter__(self):
                    return None
                def __exit__(self, t, e, tb):
                    return t is not None and issubclass(t, exc)
            return _C()

        @staticmethod
        def approx(value):
            return value


def test_chunk_empty():
    assert chunk_text("", 100, 10) == []


def test_chunk_short():
    assert chunk_text("one two", 10, 2) == ["one two"]


@pytest.mark.parametrize(
    "text,size,overlap,expected",
    [
        ("a b c d e f", 3, 1, ["a b c", "c d e", "e f"]),
        ("a b c d", 2, 0, ["a b", "c d"]),
        ("hello", 2, 0, ["he", "ll", "o"]),  # if unit=char — see next
    ],
)
def test_chunk_windows(text, size, overlap, expected):
    if expected[-1] == "o":
        assert chunk_text(text, size, overlap, unit="char") == expected
    else:
        assert chunk_text(text, size, overlap) == expected


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello world", 4, 4)


def test_cosine_hidden():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    with pytest.raises(ValueError):
        cosine_similarity([1], [1, 2])


def test_store_alignment():
    s = InMemoryVectorStore()
    with pytest.raises(ValueError):
        s.add_many(
            [{"id": "a", "text": "t", "source": "s", "chunk_index": 0, "document_id": "d"}],
            np.zeros((2, 3)),
        )

print("pytest-style examples defined")
"""
        ),
        md(
            """
Companion tests that already run: `tests/test_rag_components.py`.

```bash
python -m pytest tests/test_rag_components.py -q
```

### Per-component checklist

| Component | Happy | Boundary | Empty | Invalid | Duplicate | Large | Malformed |
|-----------|-------|----------|-------|---------|-----------|-------|-----------|
| loader | txt+json dir | 1 file | empty txt | missing, `.pdf` | same stem | 1MB txt | bad json, bad utf-8 |
| preprocess | spaces | already clean | `""` | `None` | — | huge ws | — |
| chunker | overlap 1 | exact size | `""` | overlap≥size | same text two docs | 10k tokens | bad unit |
| embedder | batch of 3 | 1 text | `[]` | non-str | same text twice | batch 1k | empty string policy |
| store | top_k=2 | k=n, k>n | empty store | k<0, dim | same id | n=5k | NaN vector |
| ranker | unique scores | ties | no scores | k=0 | duplicate scores | — | — |

**Takeaway:** if you cannot name a test for empty, k, and alignment, you are not assessment-ready.
"""
        ),
    ]
