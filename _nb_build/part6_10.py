from common import md, code


def cells():
    return [
        md(
            """
# Part 6 — Ingestion + preprocessing

Ingestion is a **pure function of files → Document records**. No embeddings here.

```
path/dir → read bytes → decode → parse → Document(id, text, source, metadata)
```

Start with `.txt`. The same function signature should later accept `.json` and directories without rewriting callers.
"""
        ),
        code(
            """
SUPPORTED = {".txt", ".json"}
_WS = re.compile(r"\\s+")


def preprocess_text(text, *, lowercase=False, collapse_whitespace=True, strip=True):
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if strip:
        text = text.strip()
    if collapse_whitespace:
        text = _WS.sub(" ", text)
    if lowercase:
        text = text.lower()
    return text


def load_document(path, *, encoding="utf-8", preprocess=True):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(path)
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"unsupported extension {path.suffix}")

    raw = path.read_text(encoding=encoding)
    if path.suffix.lower() == ".txt":
        text = preprocess_text(raw) if preprocess else raw
        return {"id": path.stem, "text": text, "source": str(path), "metadata": {}}

    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise ValueError("JSON must be an object with string field 'text'")
    text = preprocess_text(payload["text"]) if preprocess else payload["text"]
    return {
        "id": str(payload.get("id", path.stem)),
        "text": text,
        "source": str(payload.get("source", path)),
        "metadata": {k: v for k, v in payload.items() if k not in {"id", "text", "source"}},
    }


def load_documents(source, *, recursive=False):
    source = Path(source)
    if source.is_file():
        return [load_document(source)]
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = source.rglob("*") if recursive else source.glob("*")
    files = sorted(p for p in files if p.is_file() and p.suffix.lower() in SUPPORTED)
    docs, seen = [], {}
    for p in files:
        doc = load_document(p)
        n = seen.get(doc["id"], 0)
        seen[doc["id"]] = n + 1
        if n:
            doc = {**doc, "id": f"{doc['id']}_{n}"}
        docs.append(doc)
    return docs
"""
        ),
        code(
            """
from tempfile import TemporaryDirectory

with TemporaryDirectory() as td:
    root = Path(td)
    (root / "a.txt").write_text("Hello,\\n\\n  World!  ", encoding="utf-8")
    (root / "b.json").write_text(
        json.dumps({"id": "json1", "text": "JSON text", "topic": "demo"}),
        encoding="utf-8",
    )
    (root / "skip.md").write_text("ignored", encoding="utf-8")
    (root / "empty.txt").write_text("   \\n", encoding="utf-8")
    docs = load_documents(root)
    print([(d["id"], d["text"], d["metadata"]) for d in docs])

# Should you lowercase / strip punctuation?
#   Search / bag-of-words homework: maybe.
#   Dense embeddings: usually NO. The model was trained on natural punctuation and casing.
#   "US" vs "us", "C++", emails, code identifiers all lose signal if you smash the text.
"""
        ),
        md(
            """
### Ingestion cases you must handle

| Case | Behavior |
|------|----------|
| UTF-8 text | default |
| empty file | `text=""` (do not crash) |
| repeated whitespace / newlines | collapse if preprocess on |
| Unicode | keep; decode UTF-8 |
| malformed JSON | `ValueError` |
| missing file | `FileNotFoundError` |
| `.pdf` | `ValueError` unsupported |
| duplicate stems | suffix `_1`, `_2` |
| directory order | `sorted` paths |
| huge file | OA: read all is OK; mention streaming as a follow-up |
| encoding errors | do not silently skip; fail or `errors="replace"` **explicitly** |

**Hidden:** `load_documents` on a file vs directory; Windows `\\` vs `/` in `source` strings (use `Path`); BOM at start of UTF-8 file (`utf-8-sig`).

**Takeaway:** deterministic order, explicit encoding, metadata from the first moment, no aggressive normalize.
"""
        ),
        md(
            """
# Part 7 — Chunking

Chunking is its own component. Wrong overlap is a favorite hidden test.

```
document → chunk 0 → chunk 1 → chunk 2
```

Each chunk keeps `document_id`, `source`, `chunk_index`, `text`, and a **deterministic** `id = f"{doc_id}_chunk_{i}"`.
"""
        ),
        code(
            """
def chunk_text(text, chunk_size=100, overlap=20, unit="word"):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")
    if not text:
        return []
    tokens = text.split() if unit == "word" else list(text)
    joiner = " " if unit == "word" else ""
    if len(tokens) <= chunk_size:
        return [joiner.join(tokens)]
    step = chunk_size - overlap
    out, start, n = [], 0, len(tokens)
    while start < n:
        end = min(start + chunk_size, n)
        out.append(joiner.join(tokens[start:end]))
        if end == n:
            break
        start += step
    return out


def chunk_document(doc, chunk_size=100, overlap=20, unit="word"):
    texts = chunk_text(doc["text"], chunk_size, overlap, unit)
    return [
        {
            "id": f"{doc['id']}_chunk_{i}",
            "text": t,
            "source": doc["source"],
            "chunk_index": i,
            "document_id": doc["id"],
            "metadata": dict(doc.get("metadata") or {}),
        }
        for i, t in enumerate(texts)
    ]


print(chunk_text("a b c d e f", chunk_size=3, overlap=1))
print(chunk_text("short", chunk_size=10, overlap=2))
print(chunk_text("", chunk_size=10, overlap=2))
print(chunk_text("x" * 5, chunk_size=5, overlap=0, unit="char"))
try:
    chunk_text("hello world", chunk_size=3, overlap=3)
except ValueError as e:
    print(e)
"""
        ),
        md(
            """
### Beginner chunking mistakes

1. `step = overlap` instead of `step = chunk_size - overlap`.
2. Allowing `overlap == chunk_size` → infinite loop if you write `while start < n: start += 0`.
3. IDs like `hash(text)` — collide on duplicate paragraphs; not deterministic across processes if you use `id()`.
4. Dropping `source` so citations are impossible.
5. Character windows that split UTF-16 surrogates (Python 3 `str` is code points, usually OK; still do not slice bytes).
6. Re-joining words with `""` instead of `" "`.
7. Emitting an extra empty last chunk.
8. Using `range(0, n, chunk_size)` and forgetting overlap entirely.

| Case | Expected |
|------|----------|
| empty doc | `[]` |
| shorter than size | one chunk, original text (joined tokens) |
| exact size | one chunk |
| overlap 0 | adjacent, no shared tokens |
| overlap = size-1 | almost sliding-by-one |
| very large overlap (< size) | many chunks, still terminates |

**Takeaway:** validate sizes first, copy metadata, deterministic ids, test a 6-token toy by hand.
"""
        ),
        md(
            """
# Part 8 — Embedding pipeline

Lifecycle: construct once → embed documents in batches → embed each query with the **query** path.
"""
        ),
        code(
            """
class Embedder:
    def __init__(self, model=None, dim=None):
        # In an OA you will typically do:
        # from fastembed import TextEmbedding
        # self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        # self.dim = self.model.embedding_size
        self.model = model
        self.dim = dim

    def embed_documents(self, texts, batch_size=256):
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = list(self.model.passage_embed(texts, batch_size=batch_size))
        matrix = np.vstack(vectors).astype(np.float32, copy=False)
        if matrix.shape[1] != self.dim:
            raise ValueError("unexpected embedding width")
        return matrix

    def embed_query(self, text):
        if not isinstance(text, str):
            raise TypeError("query must be str")
        vec = np.asarray(next(iter(self.model.query_embed(text))), dtype=np.float32)
        return vec.reshape(-1)


# Offline stand-in so later cells run without downloading a model
class FakeModel:
    def __init__(self, dim=8):
        self.dim = dim

    def _vec(self, text, salt):
        rng = np.random.default_rng(abs(hash(salt + text)) % (2**32))
        v = rng.normal(size=self.dim).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v

    def passage_embed(self, texts, batch_size=256):
        for t in texts:
            yield self._vec(t, "passage:")

    def query_embed(self, query):
        yield self._vec(query, "query:")


fake = Embedder(model=FakeModel(8), dim=8)
print(fake.embed_documents([]).shape)
print(fake.embed_documents(["one chunk"]).shape)
print(fake.embed_documents(["a", "b", "a"]).shape)  # duplicates allowed, same text → same vec here
print(fake.embed_query("hello").shape)
"""
        ),
        md(
            """
**Tests to rehearse:** one chunk; many; empty list → `(0, d)`; empty string (decide: skip or embed); duplicates; large batch (does not crash); query type error.

**Memory:** `n * d * 4` bytes for float32. 100k × 384 ≈ 150 MB. Fine. 10M × 384 ≈ 15 GB. Not fine for a laptop OA.

**Takeaway:** empty-list shape, dim check, never construct the model per call.
"""
        ),
        md(
            """
# Part 9 — In-memory vector index (brute force)

No FAISS. Store parallel `chunks` + `embeddings`, search every row.

```
query → query embedding → score all rows → rank → top_k {chunk, score}
```

Complexity: **O(n · d)** for scoring + **O(n log k)** for the heap (or O(n log n) full sort).

Real vector DBs (HNSW, IVF, PQ) approximate this to sublinear n. In an OA, brute force is the correct first implementation. Mention the approximation as a follow-up sentence.
"""
        ),
        code(
            """
class InMemoryVectorStore:
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.ids = set()

    def add(self, chunk, embedding):
        vec = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if vec.size == 0 or not np.isfinite(vec).all():
            raise ValueError(f"bad embedding for {chunk['id']}")
        if self.embeddings and vec.shape != self.embeddings[0].shape:
            raise ValueError("dimension mismatch")
        if chunk["id"] in self.ids:
            raise ValueError(f"duplicate id {chunk['id']}")
        self.chunks.append(chunk)
        self.embeddings.append(vec)
        self.ids.add(chunk["id"])

    def add_many(self, chunks, embeddings):
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if len(chunks) != len(embeddings):
            raise ValueError("alignment error")
        for c, e in zip(chunks, embeddings, strict=True):
            self.add(c, e)

    def search(self, query_embedding, top_k=5, method="heap"):
        if top_k < 0:
            raise ValueError("top_k")
        if top_k == 0 or not self.chunks:
            return []
        matrix = np.vstack(self.embeddings)
        q = np.asarray(query_embedding, dtype=np.float64).reshape(-1)
        if q.shape != matrix.shape[1:]:
            # q is (d,), matrix (n, d)
            if q.size != matrix.shape[1]:
                raise ValueError("dimension mismatch")
        scores = []
        for i, row in enumerate(matrix):
            scores.append(cosine_similarity(q, row))
        k = min(top_k, len(self.chunks))
        key = lambda i: (-scores[i], self.chunks[i]["id"])
        if method == "sort":
            order = sorted(range(len(scores)), key=key)[:k]
        else:
            order = heapq.nsmallest(k, range(len(scores)), key=key)
        return [{"chunk": self.chunks[i], "score": float(scores[i])} for i in order]


store = InMemoryVectorStore()
store.add({"id": "c0", "text": "alpha", "source": "s", "chunk_index": 0, "document_id": "d"}, [1.0, 0.0])
store.add({"id": "c1", "text": "beta", "source": "s", "chunk_index": 1, "document_id": "d"}, [0.8, 0.2])
store.add({"id": "c2", "text": "gamma", "source": "s", "chunk_index": 2, "document_id": "d"}, [0.0, 1.0])
print(store.search([1.0, 0.0], top_k=2))
print(store.search([1.0, 0.0], top_k=9))   # k > n
print(store.search([1.0, 0.0], top_k=0))
print(InMemoryVectorStore().search([1.0, 0.0], top_k=3))
"""
        ),
        md(
            """
| Case | Behavior |
|------|----------|
| `top_k=1` | single best, ties broken by `chunk.id` |
| `top_k=n` | all rows, ranked |
| `top_k>n` | min(k, n) |
| `top_k=0` | `[]` |
| empty index | `[]` |
| duplicate vectors | both stored, both can return |
| duplicate ids | raise |
| zero vector | score 0 |
| dim mismatch | raise |
| NaN | raise on add |
| ties | deterministic id order |

**Takeaway:** validate k and dims; never return a vector without its chunk; O(n d) is the honest complexity.
"""
        ),
        md(
            """
# Part 10 — Retrieval and ranking

Two implementations you must be able to write from memory.
"""
        ),
        code(
            """
def top_k_sort(scores, ids, k):
    if k <= 0:
        return []
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], ids[i]))
    return order[: min(k, len(scores))]


def top_k_heap(scores, ids, k):
    if k <= 0:
        return []
    return heapq.nsmallest(min(k, len(scores)), range(len(scores)),
                           key=lambda i: (-scores[i], ids[i]))


scores = [0.2, 0.9, 0.9, 0.1]
ids = ["d", "b", "a", "c"]
print(top_k_sort(scores, ids, 2))   # indices of 0.9/'a' then 0.9/'b'
print(top_k_heap(scores, ids, 2))
print(top_k_sort([], [], 3))
print(top_k_sort(scores, ids, 0))

# When to use which
# n <= 5_000 or k ~ n  → sorted is clearer and fast enough
# n is huge and k is tiny (10) → heap
# OA with n=1000 chunks → sorted is what they expect unless they mention heapq
"""
        ),
        md(
            """
Connect back to DSA: this is **K Closest Points** with a custom key and **Top K Frequent**'s heap.

`sorted` is **stable** (equal keys keep original order). After you add `id` into the key, stability should not matter.

**Takeaway:** one `key=lambda i: (-score[i], id[i])` function shared by sort and heap.
"""
        ),
    ]
