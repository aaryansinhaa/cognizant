from common import md, code


def cells():
    return [
        md(
            """
# Part 15 — Assessment simulation

Work each problem **before** scrolling to its solution. Timebox 15–25 minutes each.

Shared conventions: UTF-8, fail fast, deterministic ids `{document_id}_chunk_{i}`, cosine on raw vectors, `top_k<=0` → `[]` except where noted.

---

## Q1 — Document loader

**Statement.** Implement `load_document(path: str | Path) -> dict`.

**Signature.** `def load_document(path: str | Path) -> dict:`

**Input.** Path to a `.txt` file.

**Output.** `{"id": <stem>, "text": <stripped collapsed-whitespace text>, "source": <str path>}`

**Constraints.** `encoding="utf-8"`. Raise `FileNotFoundError` / `ValueError` for missing or unsupported suffix.

**Hidden.** empty file; extra blank lines; `.TXT` suffix; path that is a directory.

**Traps.** forgetting stem vs name; not setting encoding; using `open` without `with`.
"""
        ),
        md(
            """
<details><summary>Q1 solution</summary>

```python
def load_document(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".txt":
        raise ValueError("unsupported")
    text = re.sub(r"\\s+", " ", path.read_text(encoding="utf-8")).strip()
    return {"id": path.stem, "text": text, "source": str(path)}
```
</details>
"""
        ),
        md(
            """
## Q2 — Clean and normalize documents

**Signature.** `def preprocess_text(text: str, lowercase: bool = False) -> str`

**Expected.** Strip, collapse any whitespace to a single space. Lowercase only if flagged.

**Hidden.** tabs + newlines; already-clean text; `""`; non-str → `TypeError`.

**Traps.** removing punctuation; `text.split()` then join which also collapses but drops intentional leading/trailing — usually OK; `lower` by default.
"""
        ),
        md(
            """
<details><summary>Q2 solution</summary>

```python
def preprocess_text(text, lowercase=False):
    if not isinstance(text, str):
        raise TypeError("text")
    text = re.sub(r"\\s+", " ", text).strip()
    return text.lower() if lowercase else text
```
</details>
"""
        ),
        md(
            """
## Q3 — Chunk with overlap

**Signature.** `def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]`

**Input.** Whitespace-tokenized words. `chunk_size>0`, `0 <= overlap < chunk_size`.

**Expected.** Overlapping word windows. Last window may be shorter. Empty text → `[]`. Text shorter than size → `[text]` after a normal split-join (words separated by one space).

**Hidden.** overlap = size-1; exact multiple; one word; `overlap >= chunk_size` must raise.

**Traps.** infinite loop; `step = overlap`; character slicing when they asked words.
"""
        ),
        md(
            """
<details><summary>Q3 solution</summary>

```python
def chunk_text(text, chunk_size, overlap):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("bad sizes")
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]
    step, out, i, n = chunk_size - overlap, [], 0, len(words)
    while i < n:
        out.append(" ".join(words[i:i + chunk_size]))
        if i + chunk_size >= n:
            break
        i += step
    return out
```
</details>
"""
        ),
        md(
            """
## Q4 — Preserve metadata during chunking

**Signature.** `def chunk_document(doc: dict, chunk_size: int, overlap: int) -> list[dict]`

**Input.** `doc` has `id`, `text`, `source`, optional `metadata`.

**Expected.** Each chunk: `id`, `text`, `source`, `chunk_index`, `document_id`, `metadata` (copied).

**Hidden.** missing metadata key; shared metadata mutation (must copy); empty text → `[]`.

**Traps.** `metadata=doc["metadata"]` without copy; 1-based indexes.
"""
        ),
        md(
            """
<details><summary>Q4 solution</summary>

```python
def chunk_document(doc, chunk_size, overlap):
    meta = dict(doc.get("metadata") or {})
    return [
        {
            "id": f"{doc['id']}_chunk_{i}",
            "text": t,
            "source": doc["source"],
            "chunk_index": i,
            "document_id": doc["id"],
            "metadata": dict(meta),
        }
        for i, t in enumerate(chunk_text(doc["text"], chunk_size, overlap))
    ]
```
</details>
"""
        ),
        md(
            """
## Q5 — FastEmbed embeddings

**Signature.**
```python
class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"): ...
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
```

**Expected.** One `TextEmbedding`. Documents via `passage_embed`, query via `query_embed`. Return `float32` arrays, shape `(n, d)` and `(d,)`. Empty document list → `(0, d)`.

**Hidden.** generator exhaustion if they cache the generator; dim check.

**Traps.** constructing the model in `embed_query`; using `embed` for one side only.
"""
        ),
        md(
            """
<details><summary>Q5 solution</summary>

```python
class Embedder:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model_name)
        self.dim = self.model.embedding_size

    def embed_documents(self, texts):
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack(list(self.model.passage_embed(texts))).astype(np.float32)

    def embed_query(self, text):
        return np.asarray(next(iter(self.model.query_embed(text))), dtype=np.float32).reshape(-1)
```
</details>
"""
        ),
        md(
            """
## Q6 — In-memory vector store

**Signature.**
```python
class InMemoryVectorStore:
    def add(self, chunk: dict, embedding: np.ndarray) -> None: ...
    def add_many(self, chunks: list[dict], embeddings: np.ndarray) -> None: ...
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]: ...
```

**Output of search.** `[{"chunk": ..., "score": float}, ...]` length `min(top_k, n)`.

**Constraints.** Reject dim mismatch, duplicate ids, length mismatch, `top_k < 0`. Empty store or `top_k==0` → `[]`.

**Hidden.** k > n; ties; NaN embedding.

**Traps.** returning only scores; `zip` without length check.
"""
        ),
        md(
            """
<details><summary>Q6 solution</summary>

Use the `InMemoryVectorStore` from Part 9. Key lines: validate, `cosine_similarity` per row (or matrix), `heapq.nsmallest(k, range(n), key=lambda i: (-scores[i], chunks[i]["id"]))`.
</details>
"""
        ),
        md(
            """
## Q7 — Cosine similarity

**Signature.** `def cosine_similarity(a, b) -> float`

**Expected.** `+1` / `0` / `-1` on the textbook pairs. Zero vector → `0.0`. Dim mismatch or NaN → `ValueError`.

**Hidden.** lists vs arrays; row vectors shape `(1, d)`; very small floats.

**Traps.** integer division; forgetting to convert to float.
"""
        ),
        md(
            """
<details><summary>Q7 solution</summary>

See Part 4 `cosine_similarity`. Reshape to 1-D, check finite, norms, then `dot / (na*nb)`.
</details>
"""
        ),
        md(
            """
## Q8 — Retrieve top-k documents

**Signature.** `def retrieve(store, embedder, query: str, top_k: int) -> list[dict]`

**Expected.** Embed query once, `store.search(...)`.

**Hidden.** empty query string; `top_k` larger than corpus.

**Traps.** embedding every stored document again; reloading the model.
"""
        ),
        md(
            """
<details><summary>Q8 solution</summary>

```python
def retrieve(store, embedder, query, top_k):
    return store.search(embedder.embed_query(query), top_k=top_k)
```
</details>
"""
        ),
        md(
            """
## Q9 — Malformed documents

**Signature.** `def load_documents(directory: Path) -> list[dict]`

**Expected.** Load all `.txt` and well-formed `.json` objects with `text`. **Skip or raise?** Pick one and document it. Recommended for OAs: **raise** with the path in the message (fail fast). If they say "best effort", collect errors and continue.

**Hidden.** empty dir → `[]`; JSON array; JSON without `text`; unreadable bytes.

**Traps.** swallowing every exception; unsorted files.
"""
        ),
        md(
            """
<details><summary>Q9 solution</summary>

```python
def load_documents(directory):
    directory = Path(directory)
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".txt", ".json"})
    return [load_document(p) for p in files]  # load_document already validates
```
</details>
"""
        ),
        md(
            """
## Q10 — Refactor a monolith

**Statement.** Given a 80-line script that reads files, chunks, embeds, and searches in one function, split it into the Part 11 layout. `main` may only orchestrate.

**Hidden.** graders import `from rag.preprocessing.chunker import chunk_text` — that name must exist.

**Traps.** circular imports; putting FastEmbed inside the chunker "to be helpful".
"""
        ),
        md(
            """
<details><summary>Q10 solution</summary>

Copy the `rag/` package boundaries. `models` at the bottom. No downward imports.
</details>
"""
        ),
        md(
            """
## Q11 — Debug broken RAG code

Find the bugs (do not rewrite from scratch):

```python
model = None
def embed(text):
    global model
    model = TextEmbedding()
    return model.embed(text)          # generator, and reload every call

def search(q, docs, vecs):
    qv = embed(q)
    best = 0
    for i, v in zip(docs, vecs):      # zip not strict; qv is a generator
        if cosine(qv, v) > best:      # compares generator to array
            best = i
    return best                       # returns last doc object, not ranked list
```

**Traps.** fixing only one of: reload, generator, zip, top-1 instead of top-k, no metadata.
"""
        ),
        md(
            """
<details><summary>Q11 solution</summary>

1. Construct `TextEmbedding` once.
2. `np.asarray(next(iter(model.query_embed(q))))`.
3. `passage_embed` for `vecs`.
4. `zip(..., strict=True)` + length check.
5. Return `top_k` list of `{chunk, score}`.
</details>
"""
        ),
        md(
            """
## Q12 — Fix metadata / embedding alignment

```python
chunks.sort(key=lambda c: len(c["text"]))
index = list(zip(chunks, embeddings))
```

Embeddings were computed **before** the sort. Repair without recomputing if possible.

**Hidden.** duplicate lengths; empty chunks already removed from one list only.
"""
        ),
        md(
            """
<details><summary>Q12 solution</summary>

Never sort one side. If you must sort, zip first:

```python
paired = list(zip(chunks, embeddings, strict=True))
paired.sort(key=lambda p: len(p[0]["text"]))
```

Better: key by `chunk["id"]` on both sides.
</details>
"""
        ),
        md(
            """
## Q13 — Fix incorrect top-k

```python
scores = [cosine(q, v) for v in vecs]
return [chunks[i] for i in np.argsort(scores)[:k]]
```

**Traps.** `argsort` is ascending (worst first). No tie-break. `k > n` is OK here but `k=0` returns `[]` which is fine. Negative scores exist (cosine).
"""
        ),
        md(
            """
<details><summary>Q13 solution</summary>

```python
order = sorted(range(len(scores)), key=lambda i: (-scores[i], chunks[i]["id"]))
return [{"chunk": chunks[i], "score": scores[i]} for i in order[:min(k, len(scores))]]
```
</details>
"""
        ),
        md(
            """
## Q14 — Edge-case gauntlet

Implement `search(query_vec, matrix, chunks, top_k)` that survives:

- empty matrix
- `top_k` in `{0, 1, n, n+5, -1}` (`-1` raises)
- zero row in `matrix`
- dim mismatch
- two identical rows (deterministic id order)
- `chunks` / `matrix` length mismatch

No solution here on purpose — this **is** the drill. Your Part 9 store should already pass. If not, fix it.

---

## Exam-day checklist (memorize)

```
[ ] Path + encoding="utf-8" + sorted(glob)
[ ] preprocess = strip + collapse whitespace only
[ ] chunk: step = size - overlap; raise if overlap >= size
[ ] id = f"{doc_id}_chunk_{i}"; copy metadata
[ ] TextEmbedding once; passage_embed / query_embed; list() the generator
[ ] assert len(chunks) == len(embeddings)
[ ] cosine: dim check, zero → 0.0, no NaN
[ ] top_k: min(k,n); key = (-score, id); heap or sort
[ ] modules: models → utils → ingest/chunk/embed/store → main
```

You are ready when you can implement that list on a whiteboard in 20 minutes.
"""
        ),
    ]
