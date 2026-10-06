from common import md, code


def cells():
    return [
        md(
            """
# Part 2 — Python data manipulation for RAG

A RAG pipeline is a sequence of **records that must stay aligned**.

```
Document  {id, text, source}
    ↓ chunk
Chunk     {id, text, source, chunk_index, document_id}
    ↓ embed
Embedding {chunk_id, embedding}
    ↓ search
Result    {chunk, score}
```

If any join key is lost, retrieval returns the wrong passage with a confident score. That is the most common hidden-test failure.
"""
        ),
        code(
            """
document = {"id": "doc1", "text": "Embeddings turn text into vectors.", "source": "file.txt"}

chunk = {
    "id": "doc1_chunk_0",
    "text": "Embeddings turn text into vectors.",
    "source": "file.txt",
    "chunk_index": 0,
    "document_id": document["id"],
}

embedding_row = {"chunk_id": chunk["id"], "embedding": [0.11, -0.02, 0.07]}

retrieval_result = {"chunk": chunk, "score": 0.87}

# Alignment invariant: every embedding_row.chunk_id exists in chunks
chunks_by_id = {chunk["id"]: chunk}
assert embedding_row["chunk_id"] in chunks_by_id
"""
        ),
        md(
            """
## Why metadata alignment matters

The user does not want a raw vector. They want **text + citation**. The score is useless if it is attached to the wrong chunk.

Keep one join key (`chunk.id`) and copy metadata **at chunk time**, not at retrieval time from a global that may have changed.

### Parallel arrays (works, easy to desync)

```python
texts = [...]
embeddings = [...]
ids = [...]
# texts[i] MUST correspond to embeddings[i]
```

### Parallel arrays + zip(strict=True) (acceptable)

### List of records (clearest)

```python
{"chunk": chunk, "embedding": vec}
```

### Two maps (good for lookup)

```python
chunk_by_id[id] = chunk
vector_by_id[id] = vec
```

Assessments often start with parallel arrays. **Validate lengths before you embed or search.**
"""
        ),
        code(
            """
def align(chunks, embeddings):
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"alignment error: {len(chunks)} chunks vs {len(embeddings)} embeddings"
        )
    return [
        {"chunk": c, "embedding": e}
        for c, e in zip(chunks, embeddings, strict=True)
    ]


# Common mistakes (do not ship these)

def lose_document_id(doc):
    # BAD: chunk id is only an integer
    return [{"id": 0, "text": doc["text"]}]


def mix_by_sort(chunks, embeddings):
    # BAD: sort texts but not vectors
    return sorted(chunks, key=lambda c: c["text"]), embeddings


def mutate_while_iterating(chunks):
    for c in chunks:
        if not c["text"].strip():
            chunks.remove(c)   # BAD
    return chunks


def inconsistent_metadata(doc):
    # BAD: source on the doc is file.txt, chunk says unknown
    return {"id": doc["id"] + "_chunk_0", "text": doc["text"], "source": "unknown"}


def duplicate_ids(docs):
    return [{"id": "doc", "text": d} for d in docs]  # every id is 'doc'


def missing_fields(text):
    return {"text": text}  # no id, no source
"""
        ),
        md(
            """
### Hidden tests in this area

- Two files with the same stem (`/a/notes.txt`, `/b/notes.txt`) → colliding `id="notes"`.
- Empty text after preprocess → chunk list empty, later `embeddings[0]` IndexError.
- `json.dumps` then `json.loads` turns tuples into lists; your embedding is still a list, fine; your tuple keys are gone.
- Reusing one `metadata={}` dict for every chunk (Part 1 identity bug).
- Sorting by similarity and returning `texts[best_i]` after you already filtered `texts` but not `embeddings`.

### Practice

Build `index = {chunk_id: (chunk, embedding)}` from two lists and reject duplicates.

### Assessment takeaway

Write the invariant in a comment and enforce it:

`assert len(chunks) == len(embeddings)` and `chunk.id` is unique.
"""
        ),
    ]
