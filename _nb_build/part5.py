from common import md, code


def cells():
    return [
        md(
            """
# Part 5 — FastEmbed

Verified against current official docs: [FastEmbed README](https://github.com/qdrant/fastembed) and [Retrieval with FastEmbed](https://qdrant.github.io/fastembed/qdrant/Retrieval_with_FastEmbed/).

## What an embedding is

```
text  →  tokenizer + ONNX model  →  float vector in R^d  →  compare with cosine
```

The model maps meaning so that paraphrases land nearby. It is **not** a hash of characters (anagrams can be far; synonyms can be close).

- **Document / passage embedding:** vector for a chunk you will store.
- **Query embedding:** vector for the user's question.
- **Dimension `d`:** length of the vector. Default model `BAAI/bge-small-en-v1.5` has **d = 384**.
- **Same model + same method family** on both sides or scores are garbage.

## Current API (do not invent methods)

```python
from fastembed import TextEmbedding

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")  # downloads once, then caches
model.embedding_size                    # 384
model.embed(texts, batch_size=256)      # generator of np.ndarray
model.passage_embed(texts)              # generator — use for documents
model.query_embed(query)                # generator — use for queries
TextEmbedding.list_supported_models()
```

Constructor extras you may see: `cache_dir`, `threads`, `providers=["CUDAExecutionProvider"]`, `lazy_load=True`.

**`embed()` returns a generator.** Nothing runs until you iterate. Always `list(...)` or `np.vstack(list(...))`.

**Retrieval tip from official docs:** prefer `query_embed` for queries and `passage_embed` for documents. BGE/E5-style models are trained with prefixes (`query:` / `passage:`). Using raw `embed()` on both sides is inconsistent for those families.

## CPU / batching

- Load the model **once**. Construction downloads + builds an ONNX session (seconds).
- `batch_size` trades RAM for throughput. 64–256 is typical on CPU.
- `parallel>1` or `parallel=0` is for offline encoding of large corpora, not for a single query.
- Stay on CPU unless the assessment says otherwise (`pip install fastembed`, not `fastembed-gpu`).
"""
        ),
        code(
            """
# This cell talks to the real library if installed. It is safe to skip.
try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None
    print("fastembed not installed — skip live cells; read the API above")
else:
    print("fastembed import ok")
    print("default dim", TextEmbedding.get_embedding_size("BAAI/bge-small-en-v1.5"))
"""
        ),
        code(
            """
FASTEMBED_READY = False
if TextEmbedding is not None:
    try:
        embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        FASTEMBED_READY = True
        print("loaded", embed_model.model_name, "dim", embed_model.embedding_size)
    except Exception as exc:
        print("model load skipped:", type(exc).__name__, exc)
"""
        ),
        code(
            """
if FASTEMBED_READY:
    one = list(embed_model.passage_embed(["Embeddings turn text into vectors."]))[0]
    print("one document", type(one), one.shape, one.dtype)

    many_texts = [
        "Embeddings turn text into vectors.",
        "Chunking splits long documents.",
    ]
    many = list(embed_model.passage_embed(many_texts, batch_size=2))
    matrix = np.vstack(many)
    print("batch", matrix.shape)

    query = next(iter(embed_model.query_embed("What is an embedding?")))
    print("query", np.asarray(query).shape)

    # generator pitfall
    gen = embed_model.embed(["once"])
    print("first materialize", len(list(gen)))
    print("second materialize", len(list(gen)))
else:
    print("skip live embed demos")
"""
        ),
        md(
            """
## Common FastEmbed mistakes

1. **`TextEmbedding()` inside a per-document loop** — minutes of reload, possible rate/disk hits.
2. **Query with `embed`, docs with `passage_embed`** (or two different `model_name`s).
3. **Forgetting `list()`** — you store a generator, later it is exhausted, `add_many` gets 0 rows.
4. **`np.array(list(model.embed(texts)))` vs `np.vstack`** — if the generator yields 1-D arrays, `np.array` gives `(n, d)` (good). If any yield is 2-D, shapes explode. Prefer `np.vstack`.
5. **Mixing models** in one index (384-d and 768-d).
6. **Huge `batch_size` on a 4GB laptop** — OOM. Shrink batch, not the idea.
7. **Storing Python lists of lists of Python floats** — works, wastes RAM. Store `float32` arrays.
8. **Re-embedding the same chunks** on every query. Embed corpus once; embed the query per request.
9. **Empty string batches** — some models still return a vector (the embedding of "empty"). Prefer skipping empty chunks before embed.
10. **Assuming `embed(["a", "b"])` returns a `list`** — it is an iterable. `len(model.embed(...))` may fail.

## Assessment takeaway

```python
class Embedder:
    def __init__(self):
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.dim = self.model.embedding_size
    def embed_documents(self, texts):
        return np.vstack(list(self.model.passage_embed(texts)))
    def embed_query(self, text):
        return np.asarray(next(iter(self.model.query_embed(text))))
```

Reuse one instance. Materialize. Check `shape[-1] == self.dim`.
"""
        ),
    ]
