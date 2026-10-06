from common import md, code


def cells():
    return [
        md(
            """
# Part 14 — Build RAG from primitives

No LangChain. No LlamaIndex. This cell is a complete system you should be able to retype.

Architecture:

```
documents → loader → preprocessor → chunker → embedder → vector store → retriever → top-k
```

The same design lives on disk as the `rag/` package (`rag/main.py` is the facade). Below is a self-contained version for study.
"""
        ),
        code(
            """
@dataclass(frozen=True)
class _Doc:
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    document_id: str


class MiniRAG:
    def __init__(self, embedder, chunk_size=40, overlap=8):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.store = InMemoryVectorStore()

    def ingest_records(self, documents: list[dict]) -> int:
        chunks = []
        for doc in documents:
            for ch in chunk_document(doc, self.chunk_size, self.overlap):
                chunks.append(ch)
        if not chunks:
            return 0
        matrix = self.embedder.embed_documents([c["text"] for c in chunks])
        if len(chunks) != len(matrix):
            raise ValueError("embedder dropped or invented rows")
        self.store.add_many(chunks, matrix)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5):
        q = self.embedder.embed_query(query)
        return self.store.search(q, top_k=top_k)


docs = load_documents(Path("data/sample_docs"))
print("loaded", [d["id"] for d in docs])

mini = MiniRAG(Embedder(model=FakeModel(16), dim=16), chunk_size=20, overlap=4)
n = mini.ingest_records(docs)
print("chunks", n)
for row in mini.retrieve("Why do we keep source metadata on chunks?", top_k=3):
    print(f"{row['score']:.3f} {row['chunk']['id']}: {row['chunk']['text'][:70]}")
"""
        ),
        md(
            """
### Swap FakeModel for FastEmbed

```python
from rag.embeddings.embedder import Embedder
from rag.main import MiniRAG

rag = MiniRAG(chunk_size=30, overlap=6)
rag.ingest("data/sample_docs")
rag.retrieve("What is cosine similarity used for?", top_k=3)
```

`python -m rag.main` runs that demo (requires `fastembed`).

### Robustness already in the package

- UTF-8 + JSON + directory load, sorted paths, colliding stems
- overlap validation
- alignment check in `add_many`
- cosine zero-vector and NaN guards
- heap top-k with id tie-break
- one `TextEmbedding` instance; `passage_embed` / `query_embed`

### What you must be able to write in 45 minutes

1. `preprocess_text` + `load_documents`
2. `chunk_text` with overlap
3. `cosine_similarity`
4. `InMemoryVectorStore.add/search`
5. A 15-line `MiniRAG` that calls them in order

If any of those five takes you more than 10 minutes, drill that file only.
"""
        ),
    ]
