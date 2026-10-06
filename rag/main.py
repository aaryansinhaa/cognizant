from __future__ import annotations

from pathlib import Path

from rag.embeddings.embedder import Embedder
from rag.ingestion.loader import load_documents
from rag.preprocessing.chunker import chunk_documents
from rag.retrieval.vector_store import InMemoryVectorStore


class MiniRAG:
    """End-to-end in-memory RAG used in Part 14 of the prep notebook."""

    def __init__(
        self,
        *,
        chunk_size: int = 40,
        overlap: int = 8,
        unit: str = "word",
        model_name: str = Embedder.DEFAULT_MODEL,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.unit = unit
        self.embedder = Embedder(model_name=model_name)
        self.store = InMemoryVectorStore()

    def ingest(self, source: str | Path) -> int:
        documents = load_documents(source)
        chunks = chunk_documents(
            documents,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            unit=self.unit,
        )
        if not chunks:
            return 0
        embeddings = self.embedder.embed_chunks(chunks)
        self.store.add_many(chunks, embeddings)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedder.embed_query(query)
        return self.store.search(query_embedding, top_k=top_k)


def demo() -> None:
    root = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
    rag = MiniRAG(chunk_size=30, overlap=6)
    n = rag.ingest(root)
    print(f"indexed {n} chunks")
    for result in rag.retrieve("What is cosine similarity used for?", top_k=3):
        print(f"{result.score:.4f}  {result.chunk.id}  {result.chunk.text[:80]!r}")


if __name__ == "__main__":
    demo()
