# Framework-free RAG study module

This repository is a small, readable implementation of the ingestion and retrieval half of a Retrieval-Augmented Generation (RAG) system. It intentionally handles **`.txt` files only**. It does not call an LLM and it does not implement PDF extraction.

## What to study

1. `ingestion/text_loader.py` reads files and normalizes whitespace.
2. `ingestion/document_chunker.py` recursively splits text and preserves overlap and offsets.
3. `ingestion/embeddings.py` defines dependency injection, a FastEmbed adapter, and a deterministic test double.
4. `ingestion/vector_index.py` stores vectors and ranks results with cosine similarity.
5. `ingestion/keyword_index.py` implements dependency-free BM25 keyword retrieval.
6. `ingestion/reranker.py` reranks vector candidates with lexical overlap.
7. `ingestion/answer_rewriter.py` demonstrates an injectable LLM rewriting contract.
8. `ingestion/prompt_security.py` builds escaped prompts with untrusted context boundaries.
9. `ingestion/pipeline.py` composes loading, chunking, embedding, indexing, and retrieval.

Read the matching Markdown lesson beside each module. The tests are executable interview examples, not a replacement for understanding the implementation.

## Run it without installing dependencies

```bash
python -m unittest discover -s tests -v
```

The default demo uses `HashEmbedder`, which is deterministic but not semantically meaningful. To use a real embedding model, install FastEmbed and inject `FastEmbedder`:

```bash
pip install fastembed
```

```python
from ingestion import FastEmbedder, RAGPipeline

pipeline = RAGPipeline(FastEmbedder())
pipeline.ingest_directory("./documents")
for result in pipeline.retrieve("What is the leave policy?", top_k=3):
    print(result.score, result.metadata["source"], result.text)
```

## Retrieval and generation examples

```python
from ingestion import AnswerRewriter, build_grounded_prompt

keyword_results = pipeline.retrieve_keyword("remote work policy", top_k=3)
prompt = build_grounded_prompt("What is the policy?", [result.text for result in keyword_results])
rewriter = AnswerRewriter("my-model", generator)
short_answer = rewriter.rewrite(answer, "Use three simple bullet points.")
```

## Interview checklist

Be ready to explain token approximation versus model tokenization, why overlap helps, why offsets are useful, cosine similarity, BM25, reranking candidate sets, zero vectors, top-k validation, deterministic ordering, dependency injection, prompt injection boundaries, and why retrieval quality depends on chunking and embedding quality. A production system would add persistence, batching, filtering, observability, access control, learned reranking, and an answer-generation provider.
