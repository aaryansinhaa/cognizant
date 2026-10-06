# RAG coding-round notes

## The likely shape of the exercise

Expect a partially implemented pipeline where you complete or debug small functions: text cleanup, recursive chunking, overlap, embedding calls, similarity scoring, top-k ordering, metadata propagation, and prompt/context assembly. The implementation here keeps each seam visible so you can practice those tasks independently.

## Questions worth practicing

1. How do you prevent an overlap configuration from creating an infinite loop?
2. What happens when a paragraph is larger than the maximum token budget?
3. Why should chunk IDs and result ordering be deterministic?
4. How do you preserve source metadata through chunking and retrieval?
5. What is the difference between lexical keyword search and embedding similarity?
6. How do you handle blank input, empty indexes, zero vectors, malformed embeddings, and `top_k=0`?
7. What is the time complexity of brute-force cosine search, and when would you use an ANN index?
8. Why is a test double useful for an embedding model?
9. How would you evaluate retrieval separately from answer generation? Think recall@k, precision@k, MRR, and citation correctness.
10. How would you stop retrieved content from overriding system instructions? Treat documents as untrusted context and delimit them in the generation prompt.
11. How would you reduce verbose answers? Add a separate answer-generation or rewriting contract with explicit output constraints and tests for the requested format.
12. Why is chunking not only an implementation detail? Chunk size, overlap, structure, and metadata directly affect retrieval quality and citation usability.

## A good verbal walkthrough

"I normalize text, create deterministic document IDs, recursively split at increasingly small boundaries, pack bounded token windows with overlap, embed chunks in batches, and store vectors with their chunks. At query time I embed the query, calculate cosine similarity, sort descending with a deterministic tie-breaker, and return the top-k chunks plus metadata. The current index is O(n*d); production would add persistence, filtering, batching, monitoring, and an approximate nearest-neighbor index."

## Scope note

This repository intentionally handles plain text only. PDF extraction, OCR, an LLM answer generator, reranking, and a persistent vector database are useful production extensions, but they are separate concerns and are not hidden inside this study implementation.
