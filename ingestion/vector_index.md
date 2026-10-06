# Vector index lesson

## Job

`InMemoryVectorIndex` stores `(Chunk, vector)` pairs and linearly scans them for a query. For each pair it computes cosine similarity:

`similarity(a, b) = (a dot b) / (||a|| * ||b||)`

The highest scores are returned first. A chunk ID breaks score ties so results are deterministic.

## Edge cases

- Empty indexes and blank queries return `[]`.
- `top_k <= 0` is rejected because it is a caller error.
- Zero vectors receive score `0.0` instead of causing division by zero.
- Mismatched dimensions are rejected.

## Interview questions

**Complexity?** With `n` chunks and vector dimension `d`, a query is approximately `O(n * d)` time and `O(n * d)` memory. A production system may use an ANN index such as HNSW for scale.

**Why keep metadata on the chunk?** Retrieval must return citations or source context, not just anonymous numbers.

**What is the limitation of cosine search?** It only measures the supplied vector representation. Poor embeddings or poor chunks produce poor retrieval even when the math is correct.
