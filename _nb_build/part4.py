from common import md, code


def cells():
    return [
        md(
            """
# Part 4 — NumPy for vector operations

You need enough NumPy to store embeddings and score them. Not broadcasting trivia.

| Idea | Meaning |
|------|---------|
| `np.array(xs)` | typed contiguous block |
| `.shape` | `(n, d)` batch of n vectors of dim d |
| `.ndim` | 1 for a query, 2 for a matrix |
| `.dtype` | prefer `float32` for storage, `float64` for scoring if you want fewer rounding surprises |
| `a + b`, `a * b` | elementwise |
| `np.dot(a, b)` / `a @ b` | inner product or matmul depending on ranks |
| `np.linalg.norm(a)` | L2 length |
| `axis=0` | down rows (per column) |
| `axis=1` | across columns (per row) |
"""
        ),
        code(
            """
a = np.array([1.0, 2.0, 3.0])
print("shape", a.shape, "ndim", a.ndim, "dtype", a.dtype)
print("plus", a + 1, "times", a * 2)
print("dot", np.dot(a, a), "norm", np.linalg.norm(a))

batch = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
print("batch", batch.shape)
print("row norms", np.linalg.norm(batch, axis=1))
print("matmul batch @ [1,0]", batch @ np.array([1.0, 0.0]))

# normalize rows (safe)
norms = np.linalg.norm(batch, axis=1, keepdims=True)
safe = np.divide(batch, norms, out=np.zeros_like(batch), where=norms != 0)
print("normalized", safe)
"""
        ),
        md(
            """
## Cosine similarity

\\[
\\cos(\\theta) = \\frac{a \\cdot b}{\\|a\\| \\, \\|b\\|}
\\]

| Pair | Score |
|------|-------|
| identical direction | `+1` |
| orthogonal | `0` |
| opposite | `-1` |
| any zero vector | **undefined** — we return `0.0` and document it |
| different dims | raise |
| NaN | raise |

If both vectors are L2-normalized, cosine **is** the dot product. FastEmbed's BGE models typically already L2-normalize. **Do not assume that** in an OA unless you check `np.linalg.norm(v) ≈ 1`.
"""
        ),
        code(
            """
def cosine_similarity_python(a, b):
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("empty vector")
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def cosine_similarity(a, b):
    va = np.asarray(a, dtype=np.float64).reshape(-1)
    vb = np.asarray(b, dtype=np.float64).reshape(-1)
    if va.size == 0 or vb.size == 0:
        raise ValueError("empty vector")
    if va.shape != vb.shape:
        raise ValueError(f"dimension mismatch: {va.shape} vs {vb.shape}")
    if not np.isfinite(va).all() or not np.isfinite(vb).all():
        raise ValueError("NaN/Inf")
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


assert abs(cosine_similarity([1, 0], [1, 0]) - 1) < 1e-9
assert abs(cosine_similarity([1, 0], [0, 1])) < 1e-9
assert abs(cosine_similarity([1, 0], [-1, 0]) + 1) < 1e-9
assert cosine_similarity([0, 0], [1, 0]) == 0.0
assert abs(cosine_similarity_python([1.0, 0.0], [1.0, 0.0]) - 1) < 1e-9
# floating point: almost parallel
print(cosine_similarity([1, 0], [1, 1e-12]))
try:
    cosine_similarity([1, 0], [1, 0, 0])
    raise AssertionError
except ValueError:
    pass
print("cosine ok")
"""
        ),
        md(
            """
**Why NumPy is preferred.** A Python loop over 10k vectors × 384 dims is ~4e6 multiplies in the interpreter. NumPy does it in C/BLAS. In an OA, a correct Python loop still passes if n is small; write NumPy if they imported it.

**Batch scoring**
```python
scores = (matrix @ query) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query))
```
Guard zero norms.

**Takeaway:** raise on dim mismatch and NaN; define zero-vector behavior; vectorize the n×d loop.
"""
        ),
    ]
