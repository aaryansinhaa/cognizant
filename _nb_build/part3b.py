from common import md, code


def cells():
    return [
        md(
            """
---

## Phase 3 — Retrieval / ranking

You will implement top-k three ways:

1. `sorted(...)[:k]` — simple, O(n log n)
2. `heapq.nlargest` / `nsmallest` — O(n log k)
3. Full matrix multiply + `argpartition` — later, NumPy

`heapq` is a **min-heap**. For max-k by score, store `(-score, tie_key, item)` or use `nlargest`.
"""
        ),
        md(
            """
### 3.9 Top K Frequent Elements

**Statement.** Return the `k` most frequent numbers (order usually does not matter unless specified).

**Optimal.** Count O(n), then heap of size k, or bucket sort by frequency.

**RAG why.** Keyword retrieval, "most common terms in the corpus", cheap lexical baseline before embeddings.
"""
        ),
        code(
            """
def top_k_frequent(nums, k):
    if k <= 0:
        return []
    counts = Counter(nums)
    return [val for val, _ in counts.most_common(k)]


assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
assert top_k_frequent([1], 1) == [1]
assert top_k_frequent([], 3) == []
assert top_k_frequent([1, 2], 0) == []
assert sorted(top_k_frequent([4, 4, 4, 5, 5, 6], 3)) == [4, 5, 6]  # k == n unique
assert top_k_frequent([1, 1, 2, 2], 1) in ([1], [2])               # ties
print("top_k_frequent ok")
"""
        ),
        md(
            """
**Mistakes:** `sorted(counts, key=counts.get)` without `reverse=True`; returning keys of the Counter in insertion order; not defining tie-break when the hidden test is strict.

**Hidden:** `k` larger than unique count — return all unique. `k = 0`. All frequencies equal.
"""
        ),
        md(
            """
### 3.10 Kth Largest Element

**Statement.** Return the k-th largest value in an unsorted array (1-based).

**Optimal.** Quickselect average O(n), or min-heap of size k O(n log k). Python: `heapq.nlargest(k, nums)[-1]`.

**RAG why.** Thresholding scores ("keep everything at least as good as the k-th"), selecting a cutoff.
"""
        ),
        code(
            """
def kth_largest(nums, k):
    if not nums:
        raise ValueError("empty")
    if k < 1 or k > len(nums):
        raise ValueError("invalid k")
    return heapq.nlargest(k, nums)[-1]


assert kth_largest([3, 2, 1, 5, 6, 4], 2) == 5
assert kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
assert kth_largest([1], 1) == 1
assert kth_largest([-1, -2, -3], 1) == -1
assert kth_largest([-1, -2, -3], 3) == -3
assert kth_largest([7, 7, 7], 2) == 7
try:
    kth_largest([1, 2], 3)
    raise AssertionError
except ValueError:
    pass
print("kth_largest ok")
"""
        ),
        md(
            """
**Mistakes:** `sorted(nums)[k]` (that is k-th smallest, 0-based); mutating via `.sort()` when you must not; `k=0`.

**Complexity:** `nlargest` is O(n log k). Full sort is fine for n < ~10⁵ in an OA.
"""
        ),
        md(
            """
### 3.11 K Closest Points to Origin

**Statement.** `points[i] = [x, y]`. Return the k points with smallest Euclidean distance to `(0,0)`.

**Intuition.** This **is** nearest-neighbor search in 2-D. RAG is the same idea in 384-D with cosine instead of L2.

**Optimal.** Heap of size k, or `nlargest`/`nsmallest` on distance. Do not `sqrt` if you only compare (`x*x + y*y`).
"""
        ),
        code(
            """
def k_closest(points, k):
    if k <= 0:
        return []
    k = min(k, len(points))
    # nsmallest keeps the k nearest; tie-break by coordinates for stability
    return heapq.nsmallest(
        k,
        points,
        key=lambda p: (p[0] * p[0] + p[1] * p[1], p[0], p[1]),
    )


assert k_closest([[1, 3], [-2, 2]], 1) == [[-2, 2]]
assert k_closest([[3, 3], [5, -1], [-2, 4]], 2) == [[3, 3], [-2, 4]]
assert k_closest([], 3) == []
assert k_closest([[1, 0]], 1) == [[1, 0]]
assert k_closest([[1, 0], [-1, 0]], 2) == [[-1, 0], [1, 0]]  # tie on distance
assert k_closest([[0, 0], [1, 1]], 5) == [[0, 0], [1, 1]]    # k > n
print("k_closest ok")
"""
        ),
        md(
            """
**RAG mapping**

```
point              → embedding
origin             → query vector (after you shift to a space where that makes sense)
Euclidean distance → 1 - cosine  (if vectors are L2-normalized, cosine rank == L2 rank)
k closest          → top_k
```

If embeddings are **L2-normalized**, `argmax cosine` == `argmin ||q - d||²`.
"""
        ),
        md(
            """
### 3.12 Find K Pairs with Smallest Sums

**Statement.** Two sorted arrays `nums1`, `nums2`. Return k pairs `(u, v)` with smallest `u+v`.

**Intuition.** The matrix of sums is sorted in both directions. Expand like Dijkstra / heap from `(0,0)`.

**RAG why.** Fusing two ranked lists (keyword score + dense score) without materializing the full cartesian product — the same "do not sort n² pairs" instinct as "do not sort the entire corpus when you only need top-k".
"""
        ),
        code(
            """
def k_smallest_pairs(nums1, nums2, k):
    if not nums1 or not nums2 or k <= 0:
        return []
    heap = []
    visited = {(0, 0)}
    heapq.heappush(heap, (nums1[0] + nums2[0], 0, 0))
    out = []
    while heap and len(out) < k:
        _, i, j = heapq.heappop(heap)
        out.append([nums1[i], nums2[j]])
        if i + 1 < len(nums1) and (i + 1, j) not in visited:
            visited.add((i + 1, j))
            heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))
        if j + 1 < len(nums2) and (i, j + 1) not in visited:
            visited.add((i, j + 1))
            heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
    return out


assert k_smallest_pairs([1, 7, 11], [2, 4, 6], 3) == [[1, 2], [1, 4], [1, 6]]
assert k_smallest_pairs([1, 1, 2], [1, 2, 3], 2) == [[1, 1], [1, 1]]
assert k_smallest_pairs([], [1], 3) == []
assert k_smallest_pairs([1], [], 3) == []
assert k_smallest_pairs([1], [1], 1) == [[1, 1]]
assert k_smallest_pairs([1, 2], [3], 3) == [[1, 3], [2, 3]]  # k > all pairs
assert k_smallest_pairs([1, 2], [1, 2], 0) == []
print("k_smallest_pairs ok")
"""
        ),
        md(
            """
**Mistakes:** generating all pairs then sorting (TLE when arrays are 10⁵); forgetting visited so you push `(i+1,j)` twice; assuming arrays are unsorted.

**Complexity:** O(k log k) with the heap expansion.

### Phase 3 takeaway

In the assessment you will write something equivalent to:

```python
heapq.nsmallest(k, range(n), key=lambda i: (-scores[i], ids[i]))
```

Memorize that line.
"""
        ),
    ]
