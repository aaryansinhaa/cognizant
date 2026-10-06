from __future__ import annotations

import math

import numpy as np


def cosine_similarity_python(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity. Use only for tiny vectors / interviews."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("vectors must be non-empty")

    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def cosine_similarity(a, b) -> float:
    """NumPy cosine similarity. Returns 0.0 for any zero vector."""
    va = np.asarray(a, dtype=np.float64).reshape(-1)
    vb = np.asarray(b, dtype=np.float64).reshape(-1)
    if va.size == 0 or vb.size == 0:
        raise ValueError("vectors must be non-empty")
    if va.shape != vb.shape:
        raise ValueError(f"dimension mismatch: {va.shape} vs {vb.shape}")
    if not np.isfinite(va).all() or not np.isfinite(vb).all():
        raise ValueError("vectors contain NaN or Inf")
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def cosine_similarity_matrix(query, matrix) -> np.ndarray:
    """query: (d,), matrix: (n, d) -> scores (n,). Zero-rows score 0."""
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if q.size != m.shape[1]:
        raise ValueError(f"dimension mismatch: query {q.size} vs matrix {m.shape[1]}")
    if not np.isfinite(q).all() or not np.isfinite(m).all():
        raise ValueError("vectors contain NaN or Inf")

    q_norm = np.linalg.norm(q)
    row_norms = np.linalg.norm(m, axis=1)
    if q_norm == 0.0:
        return np.zeros(m.shape[0], dtype=np.float64)
    denom = q_norm * row_norms
    dots = m @ q
    scores = np.divide(dots, denom, out=np.zeros_like(dots), where=denom != 0)
    return scores
