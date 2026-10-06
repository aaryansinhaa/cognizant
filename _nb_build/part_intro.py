from common import md, code


def cells():
    return [
        md(
            r"""
# RAG Coding Assessment Prep (Python, from scratch)

**Goal:** sit down next week and independently implement ingestion, preprocessing, chunking, FastEmbed embeddings, an in-memory vector index, cosine search, top-k retrieval, and modular Python -- without LangChain, LlamaIndex, or AI assistance.

This is **not** a Python textbook and **not** a RAG theory course. It is an assessment workbook: syntax, patterns, edge cases, hidden tests, and the exact primitives interviewers ask you to type.

You already know DSA from C++. Every Python section maps the new syntax to that mental model, then immediately uses it on RAG data.

## How to use this notebook

1. Read a concept, run the cells, then **close the solution** and re-implement from memory.
2. Treat red assertions as the contract. Assessments hide extra cases; this notebook lists them.
3. FastEmbed cells need `pip install fastembed numpy`. Everything else runs on the standard library + NumPy.
4. A companion package lives in `rag/` -- that is the Part 14 system. Learn it here first; do not import it during a mock timed drill.
5. Official FastEmbed API was verified against [qdrant/fastembed](https://github.com/qdrant/fastembed) (current default model `BAAI/bge-small-en-v1.5`).

## Table of contents

| Part | Topic |
|------|--------|
| 1 | Python building blocks for RAG coding |
| 2 | How Python represents RAG data |
| 3 | RAG-relevant DSA (hashing, windows, top-k) |
| 4 | NumPy + cosine similarity |
| 5 | FastEmbed (current API, not magic) |
| 6 | Ingestion + preprocessing |
| 7 | Chunking |
| 8 | Embedding pipeline |
| 9 | In-memory vector index |
| 10 | Retrieval and ranking |
| 11 | Module separation |
| 12 | Experienced-engineer mistakes |
| 13 | Testing / hidden cases |
| 14 | Mini-RAG from primitives |
| 15 | Timed assessment simulations (questions first, solutions after) |

## Recurring cell template

For every important idea this notebook uses:

1. What is it?
2. Why do I need it?
3. Syntax
4. Simple example
5. RAG-related example
6. Common mistakes
7. Edge cases
8. Hidden-test cases
9. Complexity
10. Practice problem
11. Assessment takeaway
"""
        ),
        code(
            """
from __future__ import annotations

import json
import math
import heapq
import re
from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

print("numpy", np.__version__)
"""
        ),
    ]
