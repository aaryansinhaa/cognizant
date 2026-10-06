from common import md, code


def cells():
    return [
        md(
            """
# Part 1 — Python building blocks

C++ → Python cheat sheet you will use constantly:

| C++ | Python |
|-----|--------|
| `int x = 3;` | `x = 3` (runtime type, name is a reference) |
| `std::vector<T>` | `list` (mutable, dynamic, heterogeneous) |
| `std::array` / `std::pair` | `tuple` (immutable) |
| `std::unordered_map` | `dict` (insertion-ordered since 3.7) |
| `std::unordered_set` | `set` |
| `std::string` | `str` (immutable Unicode) |
| `nullptr` | `None` |
| RAII / scope guard | `with` |
| `const T&` default copy of containers | **assignment copies the reference**, not the object |
| `vector.at(i)` | `lst[i]` (IndexError, no `at`) |
| `map.count(k)` / `map[k]` inserts | `k in d`, `d.get(k)`, `d[k]` raises KeyError |

**Mutable vs immutable (this fails people in assessments):**

- Immutable: `int`, `float`, `bool`, `str`, `tuple`, `frozenset`
- Mutable: `list`, `dict`, `set`, custom objects, NumPy arrays
- Binding `b = a` does **not** copy a list/dict. Use `a.copy()`, `list(a)`, or `copy.deepcopy` for nested structures.
"""
        ),
        md(
            """
## 1.1 Variables and data types

**What:** a name bound to an object. The object has a type; the name does not.

**Why:** RAG records are nested dicts/lists. If you mutate a shared metadata dict, every chunk silently changes.
"""
        ),
        code(
            """
# Syntax / simple examples
n = 7                  # int (unlimited size, unlike 32/64-bit C++)
pi = 3.14              # float (IEEE-754 double)
ok = True              # bool, subclass of int
name = "doc1"          # str
nothing = None         # NoneType

print(type(n), type(name), type(nothing))
print(True + True)     # 2  — bool is an int. Do not use this in RAG code.

# Identity vs equality
a = [1, 2]
b = a
c = [1, 2]
print(a == c, a is c, a is b)  # True False True
"""
        ),
        code(
            """
# RAG-related: shared mutable metadata is a landmine
shared_meta = {"lang": "en"}
doc_a = {"id": "a", "metadata": shared_meta}
doc_b = {"id": "b", "metadata": shared_meta}
doc_a["metadata"]["lang"] = "fr"
print(doc_b["metadata"])  # {'lang': 'fr'}  — both changed

# Pythonic: copy at the boundary
doc_b = {"id": "b", "metadata": dict(shared_meta)}
"""
        ),
        md(
            """
**Common mistakes:** using `is` for equality (`x is 1000` can be False); treating `bool` as not an int; forgetting `None` is not `0` or `""`.

**Edge / hidden:** `bool("False")` is `True` (non-empty string). `int(True) == 1`. Huge ints never overflow.

**Complexity:** name lookup O(1) average.

**Practice:** bind two documents to the same list of tags, append a tag to one, predict the other.

**Takeaway:** copy mutable fields when you attach them to a new document/chunk.
"""
        ),
        md(
            """
## 1.2 Function parameters

**What:** Python arguments are passed by **object reference**. Rebinding a parameter does not rebind the caller's name; mutating a mutable object does.

| Style | C++ analogue | Example |
|-------|----------------|---------|
| positional | normal args | `f(1, 2)` |
| keyword | designated-ish | `f(b=2, a=1)` |
| default | default args | `def f(a, b=3)` |
| `*args` | variadic | extra positionals as tuple |
| `**kwargs` | not really | extra keywords as dict |

**Syntax**
```python
def load(path, encoding="utf-8", *, recursive=False, **kwargs):
    ...
```
The `*` makes `recursive` keyword-only (good API hygiene).
"""
        ),
        code(
            """
def greet(name, greeting="hello", *extras, shout=False, **meta):
    text = f"{greeting}, {name}"
    if shout:
        text = text.upper()
    return text, extras, meta

print(greet("Ada"))
print(greet("Ada", "hi", "note", shout=True, source="file.txt"))

# Default-argument trap (the #1 Python interview miss)
def add_chunk(chunk, bucket=[]):  # BAD: one list shared across calls
    bucket.append(chunk)
    return bucket

print(add_chunk("a"))
print(add_chunk("b"))  # ['a', 'b']  — leaked state


def add_chunk_ok(chunk, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(chunk)
    return bucket
"""
        ),
        md(
            """
**RAG example:** `chunk_text(text, chunk_size=100, overlap=20)` — keyword args at the call site document intent.

**Mistakes:** mutable defaults; putting `*args` before you need it; `def f(a=1, b)` is a SyntaxError.

**Hidden:** `def f(**kwargs)` plus later `kwargs["top_k"]` without `.get` blows up on missing keys.

**Takeaway:** keyword-only flags after `*` (`*, top_k=5`). Never default a list/dict/set to a literal.
"""
        ),
        md(
            """
## 1.3 Lists

**What:** ordered, mutable sequence. C++ `vector`, but can hold mixed types.

**Why:** documents, chunks, ranked results — almost every RAG pipeline is a list of dicts.
"""
        ),
        code(
            """
chunks = ["alpha", "beta", "gamma", "delta"]
print(chunks[0], chunks[-1])          # first / last
print(chunks[1:3])                    # [start:stop)  -> ['beta', 'gamma']
print(chunks[:2], chunks[2:], chunks[::2])
print(chunks[::-1])                   # reverse copy

chunks.append("eps")                  # O(1) amortized
chunks.extend(["z1", "z2"])           # concat in place
chunks.insert(0, "head")              # O(n)
item = chunks.pop()                   # last, O(1)
del chunks[0]                         # O(n)

# slicing copies
window = chunks[0:2]
window[0] = "MUTATED"
print(chunks[0])                      # original unchanged

# BUT elements are references
nested = [{"id": 1}, {"id": 2}]
alias = nested[0:1]
alias[0]["id"] = 99
print(nested[0])                      # {'id': 99}
"""
        ),
        code(
            """
# BAD: mutate while iterating
ids = ["d1", "d2", "d3"]
for i, item in enumerate(ids):
    if item == "d2":
        ids.pop(i)   # skips elements — do not do this

# GOOD
ids = [x for x in ["d1", "d2", "d3"] if x != "d2"]
print(ids)
"""
        ),
        md(
            """
**Edge:** `lst[10:20]` on a short list returns `[]` (no error). `lst[10]` raises. `lst[-0]` is `lst[0]`.

**Hidden:** `lst += other` mutates in place (so it works on a list passed into a function); `lst = lst + other` rebinds.

**Complexity:** index O(1), append O(1) amortized, insert/pop-front O(n), `x in lst` O(n).

**Practice:** given tokens, return overlapping windows of size 3 step 2 using only slicing.

**Takeaway:** slicing is how you implement character chunking. Membership tests on long lists belong in a `set`.
"""
        ),
        md(
            """
## 1.4 Tuples

**What:** immutable sequence. C++ `tuple` / `pair`.

**Why:** heap entries `(score, chunk_id, index)`, function multi-returns, dict keys.
"""
        ),
        code(
            """
record = ("doc1_chunk_0", 0.87)
chunk_id, score = record              # unpacking
print(record[0])

# Single-element tuple needs a comma
not_a_tuple = ("only")
is_a_tuple = ("only",)
print(type(not_a_tuple), type(is_a_tuple))

# Tuple is immutable, but it can CONTAIN mutables
pair = ("doc", {"lang": "en"})
pair[1]["lang"] = "de"                # allowed
# pair[0] = "other"                   # TypeError

# Dict key must be hashable — tuple of immutables works
seen = {("doc1", 0): True}
"""
        ),
        md(
            """
**Mistakes:** `("x")` is a string; mutating an object *inside* a tuple and thinking the tuple is a deep snapshot.

**Takeaway:** return `(chunk, score)` tuples from search, or a small dataclass. Prefer dataclasses once fields grow.
"""
        ),
        md(
            """
## 1.5 Sets

**What:** unordered unique hashables. C++ `unordered_set`.

**Why:** duplicate document IDs, visited characters in a sliding window, stopword filters, O(1) membership.
"""
        ),
        code(
            """
ids = ["a", "b", "a", "c"]
print(set(ids))                       # uniqueness
print(len(ids) != len(set(ids)))      # True => contains duplicate

allowed = {".txt", ".json"}
print(".md" in allowed)               # O(1) avg

# set operations
a, b = {"doc1", "doc2"}, {"doc2", "doc3"}
print(a | b, a & b, a - b)

# unhashable elements fail
try:
    {{"id": 1}}                       # TypeError
except TypeError as e:
    print("expected:", e)

# Pythonic duplicate check
def has_duplicate_ids(docs):
    seen = set()
    for d in docs:
        if d["id"] in seen:
            return True
        seen.add(d["id"])
    return False
"""
        ),
        md(
            """
**Edge:** empty set is `set()`, not `{}` (`{}` is a dict). Sets are unordered — do not rely on print order in tests (3.7+ dicts are ordered; sets are not).

**Hidden:** `set("aba")` is `{"a", "b"}` because strings are iterable.

**Complexity:** add/in/remove O(1) average, O(n) worst.

**Takeaway:** the instant you write `if id in list_of_ids` inside a loop, you wanted a set.
"""
        ),
        md(
            """
## 1.6 Dictionaries

**What:** hash map. The RAG record type.

**Why:** documents, chunks, embeddings-by-id, frequency counts, inverted indexes.
"""
        ),
        code(
            """
doc = {"id": "doc1", "text": "hello", "source": "a.txt"}
print(doc["id"])
print(doc.get("lang"))                # None, no exception
print(doc.get("lang", "en"))          # default
doc.setdefault("metadata", {})        # insert only if missing

# KeyError is the silent-test killer
try:
    _ = doc["missing"]
except KeyError as e:
    print("KeyError", e)

# iteration
for key, value in doc.items():
    pass
print(list(doc.keys()), list(doc.values()))

# merge (3.9+)
defaults = {"lang": "en", "source": "unknown"}
print({**defaults, **doc})            # doc wins
print(defaults | doc)                 # same

# frequency counting — two styles
text = "to be or not to be"
freq = {}
for w in text.split():
    freq[w] = freq.get(w, 0) + 1
print(freq)
print(Counter(text.split()))
"""
        ),
        code(
            """
# BAD: d[k] for optional metadata
chunk = {"id": "c0", "text": "hi"}
# lang = chunk["metadata"]["lang"]    # KeyError

# GOOD
lang = chunk.get("metadata", {}).get("lang", "en")
print(lang)

# mutating during .items() can raise RuntimeError
d = {"a": 1, "b": 2}
# for k in d:
#     d[k + k] = 0                   # RuntimeError
"""
        ),
        md(
            """
**Edge:** keys must be hashable. `dict.fromkeys(["a", "b"], [])` shares one list. Insertion order is preserved (use it for deterministic tie-breaks).

**Complexity:** get/set O(1) average.

**Practice:** invert `{chunk_id: doc_id}` to `{doc_id: [chunk_ids]}`.

**Takeaway:** `.get` / `.setdefault` / `Counter` / `defaultdict(list)` are the assessment primitives. Write them until they are muscle memory.
"""
        ),
        md(
            """
## 1.7 Strings

**What:** immutable Unicode sequence.

**Why:** every document is a string until it becomes a vector.
"""
        ),
        code(
            """
s = "Retrieval-Augmented Generation"
print(s.lower(), s.upper())
print(s.startswith("Retr"), s.endswith("tion"))
print("Aug" in s)
print(s.split())                      # whitespace split
print("a,b,,c".split(","))            # ['a', 'b', '', 'c']
print("  hi\\n".strip())
print(" ".join(["chunk", "0"]))
print(s[0:10])                        # character slice (Unicode code points)
print(len("naïve"))                   # 5 — len is code points, not bytes

# useful RAG cleanups
messy = "Hello,\\n\\n   world!\\tworld"
print(re.sub(r"\\s+", " ", messy).strip())

# immutable
t = s.replace("Generation", "Search")
print(s is t)
"""
        ),
        md(
            """
**Mistakes:** `for line in file` keeps `\\n`; comparing unstripped IDs; `text.split("")` is illegal; lowercasing then expecting "US" and "us" to stay distinct.

**Hidden:** `"ß".upper()` is `"SS"`; `"İ".lower()` can be `"i̇"` (two chars). Assessments with Unicode exist.

**Takeaway:** normalize **whitespace** by default; do not lowercase or strip punctuation unless the prompt says so.
"""
        ),
        md(
            """
## 1.8 Loops

```python
for item in iterable: ...
for i, item in enumerate(iterable, start=0): ...
while cond: ...
```

`for` in Python is "range-for", not C `for(i=0;...)`. Use `range(n)` when you need indices, `enumerate` when you need both.
"""
        ),
        code(
            """
docs = [{"id": "d1", "text": "aa"}, {"id": "d2", "text": "bb"}]

# Pythonic
for i, doc in enumerate(docs):
    print(i, doc["id"])

# C++-shaped (works, noisier)
for i in range(len(docs)):
    print(i, docs[i]["id"])

# zip — lockstep
ids = ["d1", "d2"]
texts = ["aa", "bb"]
for doc_id, text in zip(ids, texts):
    print(doc_id, text)

# zip stops at the SHORTEST — hidden test
print(list(zip([1, 2, 3], ["a"])))          # [(1, 'a')]
print(list(zip([1, 2, 3], ["a"], strict=True)))  # will raise in the next cell
"""
        ),
        code(
            """
try:
    list(zip([1, 2, 3], ["a"], strict=True))
except ValueError as e:
    print("zip(strict=True):", e)
"""
        ),
        md(
            """
**Takeaway:** `enumerate` for chunk indexes; `zip(..., strict=True)` when aligning chunks to embeddings. A silent short zip is an alignment bug.
"""
        ),
        md(
            """
## 1.9 Comprehensions

```python
[expr for x in xs if cond]
{k: v for k, v in pairs}
{x for x in xs}
```

Prefer them when the body is one expression. Use a loop when you have side effects or need to break.
"""
        ),
        code(
            """
texts = ["Hello", "", "World", "  "]
clean = [t.strip().lower() for t in texts if t.strip()]
print(clean)

# dict: id -> text
docs = [{"id": "d1", "text": "a"}, {"id": "d2", "text": "b"}]
by_id = {d["id"]: d["text"] for d in docs}
print(by_id)

# nested: flatten pages
pages = [["c0", "c1"], ["c2"]]
flat = [c for page in pages for c in page]
print(flat)

# BAD: side effects inside a comprehension
# [store.add(c) for c in chunks]      # do not
"""
        ),
        md(
            """
**Complexity:** still O(n). They are not "faster algorithms", they are faster to read.

**Takeaway:** filter empty chunks with a comprehension; never hide mutations in one.
"""
        ),
        md(
            """
## 1.10 File handling and 1.11 pathlib

C++ `ifstream` + RAII ≈ `with open(...)` / `Path.read_text()`.

Always use `pathlib.Path` in assessments. Always set `encoding="utf-8"` unless told otherwise.
"""
        ),
        code(
            """
from tempfile import TemporaryDirectory

with TemporaryDirectory() as td:
    root = Path(td)
    p = root / "notes.txt"
    p.write_text("hello\\nworld\\n", encoding="utf-8")

    # GOOD
    with p.open("r", encoding="utf-8") as f:
        lines = f.read()
    print(repr(lines))
    print(p.read_text(encoding="utf-8").splitlines())

    # directory listing — deterministic
    (root / "b.txt").write_text("b", encoding="utf-8")
    (root / "a.txt").write_text("a", encoding="utf-8")
    print(sorted(root.glob("*.txt")))

    # BAD: open without encoding (locale-dependent on some machines)
    # BAD: f = open(p); data = f.read()   # leak if exception
"""
        ),
        md(
            """
**Edge:** missing file → `FileNotFoundError`. Is a directory → `IsADirectoryError`. Permission → `PermissionError`. Empty file → `""`.

**Hidden:** `Path.glob` is not sorted. Sort it. Binary files opened as text raise `UnicodeDecodeError`.

**Takeaway:** `Path`, `encoding="utf-8"`, `with`, `sorted(glob)`.
"""
        ),
        md(
            """
## 1.12 Exceptions

```python
try:
    ...
except FileNotFoundError as e:
    ...
except (UnicodeDecodeError, json.JSONDecodeError) as e:
    ...
else:
    ...   # no exception
finally:
    ...   # always
```

Assessments usually want **fail fast** on bad input, not "return None and continue". Swallowing errors hides missing documents.
"""
        ),
        code(
            """
def parse_score(raw: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid score {raw!r}") from e
    if not math.isfinite(value):
        raise ValueError("score is NaN/Inf")
    return value

print(parse_score("0.87"))
try:
    parse_score("nan")
except ValueError as e:
    print(e)
"""
        ),
        md(
            """
**Takeaway:** catch the narrow type, wrap with context, do not `except Exception: pass` in a loader.
"""
        ),
        md(
            """
## 1.13 Modules and imports

```python
import json                         # module
from pathlib import Path            # name
from rag.preprocessing import chunker
from rag.preprocessing.chunker import chunk_text
```

- Package = directory with `__init__.py` (or namespace package).
- `from x import *` is banned in assessments.
- Circular imports: A imports B imports A — fix by pushing shared types to `models/`.

C++ `#include` copies text. Python `import` executes the module once and caches it in `sys.modules`.
"""
        ),
        md(
            """
## 1.14 JSON

| Call | Input | Output |
|------|--------|--------|
| `json.loads` | `str` | Python object |
| `json.load` | file object | Python object |
| `json.dumps` | object | `str` |
| `json.dump` | object + file | writes file |

JSON types: object→dict, array→list, string→str, number→int/float, true/false→bool, null→None.
"""
        ),
        code(
            """
payload = {"id": "doc1", "text": "hello", "n": 3}
s = json.dumps(payload, ensure_ascii=False, indent=2)
print(s)
print(json.loads(s)["id"])

# keys become strings
print(json.loads(json.dumps({1: "a"})))  # {'1': 'a'}

# tuples become lists — hidden
print(json.loads(json.dumps(("a", "b"))))

try:
    json.loads("{bad")
except json.JSONDecodeError as e:
    print("malformed", e.msg)
"""
        ),
        md(
            """
**Takeaway:** `load`/`dump` for files, `loads`/`dumps` for strings. The extra `s` means "string".
"""
        ),
        md(
            """
## 1.15–1.18 Iterators, enumerate, zip, sorted

An **iterator** is lazy. FastEmbed `embed()` returns a generator: nothing is computed until you iterate. Forgetting `list(...)` is the #1 FastEmbed bug.

`sorted(xs, key=fn, reverse=False)` returns a **new list**. `list.sort` is in-place.

`sorted` is stable.
"""
        ),
        code(
            """
def fake_embed(texts):
    print("computing")
    for t in texts:
        yield [float(len(t))]

gen = fake_embed(["a", "bb"])
print("created, not computed yet")
print(list(gen))
print(list(gen))   # []  — generators exhaust

results = [
    {"id": "b", "score": 0.9},
    {"id": "a", "score": 0.9},
    {"id": "c", "score": 0.1},
]
# rank by score desc, then id asc (deterministic ties)
ranked = sorted(results, key=lambda r: (-r["score"], r["id"]))
print(ranked)

print(list(enumerate(["chunk"], start=1)))
print(list(zip(["c0", "c1"], [[0.1], [0.2]])))
"""
        ),
        md(
            """
## 1.19–1.20 Lambda and custom sorting

`lambda args: expr` is an anonymous function. Use it for `key=` only. Named `def` otherwise.

`key` is called **once per element** (Schwartzian). Do not sort by a comparison callback like C++ `sort(cmp)` — Python 3 removed that; use `key` or `functools.cmp_to_key`.
"""
        ),
        code(
            """
chunks = [
    {"id": "doc1_chunk_10", "score": 0.4, "index": 10},
    {"id": "doc1_chunk_2", "score": 0.4, "index": 2},
]
print(sorted(chunks, key=lambda c: (-c["score"], c["index"])))

# BAD: sorting the formatted id as a string
print(sorted(["chunk_2", "chunk_10"]))   # ['chunk_10', 'chunk_2']
"""
        ),
        md(
            """
## 1.21 NumPy basics (preview — full treatment in Part 4)

`np.array` is a typed, contiguous block. RAG vectors are `shape (d,)` or a batch `(n, d)`.
"""
        ),
        code(
            """
v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
print(v.shape, v.ndim, v.dtype)
batch = np.vstack([v, v])
print(batch.shape)                    # (2, 3)
print(batch @ v)                      # matrix-vector, shape (2,)
"""
        ),
        md(
            """
## 1.22 OOP basics

```python
class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed_query(self, text: str) -> np.ndarray:
        ...
```

- `__init__` is not a constructor allocator; it initializes `self`.
- `self` is explicit (unlike C++ `this`).
- `@dataclass` generates `__init__` / `__repr__` / optional `__eq__`.
- Use a class when there is **lifecycle** (loaded model, growing index). Use a function when the operation is stateless (`cosine_similarity`).
"""
        ),
        code(
            """
@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    document_id: str
    metadata: dict = field(default_factory=dict)

c = Chunk("d1_chunk_0", "hello", "a.txt", 0, "d1")
print(c)
# c.text = "x"   # FrozenInstanceError — good: chunks should not mutate after index
"""
        ),
        md(
            """
### Part 1 practice set (do these on paper)

1. Write `def word_counts(text: str) -> dict[str, int]` using `.get` (no `Counter`).
2. Write `def first_unique_char(s: str) -> int` (index or -1).
3. Given `chunks` and `embeddings` of possibly different lengths, raise a clear error (do not zip silently).
4. Sort retrieval results by score descending, `chunk_id` ascending.
5. Read all `*.txt` in a folder with pathlib, sorted, UTF-8.

### Assessment takeaway

If you can write `.get`, `set`, slices, `enumerate`, `zip(..., strict=True)`, `sorted(key=...)`, `Path.read_text`, `json.loads`, and a frozen dataclass without thinking, Part 1 is done.
"""
        ),
    ]
