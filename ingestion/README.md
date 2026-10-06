# `ingestion` package

## Python keywords and concepts

- `import`: load a module.
- `from ... import ...`: select names from a module.
- `class`: define a type that groups data and behavior.
- `def`: define a function or method.
- `return`: send a value back to the caller.
- `@dataclass`: generate common methods such as `__init__` for data containers.
- `Protocol`: describe the methods an object must provide without requiring inheritance.
- `list[T]`: a typed list annotation.
- `dict[K, V]`: a typed dictionary annotation.
- `| None`: a union that permits a value or `None`.
- `for`: iterate over a collection.
- `with`: manage a resource or context; file reading uses it internally here.
- `raise`: stop execution with an exception when an input violates a contract.

The package uses composition: the pipeline receives an embedder instead of constructing one internally. That makes the core easy to test and lets the same index work with FastEmbed or a test double.
