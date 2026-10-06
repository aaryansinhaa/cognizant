# Keyword retrieval and BM25

BM25 is a lexical baseline: it rewards query terms that occur in a document, while discounting terms that occur in nearly every document. The implementation stores term frequencies and document frequencies in memory.

The main ideas are:

- `TF`: how often a query term appears in one chunk.
- `DF`: how many chunks contain that term.
- `IDF`: rarer terms receive more weight.
- Length normalization: unusually long chunks are not rewarded only for containing more words.

BM25 complements embeddings. Exact names, error codes, IDs, and policy terms are often better handled lexically than semantically.

Interview questions: explain `k1` and `b`, why unknown terms return zero, the cost of rebuilding corpus statistics, and how you would add metadata filters.
