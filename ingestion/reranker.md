# Reranking lesson

Retrieval can happen in two stages. A fast retriever gets a candidate set, then a reranker orders those candidates more carefully. `HybridReranker` is intentionally simple: it combines normalized vector score with the fraction of query terms present in each chunk.

A production system may replace this class with a cross-encoder, but the contract stays useful: accept a query and candidates, return a deterministic top-k list, and preserve chunk metadata.

Do not rerank the entire corpus with an expensive model. The candidate limit is the latency and cost boundary.
