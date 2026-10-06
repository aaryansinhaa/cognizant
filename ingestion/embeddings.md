# Embeddings lesson

## Job

An embedding maps text to a numeric vector. Similar meanings should ideally have vectors that point in similar directions. `EmbeddingModel` is a `Protocol`, so the index depends on an interface rather than a concrete vendor library.

`FastEmbedder` lazily imports FastEmbed and calls its `TextEmbedding.embed` method. This keeps FastEmbed optional for learners who only want to run the standard-library tests. `HashEmbedder` is deterministic and useful for checking plumbing, but its hash buckets do not understand meaning.

## Interview questions

**Why batch text?** `embed` accepts a sequence, allowing model implementations to batch work and reduce overhead.

**Why dependency injection?** It makes tests fast and lets an application switch embedding models without rewriting the vector index.

**What must remain stable?** Every vector in one index must have the same dimension and should come from a compatible model.
