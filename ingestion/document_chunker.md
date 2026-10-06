# Document chunker lesson

## Job

`DocumentChunker` turns one document into bounded `Chunk` records. It tries separators from largest semantic boundary to smallest: paragraph, line, sentence-like `. `, then spaces. A final token fallback guarantees progress for long unbroken text.

The implementation first creates recursively split pieces, then packs their tokens into windows of at most `max_tokens`. The next window starts `chunk_overlap` tokens before the previous window ends.

## Metadata

Each chunk has a deterministic ID such as `policy.txt_0`, its approximate token count, copied source metadata, a zero-based `chunk_index`, and `start_char`/`end_char` offsets. Because packed text normalizes whitespace between tokens, offsets describe the original source span, not necessarily a byte-for-byte substring of `chunk.text`.

## Interview questions

- Why overlap? It reduces the chance that a fact split at a boundary loses context.
- Why not use `text[:max_tokens]`? Characters are not tokens, and hard character cuts can split words.
- Why validate `chunk_overlap < max_tokens`? Otherwise the sliding window cannot advance and can loop forever.
- What is the production improvement? Use the target embedding model's tokenizer and preserve richer structural metadata such as headings.
