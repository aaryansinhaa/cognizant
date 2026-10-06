# Pipeline lesson

## Job

`RAGPipeline` is the composition root. It loads `.txt` documents, chunks each document, embeds all chunks, stores them in the in-memory index, and converts search results into a small public `RetrievedChunk` shape.

The class does not generate an answer. Retrieval returns evidence; a separate generation layer would build a prompt from that evidence and call an LLM. Keeping those phases separate makes retrieval testable and prevents the model call from hiding indexing bugs.

## Interview questions

- What happens if the directory does not exist? The loader raises `NotADirectoryError`.
- What happens when there are no text files? Ingestion adds zero chunks and retrieval returns an empty list.
- How would you add filters? Apply metadata filtering before scoring, or add a filter parameter to the index.
- How would you improve production behavior? Batch ingestion, persist the index, add deduplication and access-control filters, and measure recall and answer faithfulness.
