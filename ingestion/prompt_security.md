# Prompt-injection protection lesson

Retrieved documents are untrusted input. `build_grounded_prompt` places them inside an explicit `<untrusted_documents>` block, escapes markup, and tells the model not to follow instructions found there. `sanitize_retrieved_text` removes null bytes and redacts a small set of obvious instruction phrases.

This is defense in depth, not a proof of security. No string filter can reliably detect every attack. Stronger production controls include treating context as data in a structured model API, limiting tools available to the model, enforcing authorization before retrieval, validating citations, and evaluating adversarial documents.

The question itself is escaped separately so document text cannot close the context block with raw markup.
