# Answer rewriter lesson

`AnswerRewriter` separates prompt construction from model execution. `build_prompt(answer, style_instructions)` creates a stable prompt contract, while `rewrite(...)` delegates to an injected `TextGenerator`.

This is the shape commonly used in OA questions: validate string inputs, include both fields, preserve factual meaning, return a string, and make the model dependency mockable. The repository does not call a real LLM.

The delimited fields are data. Style instructions still control formatting, but content inside the original answer must not be allowed to redefine the editor's task.
