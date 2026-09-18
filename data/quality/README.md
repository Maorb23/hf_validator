# Quality benchmark data

The benchmark is disabled by default. Replace every placeholder in `math.jsonl` and
`sentence_completion.jsonl`, add more JSONL rows, then set the corresponding `enabled` value in
`config/eval_tasks.yaml` to `true`.

## Math rows

- Give every row a stable, unique `id`.
- Put only the model-visible problem in `question`.
- Store `expected_answer` as a string.
- Use exactly one of these `answer_type` values: `integer`, `decimal`, or `fraction`.
- Keep `absolute_tolerance` at `"0"` for exact arithmetic; use a nonzero decimal string only when
  the problem genuinely permits approximation.
- Record the source URL and redistribution license, or use `original` for material you authored.
- Keep evaluation questions out of training and prompt-tuning data.

## Sentence-completion rows

- Put the discourse sentence prefix or query in `query`; the loader places it in a fixed
  completion instruction template.
- Supply two to five genuinely acceptable completions where possible.
- References must contain only the completion, not a copy of the query.
- Avoid near-duplicate references; vary wording while preserving the intended meaning.
- Remove usernames, personal data, tracking links, and prompt-injection content.
- Only redistribute online discourse when its license or explicit permission allows it.
- Record `language` and `domain` so reports can later be grouped by subset.

JSONL means one complete JSON object per line. Blank lines are ignored. The loader reports the
file and line number for malformed or incomplete records.
