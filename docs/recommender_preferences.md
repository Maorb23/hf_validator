# Recommender preferences

Both explicit-model and discovery requests use `recommender/utilities.py` to normalize known
phrases, extract constraints, and interpret speculative decoding preferences. The original
request is preserved. Edit `PROMPT_ALIASES` to add common misspellings; extraction also recognizes
DSpark, DFlash, EAGLE, Medusa, MTP, and draft-model wording. RapidFuzz also corrects nearby
spellings of known multiword features, including `specolative decoding` and `tool caling`.
Tune `FUZZY_FEATURE_PHRASES`, the similarity cutoffs, and `FUZZY_MAX_WORD_EDITS` in the same utility.
The defaults allow one edit or transposition per word and preserve repository IDs and URLs.
Install the added dependency with `python -m pip install rapidfuzz` in existing environments.

Use `--speculative-decoding off|prefer|require` to override the prompt:

- `off`: no speculative feature bonus or draft memory reservation.
- `prefer`: verified support adds 0.05; inferred support adds 0.02. Unknown support stays eligible.
- `require`: only candidates with a documented exact target/draft/runtime combination qualify.

The same bonus settings apply in both flows. Edit `selection.speculative_decoding` in
`config/recommender.yaml` to adjust bonuses and discovery's research pool (default: 8 candidates).
Discovery considers card/config evidence before shortlisting and checks support again after
research. Its final list remains limited to three recommendations. A verified pairing establishes
documented compatibility; it does not establish measured speedup on the requested GPU.

When the prompt and CLI option leave the preference unspecified, interactive runs ask
`Prefer speculative decoding when supported? [Y/n]`; Enter means `prefer`. JSON runs return
`needs_input`, a `speculative_decoding` field, and the question. Resume with `--run-id` and
`--speculative-decoding prefer` (or `off`/`require`; `y`/`n` are accepted).

NanoClaw's `recommend_models` accepts the optional `speculative_decoding` field. Clarification
answers use that same field through `provide_recommendation_input`. Refresh the mounted installed
runtime and restart its MCP process after updating this code; Python code changes do not require
an image rebuild.
