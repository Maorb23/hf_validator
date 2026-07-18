## Phase implementation pattern

- Read master spec, current phase, prior code, and reference scripts first.

- Implement only the named phase.

- Keep scanner/discovery phases “wide and light”; avoid per-model deep network calls unless bounded.

- Use TRACE_EVENT for every CLI/job stage.

- Run exact success commands, not substitutes.

- If uv hits sandbox/Python/cache issues, set UV_CACHE_DIR inside workspace and use approved uv commands.

- For Hugging Face Hub, inspect installed API signatures when behavior differs; don’t assume docs/version.
- Report artifacts created, command results, and remaining TODOs exactly.