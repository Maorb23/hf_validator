# Project guide

`hf-model-watch` discovers, validates, and compares Hugging Face models for inference workloads. The Python package lives in `src/hf_model_watch`; the CLI entry point is `hf_model_watch.cli:app` and can also be run with `PYTHONPATH=src python -m hf_model_watch`.

## Main flows

- The Hugging Face scanner, eligibility checks, and initial scoring are in `src/hf_model_watch/hub/`.
- `src/hf_model_watch/recommender/core.py` handles the explicit-model recommender, deployment checks, Tavily evidence, and the Nebius OpenAI-compatible client.
- `src/hf_model_watch/recommender/discovery.py` handles requests without a model ID: recent, same-family, and Gemma scans; candidate merging; Tavily research; and final recommendations.
- `src/hf_model_watch/cli.py` exposes both recommender flows and interactive model selection. The discovery UI's “Optimize Inference + Benchmark” action is a placeholder.
- `notebooks/recommender_quickstart.ipynb` demonstrates the recommender. `config/recommender.yaml` holds its limits and weights.

## Development

- Use `requirements_cpu.txt` for the lightweight local recommender/notebook environment; installing the full project can pull GPU and training dependencies.
- Run focused recommender checks with `python -m pytest -q tests/test_recommender.py tests/test_discovery.py` in an environment with those dependencies installed.
- Preserve the existing explicit-model behavior when editing discovery. Reuse the Hub scanner and scoring code rather than adding another candidate scorer.
- Keep evidence tied to source URLs and distinguish verified, inferred, and unknown claims; a web mention alone does not prove runtime support.

We use ASCII art in this repository; Codex should ignore it and leave it untouched.
