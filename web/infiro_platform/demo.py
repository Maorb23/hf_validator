"""Explicit, offline test fixtures. Never selected by the production services."""

from types import SimpleNamespace
from unittest.mock import patch

from django.utils import timezone

from .contracts import Choices, DatasetConfig
from .models import Dataset, Job, Prepared, Recommendation
from .services import perform_prepare, write_artifact
from .workloads import identity, materialize

REV = "a" * 40
MODEL = "infiro/local-dry-run-fixture"


class FixtureTokenizer:
    vocab_size = 128

    def encode(self, text):
        return list(text.encode())

    def decode(self, tokens):
        return "".join(chr(65 + t % 26) for t in tokens)


def fixture_manifest(**overrides):
    overrides.setdefault("data_file", "data.json")
    config = DatasetConfig.model_validate({
        "repo_id": "infiro/local-sharegpt-fixture",
        "model": MODEL, "model_revision": REV,
        "num_prompts": 3, "input_tokens": 16, "output_tokens": 8,
        **overrides,
    }).model_dump()
    files = [SimpleNamespace(size=4096, rfilename="data.json")]
    model_files = [SimpleNamespace(size=1024, rfilename="tokenizer.json")]
    api = SimpleNamespace(
        dataset_info=lambda *a, **kw: SimpleNamespace(sha=REV, siblings=files),
        model_info=lambda *a, **kw: SimpleNamespace(sha=REV, siblings=model_files),
    )
    rows = [
        {
            "conversations": [
                {"from": "human", "value": f"Local fixture prompt {i}"},
                {"from": "gpt", "value": "Fixture response"},
            ]
        }
        for i in range(12)
    ]
    return config, materialize(
        config, api=api, loader=lambda *a, **kw: rows, tokenizer=FixtureTokenizer()
    )


def seed_prepared(user):
    config, manifest = fixture_manifest()
    rec = Recommendation.objects.create(
        owner=user,
        status="recommended",
        requirements={
            "request": "Offline fixture: verify the private recommendation-to-approval workflow."
        },
        report={
            "data_origin": "local_test_fixture",
            "validation": {"status": "not_called_offline_fixture"},
            "recommendations": [
                {
                    "model_id": MODEL,
                    "revision": REV,
                    "hf_score": 0,
                    "final_score": 0,
                    "reasoning": "Local test fixture. No live HF scan, Nemotron validation "
                    "or runtime verification has occurred.",
                    "why_it_fits": [],
                    "limitations": ["Offline fixture; not an actual model recommendation."],
                    "task_fit": {"gpu_fit": "unknown"},
                    "evidence": [],
                }
            ],
        },
        selected={"model_id": MODEL, "revision": REV},
    )
    dataset = Dataset.objects.create(
        owner=user,
        name="Local ShareGPT test fixture",
        config=config,
        status="ready",
        identity=identity(manifest),
        preview=[
            {
                "prompt": r["prompt"][:400],
                "prompt_len": r["prompt_len"],
                "output_len": r["output_len"],
            }
            for r in manifest["requests"][:3]
        ],
    )
    dataset_job = Job.objects.create(
        owner=user,
        kind="dataset",
        target_id=dataset.id,
        state="completed",
        available_at=timezone.now(),
        dedupe_key="fixture-dataset:" + str(dataset.id),
    )
    write_artifact(dataset_job, "workload.json", manifest)
    choices = Choices(
        num_prompts=3, input_tokens=16, output_tokens=8, speculative_decoding="off", trials=2
    ).model_dump()
    prepared = Prepared.objects.create(
        owner=user,
        recommendation=rec,
        recommendation_version=rec.version,
        dataset=dataset,
        choices=choices,
    )
    job = Job.objects.create(
        owner=user,
        kind="prepare",
        target_id=prepared.id,
        available_at=timezone.now(),
        dedupe_key="fixture-prepare:" + str(prepared.id),
    )
    with patch(
        "hf_model_watch.optimization.preparation.research",
        return_value={"sources": [], "inspection": {"claims": [], "heads": []}},
    ):
        perform_prepare(job)
    job.save()
    prepared.refresh_from_db()
    return prepared
