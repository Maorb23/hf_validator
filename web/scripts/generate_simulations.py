"""Generate deterministic educational traces with real Optuna samplers; no GPU jobs."""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import optuna
import scipy
import torch

OUT = Path(__file__).resolve().parents[1] / "public/data"
torch.set_num_threads(1)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def objective(name: str, x: float, y: float) -> float:
    if name == "branin":
        a, b = -5 + 15 * x, 15 * y
        return (b - 5.1 / (4 * math.pi**2) * a**2 + 5 / math.pi * a - 6) ** 2 + 10 * (1 - 1 / (8 * math.pi)) * math.cos(a) + 10
    return 1.6 - math.exp(-((x - .24)**2 / .018 + (y - .72)**2 / .045)) - 1.35 * math.exp(-((x - .77)**2 / .035 + (y - .27)**2 / .019)) + .08 * math.sin(15*x)*math.sin(12*y)


def write(name: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    data["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def generate_search() -> None:
    experiments = []
    for name in ["branin", "hills"]:
        grid = [{"x": i/30, "y": j/30, "value": round(objective(name, i/30, j/30), 6)} for i in range(31) for j in range(31)]
        for seed in [11, 29]:
            traces = {}
            for key, sampler in [("tpe", optuna.samplers.TPESampler(seed=seed, n_startup_trials=6)), ("gp", optuna.samplers.GPSampler(seed=seed, n_startup_trials=6, deterministic_objective=True)), ("random", optuna.samplers.RandomSampler(seed=seed))]:
                study = optuna.create_study(direction="minimize", sampler=sampler)
                def run(trial):
                    return objective(name, trial.suggest_float("x", 0, 1), trial.suggest_float("y", 0, 1))
                study.optimize(run, n_trials=24)
                best = float("inf")
                rows = []
                for trial in study.trials:
                    best = min(best, trial.value)
                    rows.append({"number": trial.number, "x": trial.params["x"], "y": trial.params["y"], "value": trial.value, "best": best, "state": trial.state.name})
                traces[key] = rows
                print(f"Generated {name}, seed {seed}, {key}", flush=True)
            experiments.append({"objective": name, "seed": seed, "grid": grid, "traces": traces})
    write("simulations.json", {"schemaVersion": 1, "kind": "simulation", "description": "Actual Optuna sampler traces on deterministic synthetic objectives. No inference or GPU measurements.", "versions": {"optuna": optuna.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__}, "trialBudget": 24, "startupTrials": 6, "bounds": {"x": [0,1], "y": [0,1]}, "objectives": {"branin": "Branin: x transformed to [-5,10], y to [0,15]; lower is better.", "hills": "Two smooth basins plus sinusoidal variation; lower is better."}, "experiments": experiments})


def generate_inference() -> None:
    rows = []
    for workload, inp, out, concurrency in [("short",128,32,8),("long",512,128,32)]:
        for engine in ["vLLM", "SGLang", "TensorRT-LLM"]:
            for i in range(12):
                capacity = [8,16,32,64][i % 4]
                memory = [.70,.80,.90][i // 4]
                latency = round(65 + capacity * 2.2 + (1-memory)*85 + (90 if workload=="long" else 0), 2)
                throughput = round((400+capacity*25)*memory * (1.0 if workload=="short" else .72), 2)
                # Some configurations are deliberately inefficient to teach dominance.
                if i in [2,7]: latency *= 1.18; throughput *= .83
                rows.append({"id": f"{workload}-{engine.lower()}-{i+1:02}", "kind":"simulation", "engine":engine, "workload":workload, "cohort":workload, "model":"Illustrative model", "gpu":"Illustrative GPU", "inputTokens":inp, "outputTokens":out, "concurrency":concurrency, "latency":round(latency,2), "throughput":round(throughput,2), "ttft":round(latency*.25,2), "configuration":{"serving_capacity":capacity,"memory_fraction":memory}, "state":"COMPLETE"})
    write("inference-examples.json", {"schemaVersion":1,"kind":"simulation","description":"Synthetic serving configurations calculated from an educational formula, not engine benchmarks. All engine labels use the same formula; no engine ranking is supported.","seed":11,"formula":"latency = 65 + 2.2*capacity + 85*(1-memory) + long_offset; throughput = (400 + 25*capacity)*memory*workload_factor; two configurations receive an inefficiency penalty.","rows":rows})


if __name__ == "__main__":
    generate_search()
    generate_inference()
