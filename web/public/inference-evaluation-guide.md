# Plan an inference evaluation

Infiro is a working brand. This guide describes evaluation methodology, not a service commitment.

## Workload template

- Model repository and immutable revision:
- GPU platform and count (observed, not only requested):
- Engine version and worker image:
- Input/output token lengths:
- Prompt count and seed:
- Request concurrency:
- Objectives and units:
- Hard latency or throughput limits:
- Search budget and timeout:
- Source artifacts and fingerprints:
- Independent validation plan:

## Inference measurements

Output throughput is output tokens per second over the benchmark interval. Mean E2E latency measures request initiation to completion, in milliseconds. Mean TTFT measures time to first token. Preserve token-counting definitions and benchmark settings. Missing or default-zero fields are not evidence of successful measurements.

Compare the same model revision, GPU platform/count, workload sizes, prompt count, seed, concurrency, objectives, and measurement constraints. Serving settings may vary as the search dimensions. Keep engine and runtime versions attached.

A Pareto point is nondominated: no other candidate is no worse on every objective and strictly better on one. It is not automatically a deployment recommendation. Validate a selected configuration independently.

## Public simulations

The search lab uses Optuna 4.9.0 TPESampler, GPSampler, and RandomSampler with seeds 11 and 29. Bounds are [0,1] for x and y. Each trace has 24 trials. TPE and GP use six startup trials; GP sets deterministic_objective=True. Execution is seeded, sequential, and single-threaded. No GPU work is performed.

Branin: set a = -5 + 15*x, b = 15*y. Minimize:

    (b - 5.1/(4*pi**2)*a**2 + 5/pi*a - 6)**2
    + 10*(1 - 1/(8*pi))*cos(a) + 10

Two basins: minimize:

    1.6 - exp(-((x-.24)**2/.018 + (y-.72)**2/.045))
    - 1.35*exp(-((x-.77)**2/.035 + (y-.27)**2/.019))
    + .08*sin(15*x)*sin(12*y)

The inference explorer uses an educational formula:

    latency = 65 + 2.2*capacity + 85*(1-memory) + long_offset
    throughput = (400 + 25*capacity)*memory*workload_factor

The long-workload offset is 90 and its throughput factor is .72; short-workload values are 0 and 1. Configurations 3 and 8 receive latency *1.18 and throughput *.83 penalties to illustrate dominance. Engine labels use the identical formula; no engine ranking is supported. GPU and model names are illustrative. All displayed milliseconds and tokens/s in this dataset are synthetic.

## Publication

Public run snapshots retain model name, recorded draft model, seed, workload data, serving configurations and source fingerprints. Missing metadata does not remove runs: placeholders become explicit unknown values. Preserve known hardware and runtime versions where available. Only completed measured trials with valid throughput and p95 latency enter performance charts; failed and pruned trials remain in histories. Recorded p95 latency is distinct from simulated mean latency. Remove internal paths, prompts, credentials and private endpoints. Imported runs are not independently validated benchmarks.

## Existing workflow boundaries

Nemotron assists research and assessment. Validation uses the repository's worker orchestration. Optimization preparation delegates actual study execution to LLM Optimizer. Nebius jobs provide remote execution through those existing paths. The public website does not connect to these services.
