# From model discovery to validated LLM inference deployments

## Project goals

We are building a workflow that helps developers choose an open-weight language model and identify a serving configuration that meets their workload, hardware, latency, and budget requirements. Model popularity and published benchmarks rarely establish how a model will perform on a specific deployment. Our project connects model discovery with reproducible measurements on the target GPU.

The project combines two existing Python codebases: **hf-model-watch**, which discovers and assesses Hugging Face models, and **llm-optimizer**, which runs serving experiments and compares measured configurations. The requested GPU allocation will support an end-to-end validation campaign across vLLM, SGLang, and TensorRT-LLM, producing actionable deployment recommendations backed by recorded measurements.

## Approach and workflow

1. **Capture the deployment requirements.** Collect the application task, model size preferences, GPU resources, input/output token lengths, request concurrency, latency constraints, and spending limits through a CLI or conversational interface.
2. **Discover and assess models.** Scan Hugging Face candidates or assess a supplied model ID. Combine model metadata, model cards, deployment checks, and source-linked technical research to produce a shortlist. Claims are marked verified, inferred, or unknown; a web mention does not establish runtime compatibility.
3. **Freeze a reproducible experiment.** Select a model and pin its exact revision. Save the workload, random seed, hardware, runtime images, constraints, and experiment budget. Include speculative decoding only when the target model, draft component, revision, and runtime compatibility are supported by evidence.
4. **Run controlled GPU experiments.** After explicit approval, the controller queues independent framework jobs. Each worker starts the serving runtime, evaluates configurations within a bounded budget, and records throughput and latency. Configuration tuning covers applicable batching, scheduling, token budgets, memory allocation, attention, KV-cache, and CUDA-graph settings. Client workload settings remain fixed within each comparison.
5. **Compare measured tradeoffs.** For interactive workloads, examine end-to-end latency and time to first token. For concurrent workloads, examine output-token throughput and end-to-end latency. Apply latency constraints and retain the Pareto frontier, so users can choose a latency-first, throughput-first, or balanced configuration. Compare results only when their model revision, hardware, workload, objectives, and constraints match.
6. **Validate and deliver.** Independently rerun shortlisted configurations at the full benchmark request budget, compare them with baseline configurations, and export the launch settings, measurements, logs, and environment metadata. Unsupported or failed framework/model combinations are reported explicitly.

The controller maintains durable job state, supports cancellation and status queries, and stores results and checkpoints in S3-compatible storage. Successful measurements remain available if another framework fails. Checkpoints support recovery from interrupted runs with operator-controlled resumption.

## Proposed GPU allocation

The following is a planning request for this campaign, rather than a measured runtime requirement or an existing allocation.

| Resource | Requested configuration |
| --- | --- |
| Cloud | Nebius GPU cloud, using the existing remote-job controller |
| GPU model | NVIDIA H100 SXM, 80 GB GPU memory |
| Parallel allocation | **3 GPUs total**, as three independent one-GPU workers: one each for vLLM, SGLang, and TensorRT-LLM |
| Duration and total usage | **3 GPUs × 5 days × 24 hours = 360 GPU-hours maximum** |
| Worker host resources | Target 16 vCPUs and approximately 200 GB system RAM per one-GPU worker, subject to the available cloud preset |
| Worker disk | 450 GiB per serverless GPU job (`--disk-size 450Gi`), matching the existing job configuration; up to 1,350 GiB across three concurrent workers |
| Controller | One persistent CPU host; proposed 4 vCPUs, 16 GB RAM, and 50 GB durable disk |
| Artifact storage | Proposed 100 GB S3-compatible storage initially, expandable if retained logs require it |
| Networking | Outbound access to Hugging Face, the container registry, cloud job APIs, and object storage; controller-to-service connectivity |

The three GPUs run separate framework experiments; they are not a three-GPU tensor-parallel deployment. Benchmark request concurrency is a separate workload setting. No multi-GPU interconnect is required for this initial campaign. Models must fit a single H100 with sufficient headroom for their selected context length and KV cache.

The budget includes server startup, model loading, warmup, configuration trials, validation reruns, and troubleshooting. Workers will stop when their assigned work completes; 360 GPU-hours is the requested ceiling. GPU charges depend on the program's supplied rate; the compute budget is 360 multiplied by the applicable GPU-hour price, with CPU, storage, and network charges accounted for separately.

## Five-day execution plan

| Phase | Work | Maximum GPU-hours |
| --- | --- | ---: |
| Day 1 | Rebuild and smoke-test worker images; verify GPU visibility, model loading, native runtime configuration, and baseline measurements | 72 |
| Days 2–3 | Run bounded configuration studies across supported model/framework/workload combinations | 144 |
| Day 4 | Independently repeat baseline and shortlisted configurations; investigate stability and failures | 72 |
| Day 5 | Complete remaining validation, verify reproducibility, and produce the final comparison report | 72 |
| **Total** | **Up to three concurrent one-GPU workers over five days** | **360** |

## Experiment scope and evaluation

The initial campaign will target **two open-weight model revisions in the approximately 4B–8B parameter range**, selected through the discovery workflow and runtime preflight. Exact model IDs will be frozen before the main campaign. A framework/model combination that fails preflight will be documented rather than presented as supported.

For each supported model/framework pair, we propose three workload profiles:

| Profile | Input tokens | Output tokens | Concurrent requests |
| --- | ---: | ---: | ---: |
| Interactive | 512 | 128 | 1 |
| Concurrent serving | 512 | 128 | 32 |
| Longer-context serving | 4,096 | 256 | 32 |

This yields up to **18 configuration studies**: two models × three frameworks × three workload profiles. Start with 30 attempted configurations per study, for up to 540 attempts, adjusting downward if measured trial duration would exceed the allocation. Startup and failed attempts count toward the time budget.

Use a fixed seed and 256 requests per configuration during exploration. Independently validate the baseline and up to two shortlisted configurations per study using 1,024 requests and three repeats. These are proposed campaign settings, subject to the first day's timing and memory checks. Synthetic token-controlled workloads establish comparable serving measurements; results will be labeled accordingly and will not be presented as application-quality scores.

Record output tokens per second, time to first token, inter-token latency, end-to-end latency, request failures, and available mean and tail statistics. Compare throughput and latency against a recorded baseline under identical workload conditions. Report repeat-to-repeat variation and the number of requests behind tail statistics. Any latency limit must be specified before selection; if no configuration satisfies it, report that outcome explicitly.

The aim is to find reproducible improvements and clarify deployment tradeoffs. No performance uplift is claimed in advance. Serving speed alone does not establish model quality; any precision change or speculative mode included in a deployment recommendation also requires an appropriate output-quality check.

## Configuration and operational requirements

- Linux GPU workers with a compatible NVIDIA driver/CUDA stack and separate, version-pinned container images for each framework. Runtime versions and image references must be recorded.
- Rebuild images with the current optimizer worker before the campaign, including support for generated TensorRT-LLM configuration files and compressed experiment payloads. Existing image tags require live verification.
- Python 3.11+ for the recommendation/controller environment, Nebius authentication with GPU-job permissions, container-registry access, and S3 credentials supplied through the execution environment.
- Hugging Face access for selected checkpoints; gated models require approved access and a securely supplied token. Recommendation research additionally uses configured Nemotron and Tavily API access on the CPU side.
- Durable controller storage and a single active controller per database; per-job timeouts, explicit launch approval, and concurrency capped at three GPU workers.
- Prefer uninterrupted capacity for final validation. Preemptible capacity may be used for exploratory jobs, with checkpoints and explicit recovery procedures; reruns stay within the overall GPU-hour ceiling.

## Deliverables and completion criteria

We will deliver a reproducible comparison report, measured latency/throughput tradeoff plots, validated serving configurations with launch instructions, and versioned experiment manifests with model revisions, workload definitions, runtime versions, logs, and failure records.

Completion means every attempted model/framework/workload combination has either a documented compatibility failure or a baseline and independently repeated candidate measurements. Recommendations must reference those measurements, preserve their hardware and workload context, and state any remaining quality or compatibility limitations. The report will also reconcile actual GPU usage against the 360 GPU-hour allocation.

## Current implementation and work funded by this request

The repositories already contain model discovery and assessment, evidence tracking, request preparation, three-framework benchmark adapters, remote job orchestration, checkpoint/artifact handling, and result comparison. The work funded by this request is to verify the current worker images on the target hardware, execute the proposed campaign, independently validate candidate configurations, and produce a reproducible deployment report. Existing code and simulated results are not substitutes for this hardware validation.
