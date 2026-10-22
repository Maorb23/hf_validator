"""Deployment choices collected for optimization preparation after model selection."""
from __future__ import annotations

import json
import os

from pydantic import BaseModel, Field


DEFAULT_GPU_HOURLY_COST_USD = 1.0

IMAGE_REGISTRY = 'cr.eu-north1.nebius.cloud/e00p24e8mwqab11v1e'
DEFAULT_IMAGES = {
    framework: f'{IMAGE_REGISTRY}/llm-optimizer-{framework}:expanded-v1'
    for framework in ('vllm', 'sglang', 'trtllm')
}
DEFAULT_PRESETS = {
    ('gpu-h100-sxm', 1): '1gpu-16vcpu-200gb',
    ('gpu-h100-sxm', 8): '8gpu-128vcpu-1600gb',
    ('gpu-h200-sxm', 1): '1gpu-16vcpu-200gb',
    ('gpu-h200-sxm', 8): '8gpu-128vcpu-1600gb',
    ('gpu-l40s-a', 1): '1gpu-8vcpu-32gb',
    ('gpu-l40s-d', 1): '1gpu-16vcpu-96gb',
}


class OptimizationChoices(BaseModel):
    gpu_type: str | None = Field(default=None, pattern=r'^gpu-')
    input_tokens: int | None = Field(default=None, gt=0)
    output_tokens: int | None = Field(default=None, gt=0)
    concurrency: int | None = Field(default=None, gt=0)
    frameworks: list[str] | None = None
    optimizer_bucket: str | None = None
    optimizer_preset: str | None = None
    optimizer_images: str | None = None


QUESTIONS = {
    'gpu_type': 'Nebius GPU platform for optimization use (gpu-h100-sxm, gpu-h200-sxm, gpu-l40s-d, etc.)',
    'input_tokens': 'How many input tokens per request (512 for short prompts, 2048 for coding, 4096 for documents)?',
    'output_tokens': 'How many output tokens per request (128 for short answers, 512 for coding)?',
    'concurrency': 'How many concurrent requests?',
    'frameworks': 'Which serving frameworks should be optimized (e.g.: sglang, vllm, trtllm)?',
    'optimizer_bucket': 'Which existing artifact bucket should optimization use?',
    'optimizer_preset': 'Which Nebius instance preset should optimization use? (can be customized, enter for default)',
    'optimizer_images': 'Provide configured serving images as JSON, mapping each selected framework to its image.',
}


def execution_defaults(values: dict, execution: dict, frameworks: list[str]) -> dict:
    """Apply user-authorized defaults, preserving explicit configuration."""
    execution = dict(execution)
    execution.setdefault('preemptible', True)
    if not any(k in execution for k in ('spot_pricing_policy_id', 'follows_spot_price')):
        policy = os.getenv('NEBIUS_SPOT_PRICING_POLICY_ID', '').strip() if execution['preemptible'] else ''
        execution['spot_pricing_policy_id'] = policy or None
        execution['follows_spot_price'] = bool(execution['preemptible'] and not policy)
    execution['images'] = {**{f: DEFAULT_IMAGES[f] for f in frameworks if f in DEFAULT_IMAGES},
                           **execution.get('images', {})}
    if not execution.get('preset'):
        preset = DEFAULT_PRESETS.get((values.get('gpu_type'), values.get('gpu_count') or 1))
        if preset:
            execution['preset'] = preset
    return execution


def intake(values: dict, execution: dict) -> tuple[dict, dict, list[str]]:
    """Normalize and validate supplied choices; never infer infrastructure."""
    values, execution = dict(values), dict(execution)
    if isinstance(values.get('frameworks'), str):
        values['frameworks'] = [s.strip() for s in values['frameworks'].split(',') if s.strip()]
    choices = OptimizationChoices.model_validate(values)
    values.update(choices.model_dump(exclude_none=True))
    for field, key in [('optimizer_bucket', 'bucket'), ('optimizer_preset', 'preset')]:
        if values.get(field):
            execution[key] = values[field]
    if values.get('optimizer_images'):
        images = json.loads(values['optimizer_images'])
        if not isinstance(images, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in images.items()):
            raise ValueError('Serving images must be a JSON object of framework names to nonempty image strings')
        execution['images'] = {**execution.get('images', {}), **images}
    frameworks = values.get('frameworks') or list(execution.get('images', {}))
    if frameworks:
        if len(set(frameworks)) != len(frameworks):
            raise ValueError('Serving frameworks must be unique')
        values['frameworks'] = frameworks
    execution = execution_defaults(values, execution, frameworks)
    missing = [k for k in ('gpu_type', 'input_tokens', 'output_tokens', 'concurrency', 'frameworks') if not values.get(k)]
    for field, key in [('optimizer_bucket', 'bucket'), ('optimizer_preset', 'preset')]:
        if not execution.get(key) and (key != 'preset' or values.get('gpu_type')):
            missing.append(field)
    if any(not execution['images'].get(f) for f in frameworks):
        missing.append('optimizer_images')
    if execution.get('preset') and values.get('gpu_count') and not execution['preset'].startswith(str(values['gpu_count']) + 'gpu-'):
        raise ValueError('Instance preset must match the saved GPU count')
    return values, execution, missing
