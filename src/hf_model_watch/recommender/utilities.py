"""Shared prompt interpretation and speculative support rules for both flows."""
from __future__ import annotations

import json
import re
import shlex
from typing import Any, Literal

from rapidfuzz import fuzz, process
from rapidfuzz.distance import DamerauLevenshtein

SpeculativePreference = Literal["off", "prefer", "require"]

# Edit this list to reject model IDs or tags containing these keywords in both flows.
MODEL_REJECT_KEYWORDS = ("auditor",)
UNSUPPORTED_MODEL_FORMATS = ("mlx", "gguf", "compressed-tensors", "compressed_tensors")


def unsupported_model_format(model_id: str, config: dict[str, Any], tags: list[str]) -> bool:
    """Keep the existing Hub format restrictions in one shared implementation."""
    searchable = " ".join([model_id, *(str(tag) for tag in tags)]).lower()
    quantization = config.get("quantization_config")
    method = ""
    if isinstance(quantization, dict):
        method = str(quantization.get("quant_method") or quantization.get("format") or "").lower()
    return any(term in searchable for term in UNSUPPORTED_MODEL_FORMATS) or method in {
        "compressed-tensors", "compressed_tensors"
    }


def model_rejection_reasons(record: dict[str, Any]) -> list[str]:
    """Return common model/format blockers before research and during final eligibility."""
    ident = str(record.get("model_id") or "")
    tags = record.get("tags") if isinstance(record.get("tags"), list) else []
    config = record.get("config") if isinstance(record.get("config"), dict) else {}
    searchable = " ".join([ident, *(str(tag) for tag in tags)]).lower()
    reasons = []
    if unsupported_model_format(ident, config, tags):
        reasons.append("unsupported_finetune_format")
    if any(keyword.lower() in searchable for keyword in MODEL_REJECT_KEYWORDS):
        reasons.append("model_keyword_rejected")
    return reasons

# Recognized license names only; unfamiliar wording must not manufacture a conflict.
LICENSE_PATTERNS = {
    "apache-2.0": r"apache(?:\s+license)?[\s,-]*(?:version\s+)?2\.0",
    "mit": r"mit",
    "llama3.2": r"(?:meta[\s-]+)?llama[\s-]*3\.2(?:\s+community)?(?:\s+license)?",
    "llama3.1": r"(?:meta[\s-]+)?llama[\s-]*3\.1(?:\s+community)?(?:\s+license)?",
    "llama3": r"(?:meta[\s-]+)?llama[\s-]*3(?:\s+community)?(?:\s+license)?",
    "llama2": r"(?:meta[\s-]+)?llama[\s-]*2(?:\s+community)?(?:\s+license)?",
    "bsd-3-clause": r"bsd[\s-]*3[\s-]*clause",
    "bsd-2-clause": r"bsd[\s-]*2[\s-]*clause",
    "cc-by-4.0": r"cc[\s-]*by[\s-]*4\.0",
    "cc-by-nc-4.0": r"cc[\s-]*by[\s-]*nc[\s-]*4\.0",
}


def normalize_license(value: str | None) -> str | None:
    """Canonicalize a recognized complete name; leave unknown licenses unknown."""
    text = _normalize_inline_markdown(str(value or "")).strip().lower()
    text = re.sub(r"^(?:the|an?|under)\s+", "", text).strip(' \"\'.,;:')
    for canonical, pattern in LICENSE_PATTERNS.items():
        if re.fullmatch(pattern + r"(?:\s+license)?", text):
            return canonical
    return None


def license_assessment(metadata_license: str | None, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare recognized source claims with metadata, retaining their source URLs."""
    metadata = normalize_license(metadata_license)
    claims = []
    for source in evidence:
        text = _normalize_inline_markdown(str(source.get("retrieved_text") or ""))
        for match in re.finditer(r"\blicen[sc]ed?\s+under\s+(?:the\s+|an?\s+)?", text, re.I):
            remainder = text[match.end():]
            for canonical, pattern in LICENSE_PATTERNS.items():
                # Match whole names, not partial versions (Llama 3 is not Llama 3.2).
                named = re.match(pattern + r"(?![\w-]|\.\d)", remainder, re.I)
                if named:
                    claims.append({"license": canonical, "source_url": source.get("source_url")})
                    break
    # Search and extraction may supply the same page; retain each sourced claim once.
    claims = list({(claim["license"], claim["source_url"]): claim for claim in claims}.values())
    conflict = bool(metadata and any(claim["license"] != metadata for claim in claims))
    return {"metadata_license": metadata, "claims": claims, "conflict": conflict,
            "status": "conflicting" if conflict else "consistent" if metadata and claims else "unknown"}

# Edit these aliases to teach both flows additional spellings; never rewrite model IDs.
PROMPT_ALIASES = {
    "speculative decuding": "speculative decoding",
    "speculative decodng": "speculative decoding",
    "speculativ decoding": "speculative decoding",
    "speculitive decoding": "speculative decoding",
    "speculative decode": "speculative decoding",
    "spec decoding": "speculative decoding",
}
SPECULATIVE_PATTERN = (
    r"speculative decoding|dflash|d-flash|d-spark|dspark|medusa|"
    r"eagle(?:-\d+)?|multi-token prediction|mtp|draft model"
)

# Keep fuzzy correction restricted to phrases, rather than ordinary single words.
FUZZY_FEATURE_PHRASES = ("speculative decoding", "tool calling", "draft model", "multi-token prediction")
FUZZY_SCORE_CUTOFF = 85
FUZZY_WORD_CUTOFF = 85
FUZZY_MAX_WORD_EDITS = 1
_PROTECTED_TEXT = re.compile(r"https?://\S+|[\w.-]+/[\w./-]+", re.I)
_LAUNCH_COMMAND = re.compile(
    r"(?:python(?:3)?\s+-m\s+sglang\.launch_server|vllm\s+serve)\b[^\n\r]*"
)


def _normalize_feature_phrases(text: str) -> str:
    """Apply explicit aliases first, then conservatively match nearby phrase spellings."""
    for alias, canonical in PROMPT_ALIASES.items():
        text = re.sub(r"\b" + re.escape(alias) + r"\b", canonical, text, flags=re.I)
    words = list(re.finditer(r"\b[A-Za-z]+(?:-[A-Za-z]+)*\b", text))
    replacements = []
    consumed_until = 0
    for index, word in enumerate(words):
        if word.start() < consumed_until:
            continue
        for count in sorted({len(phrase.split()) for phrase in FUZZY_FEATURE_PHRASES}, reverse=True):
            window = words[index:index + count]
            if len(window) != count or any(
                not text[left.end():right.start()].isspace()
                for left, right in zip(window, window[1:])
            ):
                continue
            phrase = " ".join(token.group().lower() for token in window)
            match = process.extractOne(
                phrase, [p for p in FUZZY_FEATURE_PHRASES if len(p.split()) == count],
                scorer=fuzz.ratio, score_cutoff=FUZZY_SCORE_CUTOFF,
            )
            # Each word must be close too: 'speculative encoding' is a different phrase.
            if match and all(fuzz.ratio(a, b) >= FUZZY_WORD_CUTOFF
                             and DamerauLevenshtein.distance(a, b) <= FUZZY_MAX_WORD_EDITS
                             for a, b in zip(phrase.split(), match[0].split())):
                if phrase != match[0]:
                    replacements.append((word.start(), window[-1].end(), match[0]))
                consumed_until = window[-1].end()
                break
    for start, end, canonical in reversed(replacements):
        text = text[:start] + canonical + text[end:]
    return text


def normalize_prompt(request: str) -> str:
    """Normalize recognized phrases while preserving unrelated text and the original input."""
    # Protect repository IDs and URLs even when they contain a feature-like spelling.
    parts = []
    cursor = 0
    for match in _PROTECTED_TEXT.finditer(request):
        parts.extend((_normalize_feature_phrases(request[cursor:match.start()]), match.group()))
        cursor = match.end()
    parts.append(_normalize_feature_phrases(request[cursor:]))
    return "".join(parts)


def speculative_preference(request: str, explicit: str | None = None) -> SpeculativePreference | None:
    """Explicit input wins; absent feature wording remains a missing input."""
    if explicit is not None:
        aliases = {"y": "prefer", "yes": "prefer", "n": "off", "no": "off"}
        value = aliases.get(str(explicit).strip().lower(), str(explicit).strip().lower())
        if value not in {"off", "prefer", "require"}:
            raise ValueError("Speculative decoding must be off, prefer, or require (or y/n)")
        return value
    text = normalize_prompt(request).lower()
    if not re.search(r"\b(?:" + SPECULATIVE_PATTERN + r")\b", text):
        return None
    # Scope requirement/negation wording to the feature, not to unrelated constraints.
    feature = r"(?:" + SPECULATIVE_PATTERN + r")"
    negative = (
        r"\b(?:no|without|disable|avoid|do not use|don't use|do not want|"
        r"don't want|must not have)\s+(?:(?:any|a)\s+)?"
    )
    required = (
        r"\b(?:must(?:\s+have|\s+support|\s+use)?|require[ds]?|mandatory)"
        r"\s+(?:an?\s+)?(?:support for\s+)?(?:verified\s+)?"
    )
    if re.search(negative + feature, text):
        return "off"
    if re.search(required + feature, text) or re.search(
        feature + r"\s+(?:is\s+)?(?:required|mandatory)\b", text
    ):
        return "require"
    return "prefer"


def detect_features(request: str, explicit: str | None = None) -> list[str]:
    """Detect canonical feature names using the same vocabulary in both paths."""
    text = normalize_prompt(request).lower()
    features = [term for term in ("reasoning", "vision", "tool calling", "coding")
                if re.search(r"\b" + re.escape(term) + r"\b", text)]
    if speculative_preference(request, explicit) in {"prefer", "require"}:
        features.insert(0, "speculative decoding")
    return features


def speculative_status(record: dict[str, Any], evidence: list[dict[str, Any]],
                       assessment: dict[str, Any] | None = None) -> str:
    """A usable exact pair is verified; a candidate-specific mention is only inferred."""
    if (assessment or {}).get("usable") or _pair_evidence(record, evidence):
        return "verified"
    ident = str(record["model_id"]).lower()
    texts = [str(record.get("card_excerpt") or "")]
    texts.extend(str(source.get("retrieved_text") or "") for source in evidence
                 if ident in (str(source.get("source_url") or "") + " "
                              + str(source.get("retrieved_text") or "")).lower())
    return "inferred" if any(re.search(r"\b(?:" + SPECULATIVE_PATTERN + r")\b", text, re.I)
                             for text in texts) else "unknown"


def speculative_bonus(status: str, preference: str | None, settings: dict[str, Any]) -> float:
    """Use the same configurable bonus in shortlisting and final ranking."""
    if preference not in {"prefer", "require"}:
        return 0.0
    weights = settings.get("selection", {}).get("speculative_decoding", {})
    if status == "unknown":
        return 0.0
    defaults = {"verified": 0.05, "inferred": 0.02}
    return float(weights.get(status + "_bonus", defaults.get(status, 0)))


def _normalize_inline_markdown(text: str) -> str:
    """Remove formatting delimiters without changing underscores inside model IDs."""
    text = re.sub(r"\[([^\]\n]+)\]\(https?://[^\s)]+\)", r"\1", text)
    return re.sub(
        r"(?<!\w)(?:\*\*|__|`)|(?:\*\*|__|`)(?!\w)",
        "",
        text,
    )


def documented_launches(text: str) -> list[dict[str, str]]:
    """Read complete launch commands; never combine flags from separate examples."""
    text = _normalize_inline_markdown(re.sub(r"\\\s*\n", " ", text))
    launches = []
    for match in _LAUNCH_COMMAND.finditer(text):
        try:
            args = shlex.split(match.group(), comments=True)
        except ValueError:
            continue
        def option(name: str) -> str | None:
            for index, argument in enumerate(args):
                if argument.startswith(name + "="):
                    return argument.partition("=")[2]
                if argument == name and index + 1 < len(args):
                    return args[index + 1]
            return None
        runtime = "vllm" if args[0] == "vllm" else "sglang"
        target = args[2] if runtime == "vllm" and len(args) > 2 else option("--model-path")
        if not target or target.startswith("-"):
            continue
        launch = {"runtime": runtime, "target_model": target}
        draft = option("--speculative-draft-model-path")
        algorithm = option("--speculative-algorithm")
        if runtime == "sglang" and draft and algorithm:
            launch.update(draft_model=draft, algorithm=algorithm)
        if runtime == "vllm":
            raw = option("--speculative-config") or option("--speculative_config")
            try:
                speculative = json.loads(raw) if raw else {}
            except (ValueError, TypeError):
                speculative = {}
            if isinstance(speculative, dict):
                draft, algorithm = speculative.get("model"), speculative.get("method")
                if isinstance(draft, str) and isinstance(algorithm, str) and draft and algorithm:
                    launch.update(draft_model=draft, algorithm=algorithm)
        launches.append(launch)
    return launches


def official_target_source(url: str, target: str) -> bool:
    """Only the exact target repository or its README can establish launch support."""
    return bool(re.fullmatch(
        r"https://huggingface\.co/" + re.escape(target)
        + r"/?(?:\?[^#]*)?|https://huggingface\.co/" + re.escape(target)
        + r"/(?:blob|resolve)/[^?#]+/README\.md(?:\?[^#]*)?", url
    ))


def evidence_excerpt(text: str, limit: int) -> str:
    """Keep deployment command sections even when the model-card introduction is long."""
    if len(text) <= limit:
        return text
    blocks = re.findall(
        r"```[^\n]*\n(?:(?!```)[\s\S])*```", text
    )
    deployment = [block for block in blocks if documented_launches(block)]
    # Extractors also return inline-backtick or bare commands, without fenced blocks.
    flattened = _normalize_inline_markdown(re.sub(r"\\\s*\n", " ", text))
    deployment.extend(match.group() for match in _LAUNCH_COMMAND.finditer(flattened)
                      if documented_launches(match.group()))
    deployment = list(dict.fromkeys(deployment))
    if not deployment:
        return text[:limit]
    # Prefer complete speculative commands, then other launch examples, over introductory text.
    deployment.sort(key=lambda block: "--speculative-draft-model-path" in block, reverse=True)
    commands = "\n\n".join(deployment)
    return (commands + "\n\n" + text[:max(0, limit - len(commands) - 2)])[:limit]


SPECULATIVE_METHODS = ("dspark", "dflash", "eagle", "mtp", "medusa")


def speculative_rank(pair: dict[str, Any], preference: str | None,
                     method_order: list[str]) -> tuple[int, int]:
    """Usable documented pairs outrank mentions; method priority precedes scores."""
    if preference not in {"prefer", "require"}:
        return (0, 0)
    if not pair.get("usable"):
        return (1, len(method_order))
    method = pair.get("method") or pair_method(pair)
    return (0, method_order.index(method) if method in method_order else len(method_order))


def pair_method(pair: dict[str, Any]) -> str:
    text = str(pair.get("draft_model", "")) + " " + str(pair.get("algorithm", ""))
    for method, pattern in (("dspark", r"d[-_]?spark"), ("dflash", r"d[-_]?flash"),
                            ("eagle", r"eagle"), ("mtp", r"mtp|multi.token"), ("medusa", r"medusa")):
        if re.search(pattern, text, re.I):
            return method
    return "unknown"


def speculative_method_order(request: str, configured: list[str] | None = None) -> list[str]:
    order = list(configured or SPECULATIVE_METHODS)
    if len(set(order)) != len(order) or any(m not in SPECULATIVE_METHODS for m in order):
        raise ValueError("Unsupported or duplicate speculative research method")
    # Do not interpret a method name embedded in a model ID/URL as a preference.
    text = _PROTECTED_TEXT.sub("", request)
    mentioned = []
    for method in SPECULATIVE_METHODS:
        pattern = {"dspark": r"\bd[-_]?spark\b", "dflash": r"\bd[-_]?flash\b",
                   "eagle": r"\beagle(?:[-_]?\d+)?\b", "mtp": r"\bmtp\b|multi.token prediction",
                   "medusa": r"\bmedusa\b"}[method]
        match = re.search(pattern, text, re.I)
        if match:
            mentioned.append((match.start(), method))
    explicit = [m for _, m in sorted(mentioned)]
    return explicit + [m for m in order if m not in explicit]


def documented_pairs(record: dict[str, Any], evidence: list[dict[str, Any]],
                     method_order: list[str] | None = None) -> list[dict[str, Any]]:
    """
    Identify a target/drafter pair and runtime from the evidence, if present.
    Accept an exact launch on the official target card, or a draft card explicitly
    naming the target and runtime in the same source. Execution remains unverified.
    """
    target = str(record["model_id"])
    candidates: list[tuple[int, float, dict[str, Any]]] = []
    for source in evidence:
        url = str(source.get("source_url") or "")
        body = str(source.get("retrieved_text") or "")
        normalized_body = _normalize_inline_markdown(body)
        # A target's own card may document the complete external draft/runtime pairing.
        if official_target_source(url, target):
            for launch in documented_launches(body):
                draft = launch.get("draft_model")
                if (launch["target_model"] == target and draft and draft != target
                    and re.fullmatch(r"[\w.-]+/[\w.-]+", draft)
                    and not any(term in draft.lower() for term in ("gguf", "gptq"))):
                    candidates.append((2, 0.0, {
                        "target_model": target, "draft_model": draft,
                        "runtimes": [launch["runtime"]], "algorithm": launch["algorithm"],
                        "source_url": url,
                    }))
            continue
        draft_match = re.fullmatch(r"https://huggingface\.co/([^/?#]+/[^/?#]+)/?(?:\?[^#]*)?", url)
        if not draft_match or draft_match.group(1) == target:
            continue
        draft_id = draft_match.group(1)
        lower_body = normalized_body.lower()
        # GGUF and other converted weights cannot inherit the source card's
        # SGLang instructions merely because those instructions were copied.
        if (
            "gguf" in draft_id.lower()
            or "gptq" in draft_id.lower()
            or "quantized version of" in lower_body
            or "model card is pasted" in lower_body
        ):
            continue
        explicit_target = re.search(
            r"(?:draft model for|target model|verifier)\s*(?:\||:)?\s*[`\"']?([\w.-]+/[\w.-]+)",
            normalized_body,
            re.IGNORECASE,
        )
        launches = [launch for launch in documented_launches(body)
                    if launch.get("target_model") == target and launch.get("draft_model") == draft_id]
        if not launches and (not explicit_target or explicit_target.group(1).rstrip(".") != target):
            continue
        # The exact target claim matters; the drafting method can be DFlash,
        # DSpark, EAGLE, Medusa, or another documented speculative method.
        speculative = any(
            term in lower_body or term in draft_id.lower()
            for term in ("speculat", "draft", "dflash", "dspark", "eagle", "medusa")
        )
        runtimes = [
            runtime
            for runtime in ("sglang", "vllm", "trt-llm", "transformers")
            if speculative and runtime in lower_body
        ]
        if method_order is not None:
            runtimes = [runtime for runtime in runtimes if any(l["runtime"] == runtime for l in launches)
                        or re.search(r"(?:use|using|run|serve|supports?|with)\s+(?:the\s+)?" + re.escape(runtime) + r"\b",
                                     lower_body)]

        if runtimes:
            pair = {
                "target_model": target,
                "draft_model": draft_id,
                "runtimes": runtimes,
                "source_url": url,
            }
            direct_launch = int(
                f"--speculative-draft-model-path {draft_id}" in normalized_body
            )
            try:
                relevance = float(source.get("score") or 0)
            except (TypeError, ValueError):
                relevance = 0.0
            candidates.append((direct_launch, relevance, pair))

    def priority(item):
        method = pair_method(item[2])
        preferred = -(method_order.index(method) if method_order and method in method_order else len(method_order or []))
        return (preferred, item[0], item[1], item[2]["draft_model"])
    pairs = []
    seen = {}
    for item in sorted(candidates, key=priority, reverse=True):
        pair = dict(item[2])
        key = (pair["draft_model"], tuple(pair["runtimes"]))
        if key not in seen:
            seen[key] = pair
            pair["method"] = pair_method(pair)
            pair["source_urls"] = [pair["source_url"]]
            pairs.append(pair)
        elif pair["source_url"] not in seen[key]["source_urls"]:
            seen[key]["source_urls"].append(pair["source_url"])
    return pairs


def _pair_evidence(record: dict[str, Any], evidence: list[dict[str, Any]],
                   method_order: list[str] | None = None) -> dict[str, Any] | None:
    pairs = documented_pairs(record, evidence, method_order)
    return pairs[0] if pairs else None

def extract_constraints(request: str, preference: str | None = None) -> dict[str, Any]:
    """
    Function to extract constraints from a user request string.
    It identifies features, reference models, size, memory, budget, cost, tokens, license, family, and other relevant information from the request text.
    Args:
        request (str): The user request string containing model requirements and constraints.
    Returns:
        dict[str, Any]: A dictionary containing extracted constraints and features from the request.
    """
    original_request = request
    request = normalize_prompt(request)
    text = request.lower()
    # Recognize common capabilities directly from the original user request.
    features = detect_features(request, preference)
    for match in re.finditer(
        r"\b(?:with|supports?|including)\s+(?:an?\s+)?(?:optional\s+)?"
        r"([\w.+-]+(?:\s+[\w.+-]+){0,3})\s+head\b",
        text,
    ):
        feature = match.group(1).strip()
        if (
            feature not in features
            and "speculative decoding" not in feature
            and not re.fullmatch(SPECULATIVE_PATTERN, feature)
        ):
            features.append(feature)
    # A reference model supplies both the likely family and target parameter scale.
    reference = re.search(r"\b(?:like|similar to|based on)\s+([\w.-]+(?:/[\w.-]+)?)", request, re.I)
    reference_model = reference.group(1).rstrip(".,;") if reference else None
    size = re.search(r"\b(\d+(?:\.\d+)?)\s*b\b", reference_model or request, re.I)
    memory = re.search(r"\b(\d+(?:\.\d+)?)\s*gb\b", request, re.I)
    budget = re.search(
        r"(?:budget|maximum inference budget)[^$\n]{0,35}"
        r"\$\s*(\d+(?:\.\d+)?)",
        request,
        re.I,
    )
    cost = re.search(
        r"(?:gpu hourly cost|gpu cost|rental cost)[^$\n]{0,20}\$\s*(\d+(?:\.\d+)?)", request, re.I
    )
    tokens = re.search(
        r"(?:maximum|max|up to)\s+(?:of\s+)?(\d[\d,]*)\s+(?:generated\s+)?tokens", request, re.I
    )
    license_match = re.search(r"\b(any|apache-2\.0|mit|llama|gemma)\s+license\b", request, re.I)
    family = None
    if reference_model:
        family = re.split(r"[-_.]?\d", reference_model.rsplit("/", 1)[-1], maxsplit=1)[0]
    else:
        family_match = re.search(r"\b(qwen|llama|mistral|gemma|phi|deepseek)(?=\d|\b)", request, re.I)
        family = family_match.group(1) if family_match else None
    return {
        "original_request": original_request,
        "normalized_request": request,
        "speculative_decoding": speculative_preference(request, preference),
        "reference_model": reference_model,
        "family": family,
        "target_params_billion": float(size.group(1)) if size else None,
        "features": features,
        "use_case": re.search(
            r"(?:for|to)\s+(.+?)(?:\s+(?:with|on|using|for one)\b|[,;.]|$)", request, re.I
        ).group(1)
        if re.search(r"(?:for|to)\s+(.+?)(?:\s+(?:with|on|using|for one)\b|[,;.]|$)", request, re.I)
        else None,
        "gpu_type": (re.search(r"\b(?:A100|H100|L40S|RTX\s*\d{4})\b", request, re.I) or [None])[0],
        "gpu_memory_gb": float(memory.group(1)) if memory else None,
        "budget_usd_per_hour": float(budget.group(1)) if budget else None,
        "gpu_hourly_cost_usd": float(cost.group(1)) if cost else None,
        "max_generation_tokens": int(tokens.group(1).replace(",", "")) if tokens else None,
        "license": license_match.group(1).lower() if license_match else None,
        "framework_mode": "sweep"
        if "sweep mode" in text
        else "fast"
        if "fast mode" in text
        else None,
        "framework": next(
            (name for name in ("vllm", "sglang", "transformers", "trt-llm") if name in text), None
        ),
        "recent": any(term in text for term in ("new", "recent", "latest")),
    }


def discovery_inputs(request: str, supplied: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    Combine user-supplied constraints with those parsed from the request string.
    Args:
        request (str): The user request string containing model requirements and constraints.
        supplied (dict[str, Any]): A dictionary of user-supplied constraints.
    Returns:
        tuple[dict[str, Any], list[str]]: A tuple containing a dictionary of combined constraints and a list of missing keys.
    """
    parsed = extract_constraints(request)
    values = {"request": request}
    for key in (
        "gpu_memory_gb",
        "budget_usd_per_hour",
        "gpu_hourly_cost_usd",
        "license",
        "framework_mode",
        "speculative_decoding",
    ):
        values[key] = supplied.get(key) if supplied.get(key) is not None else parsed.get(key)
    values["speculative_decoding"] = speculative_preference(request, values["speculative_decoding"])
    # Cost is a user price assumption, never inferred from GPU memory or budget.
    missing = [key for key, value in values.items() if value is None]
    return values, missing

