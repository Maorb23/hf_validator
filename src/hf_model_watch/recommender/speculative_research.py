"""Bounded, method-specific discovery research; never treats search hits as support."""
from __future__ import annotations

import re
from typing import Any, Callable
from urllib.parse import urlsplit

from hf_model_watch.recommender.utilities import documented_pairs, pair_method


class ResearchBudgetExceeded(ValueError):
    pass


class ResearchBudget:
    def __init__(self, search: Callable, settings: dict[str, Any]):
        self.search = search
        self.search_limit = int(settings["tavily"].get("discovery_search_budget", 32))
        self.extract_limit = int(settings["tavily"].get("discovery_extract_budget", 24))
        self.searches = self.extracts = 0
        if self.search_limit < 0 or self.extract_limit < 0:
            raise ValueError("Research budgets must be nonnegative")

    def __call__(self, query, settings, *, extract_urls=None):
        if extract_urls:
            if self.extracts >= self.extract_limit:
                raise ResearchBudgetExceeded("Extraction budget exhausted")
            self.extracts += 1
            return self.search(query, settings, extract_urls=extract_urls)
        if self.searches >= self.search_limit:
            raise ResearchBudgetExceeded("Search budget exhausted")
        self.searches += 1
        return self.search(query, settings)


def research_methods(selected, evidence, settings, order, search, progress=None):
    coverage = {item["model_id"]: {m: {"status": "not_researched"} for m in order} for item in selected}
    method_limit = int(settings["tavily"].get("discovery_methods_per_target", 3))
    page_limit = int(settings["tavily"].get("discovery_draft_pages_per_method", 2))
    if method_limit < 0 or not 0 <= page_limit <= 8:
        raise ValueError("Invalid speculative method/page limits")
    # Research each method across the shortlist before spending budget on the next method.
    for method in order[:method_limit]:
        for item in selected:
            ident = item["model_id"]
            query = f'"{ident}" {method} speculative decoding draft target model site:huggingface.co OR site:github.com'
            info = coverage[ident][method]
            if progress:
                progress(f"Tavily {method} research: {ident}...")
            try:
                found = search(query, settings)
            except ResearchBudgetExceeded:
                info["reason"] = "search_budget_exhausted"
                continue
            except Exception as exc:
                info.update(status="failed", error_type=type(exc).__name__)
                continue
            pages = []
            for source in found:
                parsed = urlsplit(str(source.get("source_url") or ""))
                parts = parsed.path.strip("/").split("/")
                if parsed.scheme == "https" and parsed.hostname == "huggingface.co" and len(parts) >= 2:
                    url = f"https://huggingface.co/{parts[0]}/{parts[1]}"
                    if not any(t in url.lower() for t in ("gguf", "gptq")) and url not in pages:
                        pages.append(url)
            pages.sort(key=lambda url: re.sub(r"[-_]", "", method) not in re.sub(r"[-_]", "", url.lower()))
            extracted = []
            urls = [f"https://huggingface.co/{ident}", *[p for p in pages if p != f"https://huggingface.co/{ident}"][:page_limit]]
            try:
                extracted = search(query, settings, extract_urls=urls)
            except ResearchBudgetExceeded:
                info["extraction"] = "budget_exhausted"
            except Exception as exc:
                info["extract_error_type"] = type(exc).__name__
            sources = found + extracted
            matching = []
            for source in sources:
                pairs = documented_pairs(item["record"], [source], order)
                text = str(source.get("retrieved_text") or "")
                if ident.lower() in text.lower() or ident.lower() in str(source.get("source_url", "")).lower() or pairs:
                    matching.append(source)
            evidence[ident].extend(matching)
            pairs = [p for p in documented_pairs(item["record"], evidence[ident], order) if pair_method(p) == method]
            pattern = {"dspark": r"d[-_]?spark", "dflash": r"d[-_]?flash", "mtp": r"mtp|multi.token"}.get(method, method)
            mentioned = any(re.search(pattern, str(s.get("source_url", "")) + " " + str(s.get("retrieved_text", "")), re.I) for s in sources)
            info.update(status="documented" if pairs else "found_unverified" if mentioned else "not_found",
                        source_urls=list(dict.fromkeys(
                            p["source_url"] for p in pairs)) if pairs else list(dict.fromkeys(
                            s["source_url"] for s in sources if s.get("source_url") and
                            re.search(pattern, str(s.get("source_url", "")) + " " + str(s.get("retrieved_text", "")), re.I))))
            if mentioned and not pairs:
                info["reason"] = "exact_target_and_runtime_not_documented"
    return coverage
