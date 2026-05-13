from __future__ import annotations

import json
import time
import uuid
import os
import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Callable
import urllib.request
import urllib.error

from .metrics import (
    MetricResult,
    refusal_detection,
    response_length,
    coherence,
    lexical_richness,
    relevance,
    toxicity_heuristic,
    hallucination_surface,
    latency,
    _REFUSAL_RE,
)


@dataclass
class TestCase:
    id: str
    prompt: str
    should_refuse: bool = False
    expected_keywords: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    min_words: int = 10
    max_words: int = 300
    description: str = ""


@dataclass
class TestResult:
    test_case_id: str
    prompt: str
    response: str
    response_time_ms: float
    metrics: list
    composite_score: float
    passed: bool
    error: Optional[str] = None
    run_id: str = ""
    tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class RunSummary:
    run_id: str
    endpoint_url: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    error_cases: int
    pass_rate: float
    mean_composite_score: float
    mean_latency_ms: float
    metric_breakdown: dict
    results: list
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

def _call_openai_compatible(url: str, prompt: str, api_key: str = "",
                             model: str = "gpt-3.5-turbo",
                             timeout: int = 30) -> tuple:
    """
    Calls any OpenAI-compatible /v1/chat/completions endpoint.
    Returns (response_text, latency_ms).
    """
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        t1 = time.perf_counter()
        text = body["choices"][0]["message"]["content"]
        return text, (t1 - t0) * 1000
    except Exception as e:
        raise RuntimeError(f"Endpoint call failed: {e}") from e


def _call_anthropic(url: str, prompt: str, api_key: str = "",
                    model: str = "claude-haiku-4-5-20251001",
                    timeout: int = 30) -> tuple:
    """
    Calls the Anthropic Messages API directly.
    Returns (response_text, latency_ms).
    """
    payload = json.dumps({
        "model": model,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        t1 = time.perf_counter()
        text = body["content"][0]["text"]
        return text, (t1 - t0) * 1000
    except Exception as e:
        raise RuntimeError(f"Anthropic endpoint call failed: {e}") from e


def _call_generic_post(url: str, prompt: str, api_key: str = "",
                        prompt_field: str = "message",
                        response_field: str = "response",
                        timeout: int = 30) -> tuple:
    """Generic JSON POST adapter for custom endpoints."""
    payload = json.dumps({prompt_field: prompt}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        t1 = time.perf_counter()
        text = body
        for part in response_field.split("."):
            text = text[part]
        return str(text), (t1 - t0) * 1000
    except Exception as e:
        raise RuntimeError(f"Endpoint call failed: {e}") from e


METRIC_WEIGHTS = {
    "refusal_detection":     0.20,
    "response_length":       0.08,
    "coherence":             0.12,
    "lexical_richness":      0.08,
    "relevance":             0.20,
    "toxicity_heuristic":    0.15,
    "hallucination_surface": 0.10,
    "latency":               0.07,
}


class EvalRunner:
    """
    Orchestrates test execution against a single endpoint.

    endpoint_type options:
      "openai"     → POST /v1/chat/completions (OpenAI-compatible)
      "anthropic"  → POST /v1/messages (Anthropic Messages API)
      "generic"    → POST with configurable field names
    """

    def __init__(
        self,
        endpoint_url: str,
        endpoint_type: str = "openai",
        api_key: str = "",
        model: str = "gpt-3.5-turbo",
        prompt_field: str = "message",
        response_field: str = "response",
        timeout: int = 30,
    ):
        self.endpoint_url   = endpoint_url
        self.endpoint_type  = endpoint_type
        self.api_key        = api_key
        self.model          = model
        self.prompt_field   = prompt_field
        self.response_field = response_field
        self.timeout        = timeout

    def _call(self, prompt: str) -> tuple:
        if self.endpoint_type == "anthropic":
            return _call_anthropic(
                self.endpoint_url, prompt, self.api_key, self.model, self.timeout
            )
        elif self.endpoint_type == "openai":
            return _call_openai_compatible(
                self.endpoint_url, prompt, self.api_key, self.model, self.timeout
            )
        else:
            return _call_generic_post(
                self.endpoint_url, prompt, self.api_key,
                self.prompt_field, self.response_field, self.timeout
            )

    def _score_response(self, tc: TestCase, response: str,
                         response_time_ms: float) -> tuple:
        results = []

        refused = bool(_REFUSAL_RE.search(response))

        results.append(refusal_detection(response, tc.should_refuse))
        results.append(response_length(response, tc.min_words, tc.max_words))
        results.append(coherence(response))
        results.append(lexical_richness(response))
        results.append(relevance(tc.prompt, response,
                                  should_refuse=tc.should_refuse, refused=refused))
        results.append(toxicity_heuristic(response))
        results.append(hallucination_surface(response))
        results.append(latency(response_time_ms))

        total_weight = sum(METRIC_WEIGHTS.get(r.name, 0.1) for r in results)
        composite = sum(r.score * METRIC_WEIGHTS.get(r.name, 0.1) for r in results)
        composite /= total_weight

        return results, round(composite, 4)

    def run(self, test_cases: list,
            on_progress: Optional[Callable] = None) -> RunSummary:
        run_id  = str(uuid.uuid4())[:8]
        results = []
        errors  = 0

        for i, tc in enumerate(test_cases):
            try:
                response, rt_ms = self._call(tc.prompt)
                metric_results, composite = self._score_response(tc, response, rt_ms)
                passed = all(r.passed for r in metric_results)
                tr = TestResult(
                    test_case_id=tc.id,
                    prompt=tc.prompt,
                    response=response,
                    response_time_ms=rt_ms,
                    metrics=metric_results,
                    composite_score=composite,
                    passed=passed,
                    run_id=run_id,
                    tags=tc.tags,
                )
            except Exception as e:
                tr = TestResult(
                    test_case_id=tc.id,
                    prompt=tc.prompt,
                    response="",
                    response_time_ms=0,
                    metrics=[],
                    composite_score=0.0,
                    passed=False,
                    error=str(e),
                    run_id=run_id,
                    tags=tc.tags,
                )
                errors += 1

            results.append(tr)
            if on_progress:
                on_progress(i + 1, len(test_cases), tr)

        passed_count = sum(1 for r in results if r.passed and not r.error)
        failed_count = sum(1 for r in results if not r.passed and not r.error)
        non_error    = [r for r in results if not r.error]
        mean_score   = (sum(r.composite_score for r in non_error) / len(non_error)) if non_error else 0.0
        mean_lat     = (sum(r.response_time_ms for r in non_error) / len(non_error)) if non_error else 0.0

        metric_breakdown: dict = {}
        for r in non_error:
            for m in r.metrics:
                if m.name not in metric_breakdown:
                    metric_breakdown[m.name] = {"scores": [], "pass_count": 0}
                metric_breakdown[m.name]["scores"].append(m.score)
                if m.passed:
                    metric_breakdown[m.name]["pass_count"] += 1

        for name, data in metric_breakdown.items():
            scores = data["scores"]
            data["mean_score"] = round(sum(scores) / len(scores), 4) if scores else 0.0
            data["pass_rate"]  = round(data["pass_count"] / len(scores), 4) if scores else 0.0
            del data["scores"]

        return RunSummary(
            run_id=run_id,
            endpoint_url=self.endpoint_url,
            total_cases=len(test_cases),
            passed_cases=passed_count,
            failed_cases=failed_count,
            error_cases=errors,
            pass_rate=round(passed_count / len(test_cases), 4) if test_cases else 0.0,
            mean_composite_score=round(mean_score, 4),
            mean_latency_ms=round(mean_lat, 1),
            metric_breakdown=metric_breakdown,
            results=results,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )

    def run_from_file(self, path: str, **kwargs) -> RunSummary:
        with open(path) as f:
            raw = json.load(f)
        test_cases = [TestCase(**tc) for tc in raw]
        return self.run(test_cases, **kwargs)

    def save_results(self, summary: RunSummary, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"Results saved → {path}")
