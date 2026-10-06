"""Disk cache and budget enforcement for LLM API calls.

Guarantees 100% reproducibility and prevents exceeding the user budget (e.g. $5.00).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class BudgetTracker:
    """Tracks token spend and enforces hard spending limits."""

    def __init__(
        self,
        budget_limit_usd: float = 5.0,
        cost_per_million_input: float = 0.15,
        cost_per_million_output: float = 0.60,
    ):
        self.budget_limit_usd = budget_limit_usd
        self.cost_per_million_input = cost_per_million_input
        self.cost_per_million_output = cost_per_million_output

        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_calls: int = 0
        self.cached_hits: int = 0

    @property
    def total_cost_usd(self) -> float:
        input_cost = (self.total_prompt_tokens / 1_000_000.0) * self.cost_per_million_input
        output_cost = (self.total_completion_tokens / 1_000_000.0) * self.cost_per_million_output
        return input_cost + output_cost

    def check_budget_available(self) -> None:
        if self.total_cost_usd >= self.budget_limit_usd:
            raise RuntimeError(
                f"BUDGET LIMIT REACHED: Spent ${self.total_cost_usd:.4f} of ${self.budget_limit_usd:.2f} limit. "
                "Halting further API calls to prevent unexpected charges."
            )

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_calls += 1

    def summary(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "cached_hits": self.cached_hits,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": self.total_cost_usd,
            "budget_limit_usd": self.budget_limit_usd,
            "budget_remaining_usd": max(0.0, self.budget_limit_usd - self.total_cost_usd),
        }


class ResponseCache:
    """Persistent SHA-256 disk cache for LLM responses."""

    def __init__(self, cache_dir: Path | str | None = None):
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "ner_bench" / "llm_responses"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, model: str, prompt: str, temperature: float) -> str:
        payload = f"{model}::{temperature}::{prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, model: str, prompt: str, temperature: float) -> dict[str, Any] | None:
        key = self._hash_key(model, prompt, temperature)
        filepath = self.cache_dir / f"{key}.json"
        if filepath.exists():
            try:
                with open(filepath, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def set(
        self,
        model: str,
        prompt: str,
        temperature: float,
        response_text: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        key = self._hash_key(model, prompt, temperature)
        filepath = self.cache_dir / f"{key}.json"
        data = {
            "key": key,
            "model": model,
            "temperature": temperature,
            "prompt": prompt,
            "response_text": response_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
