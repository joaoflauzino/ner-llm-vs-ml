"""Cascade routing between local ML models and LLMs based on confidence calibration."""

from __future__ import annotations

from typing import Any

import numpy as np

from ner_llm_vs_ml.data.lener import extract_entities_from_labels
from ner_llm_vs_ml.eval.metrics import compute_relaxed_metrics, compute_strict_metrics


class CascadeRouter:
    """Routes ambiguous or low-confidence sentences to an LLM while keeping

    high-confidence sentences on the lightweight local model.
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def evaluate_cascade(
        self,
        local_preds: list[list[str]],
        local_confidences: list[float],
        llm_preds: list[list[str]],
        gold_labels: list[list[str]],
        tokens_list: list[list[str]],
        cost_per_llm_call: float = 0.0003,  # ~$0.30 per 1k sentences
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Simulate cascade at a given confidence threshold tau."""
        tau = threshold if threshold is not None else self.threshold

        n = len(gold_labels)
        cascade_preds = []
        llm_routed_count = 0

        for l_pred, conf, m_pred in zip(local_preds, local_confidences, llm_preds):
            if conf < tau:
                cascade_preds.append(m_pred)
                llm_routed_count += 1
            else:
                cascade_preds.append(l_pred)

        strict = compute_strict_metrics(gold_labels, cascade_preds)

        gold_entities = [
            extract_entities_from_labels(tok, lab) for tok, lab in zip(tokens_list, gold_labels)
        ]
        cascade_entities = [
            extract_entities_from_labels(tok, lab) for tok, lab in zip(tokens_list, cascade_preds)
        ]
        relaxed = compute_relaxed_metrics(gold_entities, cascade_entities)

        routing_rate = llm_routed_count / n if n > 0 else 0.0
        total_cost = llm_routed_count * cost_per_llm_call

        return {
            "threshold": tau,
            "llm_routed_count": llm_routed_count,
            "routing_rate": routing_rate,
            "total_cost_usd": total_cost,
            "strict_f1": strict["strict_f1"],
            "relaxed_f1": relaxed["relaxed_f1"],
        }

    def sweep_thresholds(
        self,
        local_preds: list[list[str]],
        local_confidences: list[float],
        llm_preds: list[list[str]],
        gold_labels: list[list[str]],
        tokens_list: list[list[str]],
        thresholds: list[float] | None = None,
        cost_per_llm_call: float = 0.0003,
    ) -> list[dict[str, Any]]:
        """Sweep confidence thresholds tau in [0.0, 1.0] to produce Pareto frontier."""
        if thresholds is None:
            thresholds = list(np.linspace(0.0, 1.0, 21))

        curve = []
        for tau in thresholds:
            res = self.evaluate_cascade(
                local_preds=local_preds,
                local_confidences=local_confidences,
                llm_preds=llm_preds,
                gold_labels=gold_labels,
                tokens_list=tokens_list,
                cost_per_llm_call=cost_per_llm_call,
                threshold=float(tau),
            )
            curve.append(res)
        return curve
