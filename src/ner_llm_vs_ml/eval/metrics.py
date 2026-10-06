"""Evaluation metrics for NER: Strict (seqeval) and Relaxed (overlap) F1,

with bootstrap confidence intervals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2


def compute_strict_metrics(
    y_true: list[list[str]],
    y_pred: list[list[str]],
) -> dict[str, Any]:
    """Compute strict entity-level precision, recall, and F1 using seqeval.

    Both the exact entity boundaries and the entity type must match.
    """
    overall_p = precision_score(y_true, y_pred, mode="strict", scheme=IOB2)
    overall_r = recall_score(y_true, y_pred, mode="strict", scheme=IOB2)
    overall_f1 = f1_score(y_true, y_pred, mode="strict", scheme=IOB2)

    # Detailed report as dict
    report = classification_report(
        y_true,
        y_pred,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0,
    )

    return {
        "strict_precision": float(overall_p),
        "strict_recall": float(overall_r),
        "strict_f1": float(overall_f1),
        "per_entity_strict": report,
    }


def compute_relaxed_metrics(
    ground_truth_entities: list[list[dict[str, Any]]],
    predicted_entities: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compute relaxed entity-level metrics based on span overlap.

    A predicted entity is considered a true positive if:
    1. Its label matches the gold label.
    2. Its span overlaps by at least 1 character (or token) with the gold entity.

    Each gold entity can be matched at most once.
    """
    total_gold = 0
    total_pred = 0
    true_positives = 0

    per_entity_counts: dict[str, dict[str, int]] = {}

    for gold_list, pred_list in zip(ground_truth_entities, predicted_entities):
        total_gold += len(gold_list)
        total_pred += len(pred_list)

        matched_gold_indices = set()

        for pred in pred_list:
            pred_label = pred["label"]
            pred_start = pred.get("token_start")
            pred_end = pred.get("token_end")

            if pred_label not in per_entity_counts:
                per_entity_counts[pred_label] = {"tp": 0, "fp": 0, "fn": 0}

            matched = False
            for g_idx, gold in enumerate(gold_list):
                if g_idx in matched_gold_indices:
                    continue
                if gold["label"] != pred_label:
                    continue

                g_start = gold.get("token_start")
                g_end = gold.get("token_end")

                # Check overlap
                if pred_start is not None and g_start is not None:
                    overlap = max(0, min(pred_end, g_end) - max(pred_start, g_start)) > 0
                else:
                    # Fallback to string containment / exact match
                    overlap = (
                        gold["text"].lower() in pred["text"].lower()
                        or pred["text"].lower() in gold["text"].lower()
                    )

                if overlap:
                    matched = True
                    matched_gold_indices.add(g_idx)
                    true_positives += 1
                    per_entity_counts[pred_label]["tp"] += 1
                    break

            if not matched:
                per_entity_counts[pred_label]["fp"] += 1

        # Count false negatives for unmatched gold entities
        for g_idx, gold in enumerate(gold_list):
            if g_idx not in matched_gold_indices:
                g_label = gold["label"]
                if g_label not in per_entity_counts:
                    per_entity_counts[g_label] = {"tp": 0, "fp": 0, "fn": 0}
                per_entity_counts[g_label]["fn"] += 1

    precision = true_positives / total_pred if total_pred > 0 else 0.0
    recall = true_positives / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_entity_results = {}
    for label, counts in per_entity_counts.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_entity_results[label] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": tp + fn,
        }

    return {
        "relaxed_precision": float(precision),
        "relaxed_recall": float(recall),
        "relaxed_f1": float(f1),
        "per_entity_relaxed": per_entity_results,
    }


def bootstrap_f1_ci(
    y_true: list[list[str]],
    y_pred: list[list[str]],
    n_bootstraps: int = 500,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute F1 score and non-parametric bootstrap confidence interval over sentences.

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point_f1 = f1_score(y_true, y_pred, mode="strict", scheme=IOB2)

    boot_f1s = []
    for _ in range(n_bootstraps):
        indices = rng.integers(0, n, size=n)
        sample_true = [y_true[i] for i in indices]
        sample_pred = [y_pred[i] for i in indices]
        boot_f1s.append(f1_score(sample_true, sample_pred, mode="strict", scheme=IOB2))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_f1s, alpha * 100))
    upper = float(np.percentile(boot_f1s, (1.0 - alpha) * 100))

    return float(point_f1), lower, upper
