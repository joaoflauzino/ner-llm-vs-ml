"""Hallucination and fidelity analysis for LLM-extracted entities.

Measures how frequently generative models:
1. Fabricate text spans that do not exist in the source document (pure hallucination).
2. Alter or normalize text (changing accents, punctuation, capitalization, expanding abbreviations).
3. Produce duplicate or conflicting spans.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_text_loose(text: str) -> str:
    """Strip accents, lower case, remove non-alphanumeric for loose matching."""
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.lower().split())


def analyze_sentence_hallucinations(
    source_text: str,
    predicted_entities: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze fidelity of predicted entities against source sentence."""
    total_preds = len(predicted_entities)
    verbatim_count = 0
    normalized_count = 0
    hallucinated_count = 0

    hallucinations = []
    normalized_spans = []

    seen_spans = set()
    duplicates_count = 0

    norm_source = normalize_text_loose(source_text)

    for ent in predicted_entities:
        raw_text = ent.get("text", "").strip()
        label = ent.get("label", "UNKNOWN")

        if not raw_text:
            continue

        # Check duplicate
        span_key = (raw_text.lower(), label)
        if span_key in seen_spans:
            duplicates_count += 1
        seen_spans.add(span_key)

        # 1. Verbatim check
        if raw_text in source_text:
            verbatim_count += 1
        else:
            # 2. Check if it's an altered/normalized version of something in the source
            norm_raw = normalize_text_loose(raw_text)
            if norm_raw and norm_raw in norm_source:
                normalized_count += 1
                normalized_spans.append(
                    {
                        "original_source": source_text,
                        "returned_text": raw_text,
                        "label": label,
                        "issue": "altered_or_normalized",
                    }
                )
            else:
                # 3. Pure fabrication / hallucination
                hallucinated_count += 1
                hallucinations.append(
                    {
                        "original_source": source_text,
                        "returned_text": raw_text,
                        "label": label,
                        "issue": "not_in_text",
                    }
                )

    return {
        "total_predicted": total_preds,
        "verbatim_count": verbatim_count,
        "verbatim_rate": verbatim_count / total_preds if total_preds > 0 else 1.0,
        "normalized_count": normalized_count,
        "normalized_rate": normalized_count / total_preds if total_preds > 0 else 0.0,
        "hallucinated_count": hallucinated_count,
        "hallucinated_rate": hallucinated_count / total_preds if total_preds > 0 else 0.0,
        "duplicates_count": duplicates_count,
        "hallucinations": hallucinations,
        "normalized_spans": normalized_spans,
    }


def aggregate_hallucination_report(
    sentence_analyses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate hallucination analysis across all test sentences."""
    total_predicted = sum(a["total_predicted"] for a in sentence_analyses)
    total_verbatim = sum(a["verbatim_count"] for a in sentence_analyses)
    total_normalized = sum(a["normalized_count"] for a in sentence_analyses)
    total_hallucinated = sum(a["hallucinated_count"] for a in sentence_analyses)
    total_duplicates = sum(a["duplicates_count"] for a in sentence_analyses)

    all_hallucinations = [h for a in sentence_analyses for h in a["hallucinations"]]
    all_normalized = [n for a in sentence_analyses for n in a["normalized_spans"]]

    # Per-label hallucination breakdown
    hallucination_by_label: dict[str, int] = {}
    for h in all_hallucinations:
        lbl = h["label"]
        hallucination_by_label[lbl] = hallucination_by_label.get(lbl, 0) + 1

    return {
        "total_predicted": total_predicted,
        "verbatim_count": total_verbatim,
        "verbatim_rate": total_verbatim / total_predicted if total_predicted > 0 else 1.0,
        "normalized_count": total_normalized,
        "normalized_rate": total_normalized / total_predicted if total_predicted > 0 else 0.0,
        "hallucinated_count": total_hallucinated,
        "hallucinated_rate": (
            total_hallucinated / total_predicted if total_predicted > 0 else 0.0
        ),
        "duplicates_count": total_duplicates,
        "hallucination_by_label": hallucination_by_label,
        "sample_hallucinations": all_hallucinations[:10],
        "sample_normalized": all_normalized[:10],
    }
