"""Parser and offset mapper for LLM JSON outputs.

Converts JSON entity lists back into BIO token-level tags matching the original sentence tokens.
"""

from __future__ import annotations

import json
import re

from ner_llm_vs_ml.data.lener import ENTITY_TYPES


def parse_llm_json(raw_text: str) -> list[dict[str, str]]:
    """Robustly parse JSON output from LLM, handling markdown code blocks."""
    text = raw_text.strip()

    # Strip markdown ```json ... ``` code blocks
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "entities" in data:
            entities = data["entities"]
        elif isinstance(data, list):
            entities = data
        else:
            return []

        clean_entities = []
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            span_text = str(ent.get("text", "")).strip()
            label = str(ent.get("label", "")).strip().upper()

            # Normalize label
            if label.startswith(("B-", "I-")):
                label = label[2:]

            if span_text and label in ENTITY_TYPES:
                clean_entities.append({"text": span_text, "label": label})

        return clean_entities
    except (json.JSONDecodeError, TypeError, ValueError):
        # Fallback regex extraction of {"text": "...", "label": "..."}
        pattern = r'\{\s*"text"\s*:\s*"([^"]+)"\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}'
        found = re.findall(pattern, text)
        fallback = []
        for span_text, label in found:
            label_up = label.strip().upper()
            if label_up.startswith(("B-", "I-")):
                label_up = label_up[2:]
            if label_up in ENTITY_TYPES:
                fallback.append({"text": span_text.strip(), "label": label_up})
        return fallback


def align_entities_to_tokens(
    tokens: list[str],
    predicted_entities: list[dict[str, str]],
) -> list[str]:
    """Map predicted entity texts to BIO token labels corresponding to tokens.

    Returns:
        bio_labels: list[str] of same length as tokens.
    """
    bio_labels = ["O"] * len(tokens)
    used_token_indices = set()

    # Sort longer entities first so nested/sub-entities don't claim tokens prematurely
    sorted_entities = sorted(
        predicted_entities,
        key=lambda x: len(x["text"].split()),
        reverse=True,
    )

    for ent in sorted_entities:
        ent_text = ent["text"]
        label = ent["label"]
        ent_words = ent_text.split()
        if not ent_words:
            continue

        k = len(ent_words)
        match_start = -1

        # Search for exact token sequence match in available tokens
        for i in range(len(tokens) - k + 1):
            if any(idx in used_token_indices for idx in range(i, i + k)):
                continue

            # Compare token sequence
            candidate = " ".join(tokens[i : i + k]).lower()
            target = ent_text.lower()

            if candidate == target:
                match_start = i
                break

        # Fallback: single word or partial substring match
        if match_start == -1 and k == 1:
            for i in range(len(tokens)):
                if i in used_token_indices:
                    continue
                if tokens[i].lower() == ent_words[0].lower():
                    match_start = i
                    break

        if match_start != -1:
            bio_labels[match_start] = f"B-{label}"
            used_token_indices.add(match_start)
            for j in range(match_start + 1, match_start + k):
                bio_labels[j] = f"I-{label}"
                used_token_indices.add(j)

    return bio_labels
