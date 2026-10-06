"""GLiNER baseline for zero-shot and few-shot Named Entity Recognition.

Extracts spans directly using predefined entity label names without full fine-tuning.
"""

from __future__ import annotations

import time
from typing import Any

from gliner import GLiNER

from ner_llm_vs_ml.data.lener import ENTITY_TYPES
from ner_llm_vs_ml.llm.parser import align_entities_to_tokens


class GLiNERBaseline:
    """Wrapper around GLiNER multilingual model for zero-shot NER."""

    def __init__(
        self,
        pretrained: str = "urchade/gliner_multi-v2.1",
        labels: list[str] | None = None,
        threshold: float = 0.4,
        device: str = "cpu",
    ):
        self.pretrained = pretrained
        self.labels = labels or ENTITY_TYPES
        self.threshold = threshold
        self.device = device
        self.model = GLiNER.from_pretrained(self.pretrained).to(self.device)
        self.inference_time_sec: float = 0.0

    def predict_entities(self, text: str) -> list[dict[str, Any]]:
        """Predict entities for raw sentence string."""
        return self.model.predict_entities(
            text,
            self.labels,
            threshold=self.threshold,
        )

    def predict(self, sents: list[list[str]]) -> list[list[str]]:
        """Predict BIO sequence for test sentences."""
        start_time = time.time()
        preds = []
        for tokens in sents:
            text = " ".join(tokens)
            raw_entities = self.predict_entities(text)

            clean_entities = [
                {"text": ent["text"], "label": ent["label"]}
                for ent in raw_entities
                if ent.get("label") in self.labels
            ]

            bio = align_entities_to_tokens(tokens, clean_entities)
            preds.append(bio)

        self.inference_time_sec = time.time() - start_time
        return preds
