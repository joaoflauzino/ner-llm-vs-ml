"""spaCy NER baseline.

Trains a lightweight transition-based NER parser using spacy.blank("pt")
or a pretrained pipeline.
"""

from __future__ import annotations

import time
from pathlib import Path

import spacy
from spacy.tokens import Doc
from spacy.training import Example


class SpacyNERBaseline:
    """Wrapper for training and inference with spaCy's transition-based NER parser."""

    def __init__(self, n_iter: int = 20, seed: int = 42):
        self.n_iter = n_iter
        self.seed = seed
        self.nlp = spacy.blank("pt")
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner", last=True)
        else:
            self.ner = self.nlp.get_pipe("ner")
        self.train_time_sec: float = 0.0

    def fit(
        self,
        sents: list[list[str]],
        labels: list[list[str]],
    ) -> SpacyNERBaseline:
        """Train the NER pipe from token lists and BIO labels."""
        start_time = time.time()
        spacy.util.fix_random_seed(self.seed)

        examples: list[Example] = []
        for tokens, bio_tags in zip(sents, labels):
            doc = Doc(self.nlp.vocab, words=tokens)
            # Add labels to pipe
            for tag in bio_tags:
                if tag.startswith(("B-", "I-")):
                    self.ner.add_label(tag[2:])

            example = Example.from_dict(doc, {"entities": self._bio_to_biluo(tokens, bio_tags)})
            examples.append(example)

        optimizer = self.nlp.initialize(lambda: examples)

        for _ in range(self.n_iter):
            losses: dict[str, float] = {}
            batches = spacy.util.minibatch(examples, size=spacy.util.compounding(4.0, 32.0, 1.001))
            for batch in batches:
                self.nlp.update(batch, drop=0.2, losses=losses, sgd=optimizer)

        self.train_time_sec = time.time() - start_time
        return self

    def _bio_to_biluo(self, tokens: list[str], bio_tags: list[str]) -> list[str]:
        """Convert BIO tags to BILUO for spaCy training."""
        biluo = ["O"] * len(tokens)
        n = len(bio_tags)

        i = 0
        while i < n:
            tag = bio_tags[i]
            if tag.startswith("B-"):
                label = tag[2:]
                # Look ahead for I- tags
                j = i + 1
                while j < n and bio_tags[j] == f"I-{label}":
                    j += 1
                length = j - i
                if length == 1:
                    biluo[i] = f"U-{label}"
                else:
                    biluo[i] = f"B-{label}"
                    for k in range(i + 1, j - 1):
                        biluo[k] = f"I-{label}"
                    biluo[j - 1] = f"L-{label}"
                i = j
            else:
                biluo[i] = "O"
                i += 1
        return biluo

    def predict(self, sents: list[list[str]]) -> list[list[str]]:
        """Predict BIO sequence for test sentences."""
        preds: list[list[str]] = []
        for tokens in sents:
            doc = Doc(self.nlp.vocab, words=tokens)
            doc = self.nlp(doc)

            bio = ["O"] * len(tokens)
            for ent in doc.ents:
                start = ent.start
                end = ent.end
                if start < len(bio):
                    bio[start] = f"B-{ent.label_}"
                    for idx in range(start + 1, min(end, len(bio))):
                        bio[idx] = f"I-{ent.label_}"
            preds.append(bio)
        return preds

    def save(self, dirpath: Path | str) -> None:
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(dirpath)

    @classmethod
    def load(cls, dirpath: Path | str) -> SpacyNERBaseline:
        instance = cls()
        instance.nlp = spacy.load(dirpath)
        return instance
