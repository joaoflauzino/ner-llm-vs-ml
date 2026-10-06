"""Conditional Random Field (CRF) baseline for NER.

Fast, highly interpretable discriminative sequence classifier running on CPU.
Extracts rich morphological, orthographic, and contextual n-gram features.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn_crfsuite


def word2features(sent: list[str], i: int) -> dict[str, Any]:
    """Extract features for token at index i in sentence."""
    word = sent[i]

    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word.has_digit()": any(c.isdigit() for c in word),
        "word.has_punct()": any(c in ".,-/:;" for c in word),
        "prefix-2": word[:2],
        "prefix-3": word[:3],
        "suffix-2": word[-2:],
        "suffix-3": word[-3:],
        "length": len(word),
    }

    # Previous word features
    if i > 0:
        prev_word = sent[i - 1]
        features.update(
            {
                "-1:word.lower()": prev_word.lower(),
                "-1:word.istitle()": prev_word.istitle(),
                "-1:word.isupper()": prev_word.isupper(),
                "-1:word.has_punct()": any(c in ".,-/:;" for c in prev_word),
            }
        )
    else:
        features["BOS"] = True

    # Next word features
    if i < len(sent) - 1:
        next_word = sent[i + 1]
        features.update(
            {
                "+1:word.lower()": next_word.lower(),
                "+1:word.istitle()": next_word.istitle(),
                "+1:word.isupper()": next_word.isupper(),
                "+1:word.has_punct()": any(c in ".,-/:;" for c in next_word),
            }
        )
    else:
        features["EOS"] = True

    return features


def sent2features(sent: list[str]) -> list[dict[str, Any]]:
    return [word2features(sent, i) for i in range(len(sent))]


class CRFBaseline:
    """Wrapper around sklearn-crfsuite CRF model."""

    def __init__(
        self,
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
        random_state: int = 42,
    ):
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=self.c1,
            c2=self.c2,
            max_iterations=self.max_iterations,
            all_possible_transitions=True,
        )
        self.train_time_sec: float = 0.0

    def fit(self, sents: list[list[str]], labels: list[list[str]]) -> CRFBaseline:
        """Fit CRF on sentences and corresponding BIO labels."""
        start_time = time.time()
        x_train = [sent2features(s) for s in sents]
        self.model.fit(x_train, labels)
        self.train_time_sec = time.time() - start_time
        return self

    def predict(self, sents: list[list[str]]) -> list[list[str]]:
        """Predict BIO sequence for test sentences."""
        x_test = [sent2features(s) for s in sents]
        return self.model.predict(x_test)

    def predict_with_confidence(
        self,
        sents: list[list[str]],
    ) -> tuple[list[list[str]], list[float]]:
        """Predict BIO sequences along with sentence-level marginal confidence scores.

        Confidence is the geometric mean of token marginal probabilities.
        """
        x_test = [sent2features(s) for s in sents]
        y_preds = self.model.predict(x_test)
        marginals = self.model.predict_marginals(x_test)

        confidences = []
        for sent_pred, sent_marg in zip(y_preds, marginals):
            token_probs = []
            for pred_tag, token_marg_dict in zip(sent_pred, sent_marg):
                p = token_marg_dict.get(pred_tag, 0.0)
                token_probs.append(max(p, 1e-6))
            # Geometric mean of token marginals
            conf = float(np.exp(np.mean(np.log(token_probs))))
            confidences.append(conf)

        return y_preds, confidences

    def save(self, filepath: Path | str) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path | str) -> CRFBaseline:
        with open(filepath, "rb") as f:
            return pickle.load(f)
