"""BERTimbau (neuralmind/bert-base-portuguese-cased) fine-tuning for NER.

Optimized for token classification on Apple Silicon M4 (MPS) or CUDA/CPU.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ner_llm_vs_ml.data.lener import ID2LABEL, LABEL2ID, LABEL_NAMES


class TokenClassificationDataset(Dataset):
    """Dataset with subword alignment for word-level NER tags."""

    def __init__(
        self,
        sents: list[list[str]],
        labels: list[list[str]] | None,
        tokenizer: Any,
        max_length: int = 128,
    ):
        self.sents = sents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sents)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        words = self.sents[idx]
        word_labels = self.labels[idx] if self.labels else None

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if word_labels:
            word_ids = encoding.word_ids(batch_index=0)
            aligned_labels = []
            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)  # Special tokens [CLS], [SEP], [PAD]
                elif word_idx != prev_word_idx:
                    # First subword gets the actual tag
                    tag_str = word_labels[word_idx]
                    aligned_labels.append(LABEL2ID.get(tag_str, 0))
                else:
                    # Subsequent subwords ignored in loss computation
                    aligned_labels.append(-100)
                prev_word_idx = word_idx

            item["labels"] = torch.tensor(aligned_labels, dtype=torch.long)

        return item


class BertNERBaseline:
    """Wrapper around HuggingFace AutoModelForTokenClassification."""

    def __init__(
        self,
        pretrained: str = "neuralmind/bert-base-portuguese-cased",
        lr: float = 3e-5,
        epochs: int = 3,
        batch_size: int = 16,
        max_length: int = 128,
        device: str | None = None,
        seed: int = 42,
    ):
        self.pretrained = pretrained
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.pretrained,
            num_labels=len(LABEL_NAMES),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        ).to(self.device)

        self.train_time_sec: float = 0.0

    def fit(
        self,
        sents: list[list[str]],
        labels: list[list[str]],
        val_sents: list[list[str]] | None = None,
        val_labels: list[list[str]] | None = None,
    ) -> BertNERBaseline:
        """Fine-tune BERT on training sentences and BIO labels."""
        start_time = time.time()
        torch.manual_seed(self.seed)

        train_dataset = TokenClassificationDataset(
            sents, labels, self.tokenizer, self.max_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
        )

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                target_labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=target_labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(train_loader))
            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f}")

        self.train_time_sec = time.time() - start_time
        return self

    def predict_with_confidence(
        self,
        sents: list[list[str]],
    ) -> tuple[list[list[str]], list[float]]:
        """Predict BIO sequence and return per-sentence average confidence scores."""
        self.model.eval()
        all_preds = []
        all_confidences = []

        with torch.no_grad():
            for words in sents:
                encoding = self.tokenizer(
                    words,
                    is_split_into_words=True,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                word_ids = encoding.word_ids(batch_index=0)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(0)  # [seq_len, num_labels]
                probs = torch.softmax(logits, dim=-1)

                predictions = torch.argmax(probs, dim=-1).cpu().numpy()
                probs_np = probs.cpu().numpy()

                sent_labels = []
                sent_probs = []
                prev_word_idx = None

                for idx, word_idx in enumerate(word_ids):
                    if word_idx is None:
                        continue
                    if word_idx != prev_word_idx:
                        pred_tag_id = predictions[idx]
                        sent_labels.append(ID2LABEL[pred_tag_id])
                        sent_probs.append(float(probs_np[idx, pred_tag_id]))
                    prev_word_idx = word_idx

                # If sentence was truncated, pad remaining words with 'O'
                while len(sent_labels) < len(words):
                    sent_labels.append("O")
                    sent_probs.append(0.5)

                # Ensure exact length match
                sent_labels = sent_labels[: len(words)]
                sent_probs = sent_probs[: len(words)]

                avg_conf = float(np.mean(sent_probs)) if sent_probs else 0.5
                all_preds.append(sent_labels)
                all_confidences.append(avg_conf)

        return all_preds, all_confidences

    def predict(self, sents: list[list[str]]) -> list[list[str]]:
        preds, _ = self.predict_with_confidence(sents)
        return preds

    def save(self, dirpath: Path | str) -> None:
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(dirpath)
        self.tokenizer.save_pretrained(dirpath)

    @classmethod
    def load(cls, dirpath: Path | str, device: str | None = None) -> BertNERBaseline:
        instance = cls.__new__(cls)
        instance.pretrained = str(dirpath)
        if device is None:
            instance.device = (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        else:
            instance.device = torch.device(device)
        instance.tokenizer = AutoTokenizer.from_pretrained(dirpath)
        instance.model = AutoModelForTokenClassification.from_pretrained(dirpath).to(
            instance.device
        )
        return instance
