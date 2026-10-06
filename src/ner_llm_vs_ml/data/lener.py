"""Data loader and preprocessing for LeNER-Br dataset.

Directly parses CoNLL files from the official repository and provides clean
HuggingFace Datasets / pandas interfaces.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

LABEL_NAMES = [
    "O",
    "B-ORGANIZACAO",
    "I-ORGANIZACAO",
    "B-PESSOA",
    "I-PESSOA",
    "B-TEMPO",
    "I-TEMPO",
    "B-LOCAL",
    "I-LOCAL",
    "B-LEGISLACAO",
    "I-LEGISLACAO",
    "B-JURISPRUDENCIA",
    "I-JURISPRUDENCIA",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABEL_NAMES)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_NAMES)}

ENTITY_TYPES = ["ORGANIZACAO", "PESSOA", "TEMPO", "LOCAL", "LEGISLACAO", "JURISPRUDENCIA"]

BASE_URL = "https://raw.githubusercontent.com/peluz/lener-br/master/leNER-Br/"
FILE_MAP = {
    "train": "train/train.conll",
    "validation": "dev/dev.conll",
    "test": "test/test.conll",
}


def get_cache_dir() -> Path:
    """Return local cache directory for dataset files."""
    cache_dir = Path.home() / ".cache" / "ner_bench" / "lener_br"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_conll_file(split: str, force: bool = False) -> Path:
    """Download CoNLL file for a given split if not already cached."""
    cache_dir = get_cache_dir()
    local_path = cache_dir / f"{split}.conll"

    if local_path.exists() and not force:
        return local_path

    url = BASE_URL + FILE_MAP[split]
    print(f"Downloading {split} from {url}...")
    urllib.request.urlretrieve(url, local_path)
    return local_path


def parse_conll_file(filepath: Path | str) -> list[dict[str, Any]]:
    """Parse CoNLL file into a list of sentence dictionaries."""
    filepath = Path(filepath)
    sentences: list[dict[str, Any]] = []

    tokens: list[str] = []
    ner_tags: list[int] = []
    ner_labels: list[str] = []

    sentence_idx = 0
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(
                        {
                            "id": f"{filepath.stem}_{sentence_idx}",
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                            "ner_labels": ner_labels,
                            "text": " ".join(tokens),
                        }
                    )
                    sentence_idx += 1
                    tokens = []
                    ner_tags = []
                    ner_labels = []
                continue

            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                tag = parts[-1]
                tag_id = LABEL2ID.get(tag, 0)

                tokens.append(token)
                ner_tags.append(tag_id)
                ner_labels.append(tag)

        # Flush last sentence if exists
        if tokens:
            sentences.append(
                {
                    "id": f"{filepath.stem}_{sentence_idx}",
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                    "ner_labels": ner_labels,
                    "text": " ".join(tokens),
                }
            )

    return sentences


def extract_entities_from_labels(tokens: list[str], labels: list[str]) -> list[dict[str, Any]]:
    """Extract named entities as spans and text from BIO label sequence."""
    entities = []
    current_label: str | None = None
    current_tokens: list[str] = []
    start_idx = 0

    for idx, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith("B-"):
            if current_label and current_tokens:
                entities.append(
                    {
                        "label": current_label,
                        "text": " ".join(current_tokens),
                        "token_start": start_idx,
                        "token_end": idx,
                    }
                )
            current_label = label[2:]
            current_tokens = [token]
            start_idx = idx
        elif label.startswith("I-"):
            tag_type = label[2:]
            if current_label == tag_type:
                current_tokens.append(token)
            else:
                if current_label and current_tokens:
                    entities.append(
                        {
                            "label": current_label,
                            "text": " ".join(current_tokens),
                            "token_start": start_idx,
                            "token_end": idx,
                        }
                    )
                current_label = tag_type
                current_tokens = [token]
                start_idx = idx
        else:  # "O"
            if current_label and current_tokens:
                entities.append(
                    {
                        "label": current_label,
                        "text": " ".join(current_tokens),
                        "token_start": start_idx,
                        "token_end": idx,
                    }
                )
                current_label = None
                current_tokens = []

    if current_label and current_tokens:
        entities.append(
            {
                "label": current_label,
                "text": " ".join(current_tokens),
                "token_start": start_idx,
                "token_end": len(tokens),
            }
        )

    return entities


def load_lener_br(force_download: bool = False) -> DatasetDict:
    """Load LeNER-Br dataset as a Hugging Face DatasetDict."""
    features = Features(
        {
            "id": Value("string"),
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=LABEL_NAMES)),
            "ner_labels": Sequence(Value("string")),
            "text": Value("string"),
        }
    )

    splits = {}
    for split in ["train", "validation", "test"]:
        path = download_conll_file(split, force=force_download)
        data = parse_conll_file(path)
        splits[split] = Dataset.from_list(data, features=features)

    return DatasetDict(splits)


def sample_train_split(
    dataset: Dataset,
    n_samples: int | str,
    seed: int = 42,
) -> Dataset:
    """Subsample training dataset for learning curve experiments.

    Args:
        dataset: The full training Dataset.
        n_samples: Number of sentences to sample, or 'all'.
        seed: Random seed for reproducible sampling.
    """
    if n_samples == "all" or (isinstance(n_samples, int) and n_samples >= len(dataset)):
        return dataset

    assert isinstance(n_samples, int)
    return dataset.shuffle(seed=seed).select(range(n_samples))
