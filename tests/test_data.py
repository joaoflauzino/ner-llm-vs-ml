from datasets import Dataset

from ner_llm_vs_ml.data.lener import (
    ID2LABEL,
    LABEL2ID,
    LABEL_NAMES,
    extract_entities_from_labels,
    sample_train_split,
)


def test_label_mappings():
    assert len(LABEL_NAMES) == 13
    assert LABEL_NAMES[0] == "O"
    assert LABEL2ID["B-PESSOA"] == 3
    assert ID2LABEL[3] == "B-PESSOA"


def test_extract_entities_from_labels():
    tokens = ["O", "STF", "julgou", "Lula", "em", "Brasília", "."]
    labels = ["O", "B-ORGANIZACAO", "O", "B-PESSOA", "O", "B-LOCAL", "O"]

    entities = extract_entities_from_labels(tokens, labels)
    assert len(entities) == 3
    assert entities[0] == {
        "label": "ORGANIZACAO",
        "text": "STF",
        "token_start": 1,
        "token_end": 2,
    }
    assert entities[1] == {
        "label": "PESSOA",
        "text": "Lula",
        "token_start": 3,
        "token_end": 4,
    }
    assert entities[2] == {
        "label": "LOCAL",
        "text": "Brasília",
        "token_start": 5,
        "token_end": 6,
    }


def test_extract_multitoken_entities():
    tokens = ["Lei", "nº", "8.666/93", "foi", "revogada", "."]
    labels = ["B-LEGISLACAO", "I-LEGISLACAO", "I-LEGISLACAO", "O", "O", "O"]

    entities = extract_entities_from_labels(tokens, labels)
    assert len(entities) == 1
    assert entities[0]["label"] == "LEGISLACAO"
    assert entities[0]["text"] == "Lei nº 8.666/93"
    assert entities[0]["token_start"] == 0
    assert entities[0]["token_end"] == 3


def test_subsampling():
    raw_data = [{"id": f"s_{i}", "tokens": ["a"], "ner_tags": [0], "ner_labels": ["O"], "text": "a"} for i in range(100)]
    ds = Dataset.from_list(raw_data)

    s25 = sample_train_split(ds, 25, seed=42)
    assert len(s25) == 25

    s_all = sample_train_split(ds, "all")
    assert len(s_all) == 100
