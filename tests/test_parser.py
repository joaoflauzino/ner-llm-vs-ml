from ner_llm_vs_ml.llm.parser import align_entities_to_tokens, parse_llm_json


def test_parse_clean_json():
    raw = '{"entities": [{"text": "STF", "label": "ORGANIZACAO"}, {"text": "Lula", "label": "PESSOA"}]}'
    res = parse_llm_json(raw)
    assert len(res) == 2
    assert res[0] == {"text": "STF", "label": "ORGANIZACAO"}
    assert res[1] == {"text": "Lula", "label": "PESSOA"}


def test_parse_markdown_json():
    raw = 'Aqui estão as entidades:\n```json\n{"entities": [{"text": "Lei 8.666", "label": "LEGISLACAO"}]}\n```'
    res = parse_llm_json(raw)
    assert len(res) == 1
    assert res[0] == {"text": "Lei 8.666", "label": "LEGISLACAO"}


def test_align_entities_to_tokens():
    tokens = ["O", "STF", "decidiu", "em", "Brasília", "."]
    entities = [
        {"text": "STF", "label": "ORGANIZACAO"},
        {"text": "Brasília", "label": "LOCAL"},
    ]

    bio = align_entities_to_tokens(tokens, entities)
    assert bio == ["O", "B-ORGANIZACAO", "O", "O", "B-LOCAL", "O"]


def test_align_multitoken_entity():
    tokens = ["O", "Supremo", "Tribunal", "Federal", "decidiu", "."]
    entities = [{"text": "Supremo Tribunal Federal", "label": "ORGANIZACAO"}]

    bio = align_entities_to_tokens(tokens, entities)
    assert bio == ["O", "B-ORGANIZACAO", "I-ORGANIZACAO", "I-ORGANIZACAO", "O", "O"]
