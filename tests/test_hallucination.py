from ner_llm_vs_ml.eval.hallucination import (
    aggregate_hallucination_report,
    analyze_sentence_hallucinations,
)


def test_verbatim_entity():
    source = "O Ministro Celso de Mello deferiu o pedido."
    preds = [{"text": "Celso de Mello", "label": "PESSOA"}]

    res = analyze_sentence_hallucinations(source, preds)
    assert res["verbatim_count"] == 1
    assert res["hallucinated_count"] == 0
    assert res["normalized_count"] == 0


def test_hallucinated_entity():
    source = "O Tribunal negou provimento ao recurso."
    preds = [
        {"text": "Tribunal", "label": "ORGANIZACAO"},
        {"text": "Supremo Tribunal Federal", "label": "ORGANIZACAO"},  # Not in source!
    ]

    res = analyze_sentence_hallucinations(source, preds)
    assert res["verbatim_count"] == 1
    assert res["hallucinated_count"] == 1
    assert res["hallucinations"][0]["returned_text"] == "Supremo Tribunal Federal"


def test_normalized_entity():
    source = "Conforme acórdão 1234/2018 publicado ontem."
    # LLM altered casing / punctuation: "Acórdão 1234/2018"
    preds = [{"text": "ACÓRDÃO 1234/2018", "label": "JURISPRUDENCIA"}]

    res = analyze_sentence_hallucinations(source, preds)
    assert res["normalized_count"] == 1
    assert res["verbatim_count"] == 0
    assert res["hallucinated_count"] == 0


def test_aggregate_report():
    a1 = analyze_sentence_hallucinations("Texto 1 com STF", [{"text": "STF", "label": "ORGANIZACAO"}])
    a2 = analyze_sentence_hallucinations("Texto 2", [{"text": "STJ", "label": "ORGANIZACAO"}])

    rep = aggregate_hallucination_report([a1, a2])
    assert rep["total_predicted"] == 2
    assert rep["verbatim_count"] == 1
    assert rep["hallucinated_count"] == 1
    assert rep["hallucinated_rate"] == 0.5
