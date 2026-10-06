from ner_llm_vs_ml.eval.metrics import (
    bootstrap_f1_ci,
    compute_relaxed_metrics,
    compute_strict_metrics,
)


def test_strict_metrics_perfect():
    y_true = [["B-PESSOA", "I-PESSOA", "O", "B-LOCAL"]]
    y_pred = [["B-PESSOA", "I-PESSOA", "O", "B-LOCAL"]]

    res = compute_strict_metrics(y_true, y_pred)
    assert res["strict_f1"] == 1.0
    assert res["strict_precision"] == 1.0
    assert res["strict_recall"] == 1.0


def test_strict_metrics_mismatch():
    y_true = [["B-PESSOA", "I-PESSOA", "O"]]
    y_pred = [["B-PESSOA", "O", "O"]]  # truncated boundary

    res = compute_strict_metrics(y_true, y_pred)
    assert res["strict_f1"] == 0.0  # seqeval requires exact boundary


def test_relaxed_metrics_boundary_overlap():
    # Gold: "Supremo Tribunal Federal" (0-3)
    # Pred: "Tribunal Federal" (1-3) - same label, overlapping span
    gold = [[{"label": "ORGANIZACAO", "text": "Supremo Tribunal Federal", "token_start": 0, "token_end": 3}]]
    pred = [[{"label": "ORGANIZACAO", "text": "Tribunal Federal", "token_start": 1, "token_end": 3}]]

    res = compute_relaxed_metrics(gold, pred)
    assert res["relaxed_f1"] == 1.0
    assert res["relaxed_precision"] == 1.0
    assert res["relaxed_recall"] == 1.0


def test_relaxed_metrics_label_mismatch():
    gold = [[{"label": "ORGANIZACAO", "text": "STF", "token_start": 0, "token_end": 1}]]
    pred = [[{"label": "PESSOA", "text": "STF", "token_start": 0, "token_end": 1}]]

    res = compute_relaxed_metrics(gold, pred)
    assert res["relaxed_f1"] == 0.0


def test_bootstrap_f1_ci():
    y_true = [["B-PESSOA", "O"], ["O", "B-LOCAL"]] * 20
    y_pred = [["B-PESSOA", "O"], ["O", "B-LOCAL"]] * 20

    f1, low, high = bootstrap_f1_ci(y_true, y_pred, n_bootstraps=50, seed=42)
    assert f1 == 1.0
    assert low == 1.0
    assert high == 1.0
