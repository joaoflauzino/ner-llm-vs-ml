"""CLI commands for running NER benchmarks, learning curves, and cascade routing."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ner_llm_vs_ml.data.lener import (
    extract_entities_from_labels,
    load_lener_br,
    sample_train_split,
)
from ner_llm_vs_ml.eval.hallucination import (
    aggregate_hallucination_report,
    analyze_sentence_hallucinations,
)
from ner_llm_vs_ml.eval.metrics import compute_relaxed_metrics, compute_strict_metrics
from ner_llm_vs_ml.models.crf import CRFBaseline

app = typer.Typer(help="NER Benchmark: ML Tradicional vs LLM (LeNER-Br)")
console = Console()


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


@app.command()
def test_data():
    """Verify loading and statistics of the LeNER-Br dataset."""
    console.print("[bold green]Loading LeNER-Br dataset...[/bold green]")
    ds = load_lener_br()
    table = Table(title="Estatísticas do LeNER-Br")
    table.add_column("Split", style="cyan")
    table.add_column("Sentenças", justify="right")

    table.add_row("Treino", f"{len(ds['train']):,}")
    table.add_row("Validação", f"{len(ds['validation']):,}")
    table.add_row("Teste", f"{len(ds['test']):,}")
    console.print(table)


@app.command()
def train_crf(
    train_size: str = typer.Option("all", help="Número de sentenças para treino (ex: 100, 500, all)"),
    test_size: int = typer.Option(0, help="Número de sentenças de teste (0 para conjunto completo)"),
    output_dir: Path = typer.Option(Path("results/crf"), help="Diretório para salvar resultados"),
):
    """Treina e avalia o modelo tradicional CRF."""
    console.print(f"[bold blue]Carregando dados (treino: {train_size})...[/bold blue]")
    ds = load_lener_br()

    n_samples: int | str = int(train_size) if train_size.isdigit() else "all"
    train_data = sample_train_split(ds["train"], n_samples)
    test_data = ds["test"].select(range(test_size)) if test_size > 0 else ds["test"]

    train_sents = [s["tokens"] for s in train_data]
    train_labels = [s["ner_labels"] for s in train_data]
    test_sents = [s["tokens"] for s in test_data]
    test_labels = [s["ner_labels"] for s in test_data]

    console.print(f"Treinando CRF com {len(train_sents)} sentenças...")
    crf = CRFBaseline()
    crf.fit(train_sents, train_labels)
    console.print(f"[green]Treinado em {crf.train_time_sec:.2f}s![/green]")

    console.print(f"Avaliando em {len(test_sents)} sentenças de teste...")
    preds, _confidences = crf.predict_with_confidence(test_sents)

    strict = compute_strict_metrics(test_labels, preds)
    gold_entities = [
        extract_entities_from_labels(t, l) for t, l in zip(test_sents, test_labels)
    ]
    pred_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, preds)]
    relaxed = compute_relaxed_metrics(gold_entities, pred_entities)

    table = Table(title=f"Resultados CRF (N={len(train_sents)})")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="magenta")

    table.add_row("Tempo de Treino", f"{crf.train_time_sec:.2f}s")
    table.add_row("Strict F1", f"{strict['strict_f1']:.4f}")
    table.add_row("Strict Precision", f"{strict['strict_precision']:.4f}")
    table.add_row("Strict Recall", f"{strict['strict_recall']:.4f}")
    table.add_row("Relaxed F1", f"{relaxed['relaxed_f1']:.4f}")
    table.add_row("Relaxed Precision", f"{relaxed['relaxed_precision']:.4f}")
    table.add_row("Relaxed Recall", f"{relaxed['relaxed_recall']:.4f}")
    console.print(table)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"strict": strict, "relaxed": relaxed}, f, indent=2, default=_json_default)


@app.command()
def train_bert(
    epochs: int = typer.Option(3, help="Número de épocas de fine-tuning"),
    batch_size: int = typer.Option(16, help="Tamanho do batch"),
    train_size: str = typer.Option("all", help="Amostras de treino"),
    test_size: int = typer.Option(0, help="Amostras de teste (0 para todas)"),
    output_dir: Path = typer.Option(Path("results/bert"), help="Diretório de saída"),
):
    """Fine-tuning do BERTimbau no Apple Silicon M4 (MPS)."""
    from ner_llm_vs_ml.models.bert import BertNERBaseline

    console.print("[bold blue]Inicializando BERTimbau-base em MPS...[/bold blue]")
    ds = load_lener_br()
    n_samples: int | str = int(train_size) if train_size.isdigit() else "all"
    train_data = sample_train_split(ds["train"], n_samples)
    test_data = ds["test"].select(range(test_size)) if test_size > 0 else ds["test"]

    bert = BertNERBaseline(epochs=epochs, batch_size=batch_size)
    console.print(f"Dispositivo selecionado: [bold]{bert.device}[/bold]")

    train_sents = [s["tokens"] for s in train_data]
    train_labels = [s["ner_labels"] for s in train_data]
    test_sents = [s["tokens"] for s in test_data]
    test_labels = [s["ner_labels"] for s in test_data]

    bert.fit(train_sents, train_labels)
    preds, _confs = bert.predict_with_confidence(test_sents)

    strict = compute_strict_metrics(test_labels, preds)
    gold_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, test_labels)]
    pred_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, preds)]
    relaxed = compute_relaxed_metrics(gold_entities, pred_entities)

    table = Table(title=f"Resultados BERTimbau (N={len(train_sents)})")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="green")

    table.add_row("Tempo de Treino", f"{bert.train_time_sec:.2f}s")
    table.add_row("Strict F1", f"{strict['strict_f1']:.4f}")
    table.add_row("Relaxed F1", f"{relaxed['relaxed_f1']:.4f}")
    console.print(table)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"strict": strict, "relaxed": relaxed}, f, indent=2, default=_json_default)


@app.command()
def run_gliner(
    test_size: int = typer.Option(200, help="Amostras de teste (default: 200)"),
    threshold: float = typer.Option(0.4, help="Threshold de probabilidade do GLiNER"),
    output_dir: Path = typer.Option(Path("results/gliner"), help="Diretório de saída"),
):
    """Executa o baseline zero-shot do GLiNER multilingual."""
    from ner_llm_vs_ml.models.gliner_model import GLiNERBaseline

    console.print("[bold blue]Carregando GLiNER multilingual v2.1...[/bold blue]")
    gliner = GLiNERBaseline(threshold=threshold)
    ds = load_lener_br()
    test_data = ds["test"].select(range(test_size)) if test_size > 0 else ds["test"]

    test_sents = [s["tokens"] for s in test_data]
    test_labels = [s["ner_labels"] for s in test_data]

    console.print(f"Rodando inferência em {len(test_sents)} sentenças...")
    preds = gliner.predict(test_sents)

    strict = compute_strict_metrics(test_labels, preds)
    gold_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, test_labels)]
    pred_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, preds)]
    relaxed = compute_relaxed_metrics(gold_entities, pred_entities)

    table = Table(title=f"Resultados GLiNER Zero-Shot (N={len(test_sents)})")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="yellow")
    table.add_row("Tempo de Inferência", f"{gliner.inference_time_sec:.2f}s")
    table.add_row("Strict F1", f"{strict['strict_f1']:.4f}")
    table.add_row("Relaxed F1", f"{relaxed['relaxed_f1']:.4f}")
    console.print(table)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"strict": strict, "relaxed": relaxed}, f, indent=2, default=_json_default)


@app.command()
def run_llm(
    model_id: str = typer.Option("gpt-4o-mini", help="ID do modelo OpenAI"),
    k_shot: int = typer.Option(0, help="Número de exemplos few-shot (0, 5, 20)"),
    test_size: int = typer.Option(200, help="Amostras de teste (default piloto: 200)"),
    budget_usd: float = typer.Option(5.0, help="Limite máximo de orçamento em USD"),
    output_dir: Path = typer.Option(Path("results/llm"), help="Diretório de saída"),
):
    """Executa a extração via LLM com controle de orçamento e cache."""
    from ner_llm_vs_ml.llm.cache import BudgetTracker
    from ner_llm_vs_ml.llm.client import LLMNERClient

    budget = BudgetTracker(budget_limit_usd=budget_usd)
    client = LLMNERClient(model_id=model_id, budget_tracker=budget)

    ds = load_lener_br()
    test_data = ds["test"].select(range(test_size)) if test_size > 0 else ds["test"]
    test_sents = [s["tokens"] for s in test_data]
    test_labels = [s["ner_labels"] for s in test_data]

    few_shots = None
    if k_shot > 0:
        few_shots = list(ds["train"].select(range(k_shot)))

    console.print(f"[bold blue]Iniciando extração com {model_id} ({k_shot}-shot) em {len(test_sents)} sentenças...[/bold blue]")
    preds, raw_entities_list, _meta = client.predict(test_sents, few_shot_examples=few_shots)

    # Métricas de F1
    strict = compute_strict_metrics(test_labels, preds)
    gold_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, test_labels)]
    pred_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, preds)]
    relaxed = compute_relaxed_metrics(gold_entities, pred_entities)

    # Análise de alucinação
    sent_analyses = []
    for s, raw_ents in zip(test_data, raw_entities_list):
        sent_analyses.append(analyze_sentence_hallucinations(s["text"], raw_ents))
    hallucination_report = aggregate_hallucination_report(sent_analyses)

    table = Table(title=f"Resultados LLM {model_id} ({k_shot}-shot)")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="green")

    table.add_row("Strict F1", f"{strict['strict_f1']:.4f}")
    table.add_row("Relaxed F1", f"{relaxed['relaxed_f1']:.4f}")
    table.add_row("Taxa Verbatim", f"{hallucination_report['verbatim_rate']*100:.1f}%")
    table.add_row("Taxa Normalizada", f"{hallucination_report['normalized_rate']*100:.1f}%")
    table.add_row("Taxa Alucinação Pura", f"{hallucination_report['hallucinated_rate']*100:.1f}%")
    table.add_row("Custo Estimado (USD)", f"${budget.total_cost_usd:.4f}")
    table.add_row("Tokens Entrada", f"{budget.total_prompt_tokens:,}")
    table.add_row("Tokens Saída", f"{budget.total_completion_tokens:,}")
    table.add_row("Hits de Cache", f"{budget.cached_hits}")
    console.print(table)

    out = output_dir / f"{model_id}_{k_shot}shot"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(
            {
                "strict": strict,
                "relaxed": relaxed,
                "hallucination": hallucination_report,
                "budget": budget.summary(),
            },
            f,
            indent=2,
            default=_json_default,
        )


def main():
    app()


if __name__ == "__main__":
    main()
