"""CLI commands for running NER benchmarks, learning curves, and cascade routing."""

from __future__ import annotations

import json
import time
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
from ner_llm_vs_ml.eval.report import generate_comparison_report
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
    ds = load_lener_br()
    n_samples: int | str = int(train_size) if train_size.isdigit() else "all"
    train_data = sample_train_split(ds["train"], n_samples)
    test_data = ds["test"].select(range(test_size)) if test_size > 0 else ds["test"]

    train_sents = [s["tokens"] for s in train_data]
    train_labels = [s["ner_labels"] for s in train_data]
    test_sents = [s["tokens"] for s in test_data]
    test_labels = [s["ner_labels"] for s in test_data]

    console.print(f"[bold blue]Treinando CRF com {len(train_sents)} sentenças...[/bold blue]")
    crf = CRFBaseline()
    t0_train = time.perf_counter()
    crf.fit(train_sents, train_labels)
    t_train = time.perf_counter() - t0_train

    console.print(f"Avaliando em {len(test_sents)} sentenças de teste...")
    t0_infer = time.perf_counter()
    preds, _confidences = crf.predict_with_confidence(test_sents)
    t_infer = time.perf_counter() - t0_infer

    strict = compute_strict_metrics(test_labels, preds)
    gold_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, test_labels)]
    pred_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, preds)]
    relaxed = compute_relaxed_metrics(gold_entities, pred_entities)

    lat_ms = (t_infer / max(1, len(test_sents))) * 1000

    table = Table(title=f"Resultados CRF (N={len(train_sents)})")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="magenta")
    table.add_row("Tempo de Treino", f"{t_train:.2f}s")
    table.add_row("Tempo de Inferência", f"{t_infer:.2f}s ({lat_ms:.2f} ms/doc)")
    table.add_row("Strict F1", f"{strict['strict_f1']:.4f}")
    table.add_row("Relaxed F1", f"{relaxed['relaxed_f1']:.4f}")
    console.print(table)

    output_dir.mkdir(parents=True, exist_ok=True)
    timing = {
        "train_time_sec": t_train,
        "inference_time_sec": t_infer,
        "test_sentences": len(test_sents),
        "latency_ms_per_doc": lat_ms,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"strict": strict, "relaxed": relaxed, "timing": timing}, f, indent=2, default=_json_default)


@app.command()
def train_spacy(
    n_iter: int = typer.Option(15, help="Número de iterações de treino"),
    train_size: str = typer.Option("all", help="Número de sentenças para treino (ex: 500, all)"),
    test_size: int = typer.Option(0, help="Número de sentenças de teste (0 para todas)"),
    output_dir: Path = typer.Option(Path("results/spacy"), help="Diretório para salvar resultados"),
):
    """Treina e avalia o pipeline de NER do spaCy."""
    from ner_llm_vs_ml.models.spacy_model import SpacyNERBaseline

    ds = load_lener_br()
    n_samples: int | str = int(train_size) if train_size.isdigit() else "all"
    train_data = sample_train_split(ds["train"], n_samples)
    test_data = ds["test"].select(range(test_size)) if test_size > 0 else ds["test"]

    train_sents = [s["tokens"] for s in train_data]
    train_labels = [s["ner_labels"] for s in train_data]
    test_sents = [s["tokens"] for s in test_data]
    test_labels = [s["ner_labels"] for s in test_data]

    console.print(f"[bold blue]Treinando spaCy com {len(train_sents)} sentenças...[/bold blue]")
    sp = SpacyNERBaseline(n_iter=n_iter)
    t0_train = time.perf_counter()
    sp.fit(train_sents, train_labels)
    t_train = time.perf_counter() - t0_train

    console.print(f"Avaliando em {len(test_sents)} sentenças de teste...")
    t0_infer = time.perf_counter()
    preds = sp.predict(test_sents)
    t_infer = time.perf_counter() - t0_infer

    strict = compute_strict_metrics(test_labels, preds)
    gold_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, test_labels)]
    pred_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, preds)]
    relaxed = compute_relaxed_metrics(gold_entities, pred_entities)

    lat_ms = (t_infer / max(1, len(test_sents))) * 1000

    table = Table(title=f"Resultados spaCy (N={len(train_sents)})")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="magenta")
    table.add_row("Tempo de Treino", f"{t_train:.2f}s")
    table.add_row("Tempo de Inferência", f"{t_infer:.2f}s ({lat_ms:.2f} ms/doc)")
    table.add_row("Strict F1", f"{strict['strict_f1']:.4f}")
    table.add_row("Relaxed F1", f"{relaxed['relaxed_f1']:.4f}")
    console.print(table)

    output_dir.mkdir(parents=True, exist_ok=True)
    timing = {
        "train_time_sec": t_train,
        "inference_time_sec": t_infer,
        "test_sentences": len(test_sents),
        "latency_ms_per_doc": lat_ms,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"strict": strict, "relaxed": relaxed, "timing": timing}, f, indent=2, default=_json_default)


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

    ds = load_lener_br()
    n_samples: int | str = int(train_size) if train_size.isdigit() else "all"
    train_data = sample_train_split(ds["train"], n_samples)
    test_data = ds["test"].select(range(test_size)) if test_size > 0 else ds["test"]

    bert = BertNERBaseline(epochs=epochs, batch_size=batch_size)
    console.print(f"[bold blue]Treinando BERTimbau no dispositivo: {bert.device}[/bold blue]")

    train_sents = [s["tokens"] for s in train_data]
    train_labels = [s["ner_labels"] for s in train_data]
    test_sents = [s["tokens"] for s in test_data]
    test_labels = [s["ner_labels"] for s in test_data]

    t0_train = time.perf_counter()
    bert.fit(train_sents, train_labels)
    t_train = time.perf_counter() - t0_train

    t0_infer = time.perf_counter()
    preds, _confs = bert.predict_with_confidence(test_sents)
    t_infer = time.perf_counter() - t0_infer

    strict = compute_strict_metrics(test_labels, preds)
    gold_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, test_labels)]
    pred_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, preds)]
    relaxed = compute_relaxed_metrics(gold_entities, pred_entities)

    lat_ms = (t_infer / max(1, len(test_sents))) * 1000

    table = Table(title=f"Resultados BERTimbau (N={len(train_sents)})")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="green")
    table.add_row("Tempo de Treino", f"{t_train:.2f}s")
    table.add_row("Tempo de Inferência", f"{t_infer:.2f}s ({lat_ms:.2f} ms/doc)")
    table.add_row("Strict F1", f"{strict['strict_f1']:.4f}")
    table.add_row("Relaxed F1", f"{relaxed['relaxed_f1']:.4f}")
    console.print(table)

    output_dir.mkdir(parents=True, exist_ok=True)
    timing = {
        "train_time_sec": t_train,
        "inference_time_sec": t_infer,
        "test_sentences": len(test_sents),
        "latency_ms_per_doc": lat_ms,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"strict": strict, "relaxed": relaxed, "timing": timing}, f, indent=2, default=_json_default)


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
    t0_infer = time.perf_counter()
    preds = gliner.predict(test_sents)
    t_infer = time.perf_counter() - t0_infer

    strict = compute_strict_metrics(test_labels, preds)
    gold_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, test_labels)]
    pred_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, preds)]
    relaxed = compute_relaxed_metrics(gold_entities, pred_entities)

    lat_ms = (t_infer / max(1, len(test_sents))) * 1000

    table = Table(title=f"Resultados GLiNER Zero-Shot (N={len(test_sents)})")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="yellow")
    table.add_row("Tempo de Treino", "— (Zero-shot)")
    table.add_row("Tempo de Inferência", f"{t_infer:.2f}s ({lat_ms:.2f} ms/doc)")
    table.add_row("Strict F1", f"{strict['strict_f1']:.4f}")
    table.add_row("Relaxed F1", f"{relaxed['relaxed_f1']:.4f}")
    console.print(table)

    output_dir.mkdir(parents=True, exist_ok=True)
    timing = {
        "train_time_sec": 0.0,
        "inference_time_sec": t_infer,
        "test_sentences": len(test_sents),
        "latency_ms_per_doc": lat_ms,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"strict": strict, "relaxed": relaxed, "timing": timing}, f, indent=2, default=_json_default)


@app.command()
def run_llm(
    model_id: str = typer.Option("gpt-6-luna", help="ID do modelo OpenAI"),
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
    t0_infer = time.perf_counter()
    preds, raw_entities_list, _meta = client.predict(test_sents, few_shot_examples=few_shots)
    t_infer = time.perf_counter() - t0_infer

    strict = compute_strict_metrics(test_labels, preds)
    gold_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, test_labels)]
    pred_entities = [extract_entities_from_labels(t, l) for t, l in zip(test_sents, preds)]
    relaxed = compute_relaxed_metrics(gold_entities, pred_entities)

    sent_analyses = []
    for s, raw_ents in zip(test_data, raw_entities_list):
        sent_analyses.append(analyze_sentence_hallucinations(s["text"], raw_ents))
    hallucination_report = aggregate_hallucination_report(sent_analyses)

    lat_ms = (t_infer / max(1, len(test_sents))) * 1000

    table = Table(title=f"Resultados LLM {model_id} ({k_shot}-shot)")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="green")
    table.add_row("Strict F1", f"{strict['strict_f1']:.4f}")
    table.add_row("Relaxed F1", f"{relaxed['relaxed_f1']:.4f}")
    table.add_row("Tempo de Inferência", f"{t_infer:.2f}s ({lat_ms:.0f} ms/doc)")
    table.add_row("Taxa Verbatim", f"{hallucination_report['verbatim_rate']*100:.1f}%")
    table.add_row("Custo Estimado (USD)", f"${budget.total_cost_usd:.4f}")
    table.add_row("Hits de Cache", f"{budget.cached_hits}")
    console.print(table)

    out = output_dir / f"{model_id}_{k_shot}shot"
    out.mkdir(parents=True, exist_ok=True)
    timing = {
        "train_time_sec": 0.0,
        "inference_time_sec": t_infer,
        "test_sentences": len(test_sents),
        "latency_ms_per_doc": lat_ms,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(
            {
                "strict": strict,
                "relaxed": relaxed,
                "hallucination": hallucination_report,
                "budget": budget.summary(),
                "timing": timing,
            },
            f,
            indent=2,
            default=_json_default,
        )


@app.command()
def compare(
    output_markdown: Path = typer.Option(Path("results/comparison_table.md"), help="Arquivo Markdown gerado"),
):
    """Compara todos os modelos avaliados e gera tabela consolidada para o artigo."""
    generate_comparison_report(output_markdown=output_markdown)


@app.command()
def run_all(
    train_size: str = typer.Option("all", help="Sentenças de treino para os modelos supervisionados (ex: 500, all)"),
    test_size: int = typer.Option(200, help="Sentenças de teste para avaliação comum (default: 200)"),
    bert_epochs: int = typer.Option(3, help="Épocas de treino para o BERTimbau no M4"),
    skip_llm: bool = typer.Option(False, help="Se True, pula a chamada da LLM"),
    llm_model: str = typer.Option("gpt-6-luna", help="ID do modelo LLM"),
):
    """Executa TODOS os modelos sequencialmente, mede tempos de treino/inferência e gera tabela final."""
    console.print("\n[bold green]=======================================================[/bold green]")
    console.print("[bold green]🚀 INICIANDO BENCHMARK COMPLETO (ML TRADICIONAL VS LLM)[/bold green]")
    console.print(f"Configuração: treino={train_size}, teste={test_size} sentenças")
    console.print("[bold green]=======================================================[/bold green]\n")

    # 1. CRF
    console.print("\n[bold cyan]▶ [1/5] Executando CRF (ML Tradicional)...[/bold cyan]")
    train_crf(train_size=train_size, test_size=test_size)

    # 2. GLiNER
    console.print("\n[bold cyan]▶ [2/5] Executando GLiNER v2.1 (Zero-Shot Local)...[/bold cyan]")
    run_gliner(test_size=test_size)

    # 3. spaCy
    console.print("\n[bold cyan]▶ [3/5] Executando spaCy...[/bold cyan]")
    train_spacy(train_size=train_size, test_size=test_size, n_iter=15)

    # 4. BERTimbau
    console.print("\n[bold cyan]▶ [4/5] Executando BERTimbau-base (Apple Silicon M4 MPS)...[/bold cyan]")
    train_bert(epochs=bert_epochs, train_size=train_size, test_size=test_size, batch_size=16)

    # 5. LLM
    if not skip_llm:
        console.print(f"\n[bold cyan]▶ [5/5] Executando LLM ({llm_model} 0-shot)...[/bold cyan]")
        run_llm(model_id=llm_model, k_shot=0, test_size=test_size)
    else:
        console.print("\n[yellow]Pulando execução da LLM (--skip-llm ativado).[/yellow]")

    # Relatório Final
    console.print("\n[bold green]=======================================================[/bold green]")
    console.print("[bold green]📊 CONSOLIDANDO RESULTADOS EM TABELA[/bold green]")
    console.print("[bold green]=======================================================[/bold green]\n")
    compare()


def main():
    app()


if __name__ == "__main__":
    main()
