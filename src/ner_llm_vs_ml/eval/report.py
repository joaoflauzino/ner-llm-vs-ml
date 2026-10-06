"""Generates consolidated benchmark reports and Markdown tables for article publication."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def generate_comparison_report(
    results_dir: Path | str = "results",
    output_markdown: Path | str = "results/comparison_table.md",
) -> None:
    results_path = Path(results_dir)
    models_data = []

    # 1. CRF
    crf_path = results_path / "crf" / "metrics.json"
    if crf_path.exists():
        with open(crf_path, encoding="utf-8") as f:
            d = json.load(f)
            models_data.append(
                {
                    "Modelo": "CRF (ML Tradicional)",
                    "Tipo": "Supervisionado (N=all)",
                    "Strict F1": d["strict"]["strict_f1"],
                    "Relaxed F1": d["relaxed"]["relaxed_f1"],
                    "Strict P": d["strict"]["strict_precision"],
                    "Strict R": d["strict"]["strict_recall"],
                    "Custo (USD)": "$0.00",
                    "Ambiente": "Local CPU (14s)",
                    "per_entity": d["strict"].get("per_entity_strict", {}),
                }
            )

    # 2. GLiNER
    gliner_path = results_path / "gliner" / "metrics.json"
    if gliner_path.exists():
        with open(gliner_path, encoding="utf-8") as f:
            d = json.load(f)
            models_data.append(
                {
                    "Modelo": "GLiNER v2.1",
                    "Tipo": "Zero-Shot Local",
                    "Strict F1": d["strict"]["strict_f1"],
                    "Relaxed F1": d["relaxed"]["relaxed_f1"],
                    "Strict P": d["strict"].get("strict_precision", 0.0),
                    "Strict R": d["strict"].get("strict_recall", 0.0),
                    "Custo (USD)": "$0.00",
                    "Ambiente": "Local CPU (6s)",
                    "per_entity": d["strict"].get("per_entity_strict", {}),
                }
            )

    # 3. spaCy
    spacy_path = results_path / "spacy" / "metrics.json"
    if spacy_path.exists():
        with open(spacy_path, encoding="utf-8") as f:
            d = json.load(f)
            models_data.append(
                {
                    "Modelo": "spaCy",
                    "Tipo": "Supervisionado",
                    "Strict F1": d["strict"]["strict_f1"],
                    "Relaxed F1": d["relaxed"]["relaxed_f1"],
                    "Strict P": d["strict"].get("strict_precision", 0.0),
                    "Strict R": d["strict"].get("strict_recall", 0.0),
                    "Custo (USD)": "$0.00",
                    "Ambiente": "Local CPU",
                    "per_entity": d["strict"].get("per_entity_strict", {}),
                }
            )

    # 4. BERTimbau
    bert_path = results_path / "bert" / "metrics.json"
    if bert_path.exists():
        with open(bert_path, encoding="utf-8") as f:
            d = json.load(f)
            models_data.append(
                {
                    "Modelo": "BERTimbau-base",
                    "Tipo": "Supervisionado (M4 MPS)",
                    "Strict F1": d["strict"]["strict_f1"],
                    "Relaxed F1": d["relaxed"]["relaxed_f1"],
                    "Strict P": d["strict"].get("strict_precision", 0.0),
                    "Strict R": d["strict"].get("strict_recall", 0.0),
                    "Custo (USD)": "$0.00",
                    "Ambiente": "Apple M4 (MPS)",
                    "per_entity": d["strict"].get("per_entity_strict", {}),
                }
            )

    # 5. LLMs
    llm_dir = results_path / "llm"
    if llm_dir.exists():
        for summary_file in sorted(llm_dir.glob("*/summary.json")):
            with open(summary_file, encoding="utf-8") as f:
                d = json.load(f)
                model_name = summary_file.parent.name
                cost = d.get("budget", {}).get("total_cost_usd", 0.0)
                models_data.append(
                    {
                        "Modelo": f"LLM ({model_name})",
                        "Tipo": "Zero/Few-Shot API",
                        "Strict F1": d["strict"]["strict_f1"],
                        "Relaxed F1": d["relaxed"]["relaxed_f1"],
                        "Strict P": d["strict"]["strict_precision"],
                        "Strict R": d["strict"]["strict_recall"],
                        "Custo (USD)": f"${cost:.4f}",
                        "Ambiente": "OpenAI API",
                        "per_entity": d["strict"].get("per_entity_strict", {}),
                    }
                )

    if not models_data:
        console.print("[bold yellow]Nenhum resultado encontrado em results/.[/bold yellow]")
        return

    # Tabela rica no terminal
    table = Table(title="Benchmark Consolidado: ML Tradicional vs LLM (LeNER-Br)")
    table.add_column("Modelo", style="cyan", no_wrap=True)
    table.add_column("Paradigma", style="magenta")
    table.add_column("Strict F1", justify="right", style="bold green")
    table.add_column("Relaxed F1", justify="right", style="green")
    table.add_column("Strict Prec.", justify="right")
    table.add_column("Strict Rec.", justify="right")
    table.add_column("Custo (USD)", justify="right", style="yellow")
    table.add_column("Ambiente", style="blue")

    for m in models_data:
        table.add_row(
            m["Modelo"],
            m["Tipo"],
            f"{m['Strict F1']:.4f}",
            f"{m['Relaxed F1']:.4f}",
            f"{m['Strict P']:.4f}",
            f"{m['Strict R']:.4f}",
            m["Custo (USD)"],
            m["Ambiente"],
        )

    console.print(table)

    # Gerar Markdown para o artigo
    md_lines = [
        "# Tabela Comparativa: ML Tradicional vs LLM (LeNER-Br)",
        "",
        "| Modelo | Paradigma | Strict F1 | Relaxed F1 | Strict Precision | Strict Recall | Custo API | Ambiente |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in models_data:
        md_lines.append(
            f"| **{m['Modelo']}** | {m['Tipo']} | **{m['Strict F1']:.4f}** | {m['Relaxed F1']:.4f} | {m['Strict P']:.4f} | {m['Strict R']:.4f} | {m['Custo (USD)']} | {m['Ambiente']} |"
        )

    labels = ["JURISPRUDENCIA", "LEGISLACAO", "ORGANIZACAO", "PESSOA", "TEMPO", "LOCAL"]
    md_lines.extend(
        [
            "",
            "## F1-Score por Tipo de Entidade (Strict)",
            "",
            "| Modelo | " + " | ".join(labels) + " |",
            "|---| " + " | ".join(["---"] * len(labels)) + " |",
        ]
    )

    for m in models_data:
        pe = m.get("per_entity", {})
        row = [f"**{m['Modelo']}**"]
        for lbl in labels:
            f1_val = pe.get(lbl, {}).get("f1-score", 0.0)
            row.append(f"{f1_val:.4f}")
        md_lines.append("| " + " | ".join(row) + " |")

    out_file = Path(output_markdown)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    console.print(f"\n[bold green]Tabela Markdown salva em: {out_file}[/bold green]")
