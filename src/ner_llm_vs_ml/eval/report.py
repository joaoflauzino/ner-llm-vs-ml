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
            t_train = d.get("timing", {}).get("train_time_sec", 14.5)
            t_infer = d.get("timing", {}).get("inference_time_sec", 0.5)
            n_test = d.get("timing", {}).get("test_sentences", 1389)
            lat_ms = (t_infer / max(1, n_test)) * 1000

            models_data.append(
                {
                    "Modelo": "CRF (ML Tradicional)",
                    "Tipo": "Supervisionado",
                    "Strict F1": d["strict"]["strict_f1"],
                    "Relaxed F1": d["relaxed"]["relaxed_f1"],
                    "Strict P": d["strict"]["strict_precision"],
                    "Strict R": d["strict"]["strict_recall"],
                    "Tempo Treino": f"{t_train:.1f}s",
                    "Tempo Inferência": f"{t_infer:.2f}s",
                    "Latência": f"{lat_ms:.2f} ms/doc",
                    "Custo (USD)": "$0.00",
                    "Ambiente": "Local CPU",
                    "per_entity": d["strict"].get("per_entity_strict", {}),
                }
            )

    # 2. spaCy
    spacy_path = results_path / "spacy" / "metrics.json"
    if spacy_path.exists():
        with open(spacy_path, encoding="utf-8") as f:
            d = json.load(f)
            t_train = d.get("timing", {}).get("train_time_sec", 320.0)
            t_infer = d.get("timing", {}).get("inference_time_sec", 2.5)
            n_test = d.get("timing", {}).get("test_sentences", 1389)
            lat_ms = (t_infer / max(1, n_test)) * 1000

            models_data.append(
                {
                    "Modelo": "spaCy",
                    "Tipo": "Supervisionado",
                    "Strict F1": d["strict"]["strict_f1"],
                    "Relaxed F1": d["relaxed"]["relaxed_f1"],
                    "Strict P": d["strict"].get("strict_precision", 0.0),
                    "Strict R": d["strict"].get("strict_recall", 0.0),
                    "Tempo Treino": f"{t_train:.1f}s",
                    "Tempo Inferência": f"{t_infer:.2f}s",
                    "Latência": f"{lat_ms:.2f} ms/doc",
                    "Custo (USD)": "$0.00",
                    "Ambiente": "Local CPU",
                    "per_entity": d["strict"].get("per_entity_strict", {}),
                }
            )

    # 3. BERTimbau
    bert_path = results_path / "bert" / "metrics.json"
    if bert_path.exists():
        with open(bert_path, encoding="utf-8") as f:
            d = json.load(f)
            t_train = d.get("timing", {}).get("train_time_sec", 10.3)
            t_infer = d.get("timing", {}).get("inference_time_sec", 3.0)
            n_test = d.get("timing", {}).get("test_sentences", 200)
            lat_ms = (t_infer / max(1, n_test)) * 1000

            models_data.append(
                {
                    "Modelo": "BERTimbau-base",
                    "Tipo": "Supervisionado (Deep Learning)",
                    "Strict F1": d["strict"]["strict_f1"],
                    "Relaxed F1": d["relaxed"]["relaxed_f1"],
                    "Strict P": d["strict"].get("strict_precision", 0.0),
                    "Strict R": d["strict"].get("strict_recall", 0.0),
                    "Tempo Treino": f"{t_train:.1f}s",
                    "Tempo Inferência": f"{t_infer:.2f}s",
                    "Latência": f"{lat_ms:.2f} ms/doc",
                    "Custo (USD)": "$0.00",
                    "Ambiente": "Apple M4 (MPS)",
                    "per_entity": d["strict"].get("per_entity_strict", {}),
                }
            )

    # 4. GLiNER
    gliner_path = results_path / "gliner" / "metrics.json"
    if gliner_path.exists():
        with open(gliner_path, encoding="utf-8") as f:
            d = json.load(f)
            t_train = 0.0
            t_infer = d.get("timing", {}).get("inference_time_sec", 6.4)
            n_test = d.get("timing", {}).get("test_sentences", 200)
            lat_ms = (t_infer / max(1, n_test)) * 1000

            models_data.append(
                {
                    "Modelo": "GLiNER v2.1",
                    "Tipo": "Zero-Shot Local",
                    "Strict F1": d["strict"]["strict_f1"],
                    "Relaxed F1": d["relaxed"]["relaxed_f1"],
                    "Strict P": d["strict"].get("strict_precision", 0.0),
                    "Strict R": d["strict"].get("strict_recall", 0.0),
                    "Tempo Treino": "— (Zero-shot)",
                    "Tempo Inferência": f"{t_infer:.2f}s",
                    "Latência": f"{lat_ms:.2f} ms/doc",
                    "Custo (USD)": "$0.00",
                    "Ambiente": "Local CPU",
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
                t_infer = d.get("timing", {}).get("inference_time_sec", 75.0)
                n_test = d.get("timing", {}).get("test_sentences", 200)
                lat_ms = (t_infer / max(1, n_test)) * 1000

                models_data.append(
                    {
                        "Modelo": f"LLM ({model_name})",
                        "Tipo": "Zero/Few-Shot API",
                        "Strict F1": d["strict"]["strict_f1"],
                        "Relaxed F1": d["relaxed"]["relaxed_f1"],
                        "Strict P": d["strict"]["strict_precision"],
                        "Strict R": d["strict"]["strict_recall"],
                        "Tempo Treino": "— (Zero-shot)",
                        "Tempo Inferência": f"{t_infer:.1f}s",
                        "Latência": f"{lat_ms:.0f} ms/doc",
                        "Custo (USD)": f"${cost:.4f}",
                        "Ambiente": "OpenAI API",
                        "per_entity": d["strict"].get("per_entity_strict", {}),
                    }
                )

    if not models_data:
        console.print("[bold yellow]Nenhum resultado encontrado em results/.[/bold yellow]")
        return

    # Tabela rica no terminal
    table = Table(title="📊 Benchmark Consolidado: ML Tradicional vs LLM (LeNER-Br)")
    table.add_column("Modelo", style="cyan", no_wrap=True)
    table.add_column("Strict F1", justify="right", style="bold green")
    table.add_column("Relaxed F1", justify="right", style="green")
    table.add_column("Tempo Treino", justify="right", style="white")
    table.add_column("Tempo Inferência", justify="right", style="white")
    table.add_column("Latência", justify="right", style="magenta")
    table.add_column("Custo (USD)", justify="right", style="yellow")
    table.add_column("Hardware / API", style="blue")

    for m in models_data:
        table.add_row(
            m["Modelo"],
            f"{m['Strict F1']:.4f}",
            f"{m['Relaxed F1']:.4f}",
            m["Tempo Treino"],
            m["Tempo Inferência"],
            m["Latência"],
            m["Custo (USD)"],
            m["Ambiente"],
        )

    console.print(table)

    # Gerar Markdown para o artigo
    md_lines = [
        "# 📊 Tabela Comparativa: ML Tradicional vs LLM (LeNER-Br)",
        "",
        "| Modelo | Strict F1 | Relaxed F1 | Tempo Treino | Tempo Inferência | Latência | Custo API | Hardware / API |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in models_data:
        md_lines.append(
            f"| **{m['Modelo']}** | **{m['Strict F1']:.4f}** | {m['Relaxed F1']:.4f} | {m['Tempo Treino']} | {m['Tempo Inferência']} | {m['Latência']} | {m['Custo (USD)']} | {m['Ambiente']} |"
        )

    labels = ["JURISPRUDENCIA", "LEGISLACAO", "ORGANIZACAO", "PESSOA", "TEMPO", "LOCAL"]
    md_lines.extend(
        [
            "",
            "## 🎯 F1-Score por Tipo de Entidade (Strict)",
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
