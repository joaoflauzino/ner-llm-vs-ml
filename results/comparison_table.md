# Tabela Comparativa: ML Tradicional vs LLM (LeNER-Br)

| Modelo | Paradigma | Strict F1 | Relaxed F1 | Strict Precision | Strict Recall | Custo API | Ambiente |
|---|---|---|---|---|---|---|---|
| **CRF (ML Tradicional)** | Supervisionado (N=all) | **0.8077** | 0.8788 | 0.8547 | 0.7656 | $0.00 | Local CPU (14s) |
| **GLiNER v2.1** | Zero-Shot Local | **0.3411** | 0.5220 | 0.2681 | 0.4688 | $0.00 | Local CPU (6s) |
| **BERTimbau-base** | Supervisionado (M4 MPS) | **0.4749** | 0.6529 | 0.7809 | 0.3411 | $0.00 | Apple M4 (MPS) |
| **LLM (gpt-6-luna_0shot)** | Zero/Few-Shot API | **0.7270** | 0.8507 | 0.6925 | 0.7651 | $0.0329 | OpenAI API |

## F1-Score por Tipo de Entidade (Strict)

| Modelo | JURISPRUDENCIA | LEGISLACAO | ORGANIZACAO | PESSOA | TEMPO | LOCAL |
|---| --- | --- | --- | --- | --- | --- |
| **CRF (ML Tradicional)** | 0.8164 | 0.7707 | 0.8194 | 0.8249 | 0.8705 | 0.6250 |
| **GLiNER v2.1** | 0.0825 | 0.1650 | 0.4771 | 0.3368 | 0.4615 | 0.4181 |
| **BERTimbau-base** | 0.0000 | 0.5271 | 0.5302 | 0.4825 | 0.5911 | 0.0000 |
| **LLM (gpt-6-luna_0shot)** | 0.6667 | 0.7907 | 0.6644 | 0.9505 | 0.6737 | 0.6923 |
