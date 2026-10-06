# NER Benchmark: ML Tradicional vs LLM

Projeto experimental para avaliar o custo-benefício de modelos locais de NER (CRF, spaCy, BERTimbau, GLiNER) comparados a LLMs em decisões jurídicas brasileiras ([LeNER-Br](https://huggingface.co/datasets/peluz/lener_br)).

## 🎯 Pergunta Central
> *Quantos exemplos rotulados um modelo pequeno precisa para empatar com uma LLM em NER, e quanto isso custa?*

## 📁 Estrutura do Projeto
```
ner-llm-vs-ml/
├── configs/
│   ├── config.yaml          # Configurações do experimento, modelos, seeds
│   └── prompts.yaml         # Prompts zero-shot e definições das 6 entidades
├── src/ner_llm_vs_ml/
│   ├── data/
│   │   └── lener.py         # Loader e parser CoNLL do LeNER-Br com cache
│   ├── eval/
│   │   ├── metrics.py       # F1 Estrito (seqeval), F1 Relaxado e bootstrap 95%
│   │   └── hallucination.py # Detector de alucinações e normalizações indevidas
│   ├── llm/
│   │   └── parser.py        # Parser JSON tolerante e alinhador de BIO spans
│   ├── models/              # Baselines: CRF, spaCy, BERTimbau, GLiNER
│   └── cascade/             # Roteador de baixa confiança para LLM
├── tests/                   # 17 testes unitários (pytest)
├── PROPOSAL_NER.md          # Proposta detalhada do estudo e desenho experimental
└── pyproject.toml           # Gerenciado por uv (Python 3.13)
```

## 🚀 Como Começar

1. Instalar dependências com `uv`:
   ```bash
   uv sync
   ```

2. Executar a suíte de testes:
   ```bash
   uv run pytest -v
   ```

3. Rodar linter:
   ```bash
   uv run ruff check src tests
   ```

## 📊 Status das Fases
- [x] **Fase 1: Setup & Fundações** — `uv init`, PyTorch com MPS (M4), dataset loader LeNER-Br com cache local, métricas estritas/relaxadas, detector de alucinações, parser JSON/BIO e 17 testes automatizados.
- [ ] **Fase 2: Baselines Locais** — Implementação e treino do CRF, spaCy e BERTimbau-base (MPS).
- [ ] **Fase 3: Pipeline LLM & Cache** — Integração com OpenAI API, cache em disco e trava de orçamento ($5).
- [ ] **Fase 4: Experimentos 1 & 2** — Tabela Custo × F1 e curva de aprendizagem por número de exemplos $N$.
- [ ] **Fase 5: Alucinação & Cascata** — Medição de fabricação de texto e economia da cascata.
- [ ] **Fase 6: Artigo** — Gráficos comparativos e rascunho do artigo.
