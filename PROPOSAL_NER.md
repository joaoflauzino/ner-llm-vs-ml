# Proposta: NER/Extração em português — modelo pequeno vs LLM

> Escopo escolhido: **ângulos A + B** (custo × acurácia e curva de quantidade de dados), com **C** (alucinação) e **D** (cascata) como extras. Dados em **português**. **MVP enxuto**: 1 dataset, 3 modelos pequenos, 2 LLMs.
> Esta proposta é independente de [PROPOSAL.md](file:///Users/joaoflauzino/Documents/projetos/llm-confidence/PROPOSAL.md) (calibração de confiança), que fica intacta.

## 1. Tese

> "Para extração de entidades **bem definidas e com algumas centenas de exemplos rotulados**, um modelo pequeno fine-tuned iguala ou supera LLMs em F1, a uma fração do custo e da latência. LLMs só compensam quando há pouquíssimos exemplos, e ainda **inventam ou alteram** trechos que não existem no texto."

**Pergunta central do artigo:** *quantos exemplos rotulados eu preciso para um modelo pequeno empatar com uma LLM, e quanto isso custa?*

> [!NOTE]
> Resultado honesto esperado: LLM provavelmente vence com 0–50 exemplos; o modelo pequeno cruza em algumas centenas. Se o cruzamento for em outro ponto, esse é o resultado. A proposta mede isso.

## 2. Dataset

**LeNER-Br** (`peluz/lener_br` no Hugging Face): NER em textos jurídicos brasileiros.

| Item | Valor (aprox., confirmar no dataset card) |
|---|---|
| Documentos | 70 (66 decisões judiciais + 4 legislações) |
| Entidades | 6: `PESSOA`, `ORGANIZACAO`, `LOCAL`, `TEMPO`, `LEGISLACAO`, `JURISPRUDENCIA` |
| Sentenças | ~7.800 treino / ~1.180 validação / ~1.390 teste |
| Formato | tokens + tags BIO |

Por que serve: tem um bom mix de entidades "fáceis" (`PESSOA`, `TEMPO`) e **difíceis/específicas do domínio** (`LEGISLACAO`, `JURISPRUDENCIA`), cobertura em português e licença aberta.

**Cuidados:**
- **Contaminação:** o dataset é público e pode ter entrado no treino das LLMs. Registrar como limitação (e, se der, testar em um pequeno conjunto próprio ou de texto novo).
- **Convenção de anotação:** LLM não conhece as regras de fronteira do dataset (ex.: incluir "Lei nº 8.666/93" inteiro). Por isso a avaliação usa métrica **estrita e relaxada** (seção 5).

## 3. Modelos comparados

### 3.1 Modelos pequenos / tradicionais
| # | Modelo | Papel |
|---|---|---|
| 1 | **CRF** (`sklearn-crfsuite`) com features manuais | ML realmente "tradicional"; treina em segundos na CPU |
| 2 | **spaCy** (pipeline treinado do zero, tok2vec) | Ferramenta de produção, leve |
| 3 | **BERTimbau-base** fine-tuned (`neuralmind/bert-base-portuguese-cased`) | Estado da arte "pequeno"; roda em Mac (MPS) ou GPU modesta |
| Opcional | **GLiNER** (zero-shot e fine-tuned) | Modelo leve zero-shot, boa referência intermediária |
| Opcional | Regras/regex só para `TEMPO` e `LEGISLACAO` | Baseline de custo ~zero |

### 3.2 LLMs
- **2 LLMs**: 1 frontier e 1 pequeno/barato (a definir conforme APIs disponíveis).
- Configurações por LLM: **zero-shot** (só definição das entidades) e **few-shot** (k = 5 e 20 exemplos), com saída JSON estruturada: `[{"text": "...", "label": "..."}]`.
- Mapeamento da saída para offsets por busca no texto (tratando entidades repetidas pela ordem de ocorrência).

## 4. Experimentos

### Exp. 1 — Custo × acurácia (ângulo A)
Todos os modelos no mesmo conjunto de teste. Métricas por modelo:
- F1 estrito e relaxado, no geral e **por entidade**.
- **Custo por 1.000 sentenças:** US$ para LLMs; tempo de CPU/GPU (e energia estimada, opcional) para modelos locais.
- **Latência** p50/p95 e **throughput**.
- Taxa de saída inválida (JSON quebrado, label inexistente).

### Exp. 2 — Curva de quantidade de dados (ângulo B), o gráfico central
- Modelos pequenos treinados com **N = 25, 50, 100, 250, 500, 1.000, 2.500, tudo** sentenças (3–5 seeds de amostragem).
- LLMs no eixo equivalente: **k exemplos no prompt** (0, 5, 20), mais uma linha horizontal por LLM.
- Saída: curva F1 × N com ponto de cruzamento e **IC 95%** (bootstrap entre seeds/sentenças).
- Resultado de decisão: "a partir de ~N exemplos, vale treinar".

### Exp. 3 — Alucinação na extração (extra C)
Para as saídas da LLM, medir:
- % de entidades cujo texto **não aparece literalmente** na sentença.
- % com **normalização indevida** (corrige grafia, expande sigla, reformata número de lei).
- Entidades **duplicadas ou fora de ordem**.
- Comparar com NER por span (CRF/BERT/spaCy), que **só marca texto existente** por construção.
- Verificar se alucinação aumenta em `LEGISLACAO`/`JURISPRUDENCIA`.

### Exp. 4 — Cascata (extra D)
- Confiança do modelo pequeno por sentença (ex.: mínimo/média das probabilidades por token de BERT ou marginais do CRF), **calibrada** com isotônica/Platt no conjunto de validação.
- Regra: se a confiança < τ, enviar a sentença à LLM.
- Curva: **custo × F1** variando τ, em comparação com "só modelo pequeno" e "só LLM".
- Mostra o ganho prático e retoma a ideia do projeto original (confiança só é útil se for calibrada).

## 5. Métricas e avaliação

- **Entity-level F1 estrito** (`seqeval`): fronteira e tipo exatos.
- **F1 relaxado:** sobreposição de span + tipo correto (evita punir a LLM só por convenção de fronteira).
- Reportar precisão e recall separados (LLMs tendem a ter precisão e recall diferentes dos modelos fine-tuned).
- Matriz de confusão entre tipos de entidade.
- **Estatística:** bootstrap sobre sentenças (IC 95%), bootstrap pareado para diferenças entre modelos, seeds fixas.
- **Custo:** preço da API na data do experimento (registrar tabela de preços e versão do modelo).

## 6. Hipóteses (escritas antes de rodar)

- **H1:** com ≥ ~N* exemplos (hipótese: 200–1.000), BERTimbau fine-tuned ≥ LLM few-shot em F1 estrito.
- **H2:** o custo por 1.000 sentenças do modelo local é ≥ 100× menor que o da LLM frontier.
- **H3:** a LLM tem F1 relaxado muito mais próximo do estrito dos modelos treinados do que o F1 estrito (a diferença é fronteira, não conceito).
- **H4:** LLMs apresentam taxa de alucinação/normalização não nula; modelos de span têm 0%.
- **H5:** a cascata atinge ≥ 95% do F1 da LLM com ≤ 30% do custo (hipótese a ser testada, o número é um palpite).
- **H6:** o CRF fica atrás do BERTimbau, mas ainda é competitivo em entidades regulares (`TEMPO`, `PESSOA`) a custo ≈ 0.

## 7. Estrutura do projeto (uv + Python 3.13)

```
llm-confidence/                  # (renomear opcional, ex.: ner-llm-vs-ml)
├── pyproject.toml               # uv init --python 3.13
├── uv.lock
├── README.md
├── .env.example
├── configs/                     # modelos, prompts, N das curvas, seeds
├── src/ner_bench/
│   ├── data/                    # loader LeNER-Br, subamostragem por N/seed
│   ├── models/                  # crf.py, spacy_model.py, bert.py, gliner.py
│   ├── llm/                     # clients, prompts, parsing JSON, cache em disco
│   ├── eval/                    # F1 estrito/relaxado, alucinação, bootstrap
│   ├── cost/                    # cálculo de custo/latência
│   ├── cascade/                 # roteamento + calibração
│   └── cli.py
├── results/                     # raw (jsonl/parquet), figuras, tabelas
├── notebooks/
├── tests/                       # métricas, parsing, mapeamento de offsets
└── paper/
```

**Dependências previstas:** `datasets`, `seqeval`, `scikit-learn`, `sklearn-crfsuite`, `spacy`, `transformers`, `torch`, `accelerate`, `gliner` (opcional), `litellm`, `pydantic`, `tenacity`, `pandas`, `matplotlib`, `typer`; dev: `pytest`, `ruff`, `jupyter`.

**Reprodutibilidade:** cache de todas as chamadas de LLM (hash de prompt+modelo+params), saídas brutas em JSONL, versões fixadas no `uv.lock`, seeds, estimativa de custo antes de rodar.

## 8. Plano de execução

| Fase | Entregável |
|---|---|
| 1. Setup | `uv init`, estrutura, loader do LeNER-Br, métricas com testes unitários |
| 2. Baselines locais | CRF → spaCy → BERTimbau com dataset completo; F1 de referência |
| 3. Pipeline LLM | prompts, parsing, mapeamento de offsets, cache; rodar zero/few-shot |
| 4. Exp. 1 e 2 | tabela custo×F1 e curva por N |
| 5. Exp. 3 | análise de alucinação |
| 6. Exp. 4 | cascata e curva custo×F1 |
| 7. Artigo | texto + figuras |

**Estimativa de custo de LLM (grosseira, a refinar):** ~1.400 sentenças de teste × ~1.500 tokens de prompt (com definições e few-shot) ≈ 2M tokens de entrada por configuração. Com um modelo barato, centavos; com frontier, algumas dezenas de dólares no total das configurações. Dá para rodar em subamostra de teste (ex.: 500) se necessário.

## 9. Esqueleto do artigo

1. **Hook:** "Você está pagando por token para extrair nomes de lei?"
2. Problema e dataset (LeNER-Br, entidades fáceis vs difíceis).
3. Metodologia (modelos, prompts, métricas estrita/relaxada).
4. **Resultado 1:** tabela custo × F1 × latência.
5. **Resultado 2:** curva F1 × nº de exemplos, com o ponto de cruzamento.
6. **Resultado 3:** alucinações da LLM em extração.
7. **Resultado 4:** cascata — o melhor dos dois mundos.
8. Quando LLM ainda é a melhor escolha (zero exemplo, domínio móvel, entidades semânticas).
9. Limitações e conclusão; repositório reproduzível.

## 10. Riscos e limitações

- **Contaminação do dataset** nas LLMs (público); mitigar com amostra nova se possível e declarar.
- **Anotação e fronteiras:** F1 estrito pune a LLM por convenção; por isso o F1 relaxado.
- **Domínio único (jurídico):** conclusões podem não generalizar. Expansão natural: HAREM (geral) ou documentos próprios.
- **Comparação não é 100% igual:** modelos pequenos usam dados rotulados; LLM usa prompt. O Exp. 2 trata isso explicitamente.
- **Preço e versão de LLM mudam:** registrar data e versão.
- **Hardware:** fine-tuning do BERTimbau-base em Mac (MPS) é viável; confirmar tempo de treino.

## 11. Decisões em aberto

1. **LLMs e orçamento:** quais APIs você tem (OpenAI, Anthropic, Google, OpenRouter, Ollama local)?
2. **Hardware:** qual Mac (chip/RAM) para o fine-tuning do BERTimbau?
3. **Nome do projeto/repositório:** manter `llm-confidence` ou renomear (ex.: `ner-llm-vs-ml`)?
4. **Se incluir** GLiNER e regex como baselines extras no MVP?

---

## 12. Decisões tomadas (atualização)

- **LLM:** OpenAI, orçamento de **US$ 5**. Modelo indicado: `gpt-6-luna` (**a confirmar** que o ID existe na conta e qual o preço; o ID fica em `configs/`, trocável sem mudar código). Com esse orçamento o MVP usa **1 LLM** (não 2).
- **Hardware:** Apple M4 → fine-tuning do BERTimbau-base via MPS é viável.
- **Nome:** renomear para `ner-llm-vs-ml` (pacote `ner_bench`).
- **GLiNER:** incluir como baseline opcional (ver explicação na conversa); regex só para `TEMPO`/`LEGISLACAO`.

### Controle de orçamento (US$ 5)
- Rodar primeiro em **subamostra de ~200 sentenças** para medir custo real por configuração, e só então escalar.
- Usar **Batch API** (desconto, se disponível) e **prompt caching** (prefixo fixo com definições + few-shot).
- Contador de tokens/custo no cache de chamadas, com **trava de gasto máximo** configurável.
- Ordem de prioridade das configurações: zero-shot → 5-shot → 20-shot → cascata (reaproveita saídas já cacheadas).
