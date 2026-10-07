"""LLM client for zero-shot and few-shot NER with disk caching and budget enforcement."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ner_llm_vs_ml.data.lener import extract_entities_from_labels
from ner_llm_vs_ml.llm.cache import BudgetTracker, ResponseCache
from ner_llm_vs_ml.llm.parser import align_entities_to_tokens, parse_llm_json

# Carrega variáveis do arquivo .env
load_dotenv()


class LLMNERClient:
    """Manages OpenAI API calls, prompt construction, few-shot formatting,

    caching, and token-level BIO output alignment.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        temperature: float = 1,
        prompts_file: Path | str | None = None,
        budget_tracker: BudgetTracker | None = None,
        cache: ResponseCache | None = None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "dummy_key"),
            base_url=os.environ.get("OPENAI_BASE_URL", None),
        )
        self.budget = budget_tracker or BudgetTracker()
        self.cache = cache or ResponseCache()

        if prompts_file is None:
            prompts_file = Path(__file__).parents[3] / "configs" / "prompts.yaml"
        with open(prompts_file, encoding="utf-8") as f:
            self.prompts_cfg = yaml.safe_load(f)

        self.system_prompt = self.prompts_cfg["system_prompt"]
        self.entity_definitions = self.prompts_cfg["entity_definitions"]
        self.zero_shot_template = self.prompts_cfg["zero_shot_template"]

    def build_prompt(
        self,
        sentence: str,
        few_shot_examples: list[dict[str, Any]] | None = None,
    ) -> str:
        """Construct the prompt with entity definitions, optional few-shot demonstrations,

        and target sentence.
        """
        prompt = self.zero_shot_template.format(
            sentence=sentence,
            desc_PESSOA=self.entity_definitions["PESSOA"],
            desc_ORGANIZACAO=self.entity_definitions["ORGANIZACAO"],
            desc_LOCAL=self.entity_definitions["LOCAL"],
            desc_TEMPO=self.entity_definitions["TEMPO"],
            desc_LEGISLACAO=self.entity_definitions["LEGISLACAO"],
            desc_JURISPRUDENCIA=self.entity_definitions["JURISPRUDENCIA"],
        )

        if few_shot_examples:
            demo_text = "\n\nExemplos de referência anotados:\n"
            for idx, ex in enumerate(few_shot_examples, 1):
                demo_text += f'\nExemplo {idx}:\nTexto: "{ex["text"]}"\n'
                ents = extract_entities_from_labels(ex["tokens"], ex["ner_labels"])
                demo_text += f'{json.dumps({"entities": ents}, ensure_ascii=False)}\n'
            prompt = demo_text + "\n\nAgora analise o texto abaixo:\n" + prompt

        return prompt

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_api(self, prompt: str) -> tuple[str, int, int]:
        """Call OpenAI API with retry logic."""
        response = self.client.chat.completions.create(
            model=self.model_id,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        prompt_tokens = response.usage.prompt_tokens if response.usage else len(prompt) // 4
        completion_tokens = (
            response.usage.completion_tokens if response.usage else len(content) // 4
        )
        return content, prompt_tokens, completion_tokens

    def extract_entities_for_sentence(
        self,
        sentence: str,
        few_shot_examples: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Extract entities for a sentence string, checking cache and budget first."""
        prompt = self.build_prompt(sentence, few_shot_examples)

        # 1. Check cache
        cached = self.cache.get(self.model_id, prompt, self.temperature)
        if cached is not None:
            self.budget.cached_hits += 1
            raw_text = cached["response_text"]
            meta = {
                "cached": True,
                "prompt_tokens": cached.get("prompt_tokens", 0),
                "completion_tokens": cached.get("completion_tokens", 0),
            }
            entities = parse_llm_json(raw_text)
            return entities, meta

        # 2. Check budget before calling API
        self.budget.check_budget_available()

        # 3. Call API
        raw_text, p_tokens, c_tokens = self._call_api(prompt)
        self.budget.record_usage(p_tokens, c_tokens)

        # 4. Save to cache
        self.cache.set(
            self.model_id,
            prompt,
            self.temperature,
            raw_text,
            p_tokens,
            c_tokens,
        )

        entities = parse_llm_json(raw_text)
        meta = {
            "cached": False,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
        }
        return entities, meta

    def predict(
        self,
        sents: list[list[str]],
        few_shot_examples: list[dict[str, Any]] | None = None,
    ) -> tuple[list[list[str]], list[list[dict[str, str]]], dict[str, Any]]:
        """Run extraction across test sentences and return BIO labels and parsed entities."""
        start_time = time.time()
        bio_predictions = []
        raw_entities_list = []

        for tokens in sents:
            sentence_text = " ".join(tokens)
            entities, _ = self.extract_entities_for_sentence(sentence_text, few_shot_examples)
            bio = align_entities_to_tokens(tokens, entities)

            bio_predictions.append(bio)
            raw_entities_list.append(entities)

        elapsed = time.time() - start_time
        meta = {
            "elapsed_sec": elapsed,
            "budget_summary": self.budget.summary(),
        }
        return bio_predictions, raw_entities_list, meta
