"""Thin LLM inference wrapper.

Designed for offline batch inference on a GPU server via HF transformers.
Every call is cached on disk keyed by (model, prompt) so that re-runs and
ablations are free.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
        )

    @staticmethod
    def key(model: str, system: str, user: str) -> str:
        return hashlib.sha256(f"{model}\x00{system}\x00{user}".encode()).hexdigest()

    def get(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM cache WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, value))
        self.conn.commit()


class LLMClient:
    """Batched chat completion with HF transformers.

    Usage:
        client = LLMClient("Qwen/Qwen3-4B-Instruct-2507")
        outs = client.chat_batch(system, [user1, user2, ...])

    Note: `tensor_parallel` and `max_model_len` are accepted for backward
    compatibility with existing configs/callers but are inert under the
    transformers backend. Multi-GPU sharding is handled by device_map="auto";
    context length is governed by the model/tokenizer config.
    """

    def __init__(
        self,
        model_name: str,
        cache_path: str | Path = "outputs/llm_cache.sqlite",
        tensor_parallel: int = 1,        # inert: kept for config compatibility
        max_model_len: int = 8192,        # inert: kept for config compatibility
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = 13,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.cache = PromptCache(cache_path)
        self._llm = None
        self._tokenizer = None
        self._tp = tensor_parallel
        self._max_model_len = max_model_len
        self.batch_size = batch_size

    def _ensure_engine(self):
        if self._llm is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            try:
                torch.manual_seed(self.seed)
            except Exception:
                pass

            logger.info("Loading HF model: %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Left padding is required for correct batched decoder-only generation.
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._llm = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self._llm.eval()

    def _generate(self, messages_batch: list[list[dict]]) -> list[str]:
        import torch

        prompts = [
            self._tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            for m in messages_batch
        ]
        inputs = self._tokenizer(
            prompts, return_tensors="pt", padding=True
        ).to(self._llm.device)

        with torch.no_grad():
            gen = self._llm.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        texts = []
        for row in gen:
            completion_ids = row[prompt_len:]
            texts.append(
                self._tokenizer.decode(completion_ids, skip_special_tokens=True)
            )
        return texts
    
    # def _render_chat(self, m):
    #     try:
    #         return self._tokenizer.apply_chat_template(
    #             m, tokenize=False, add_generation_prompt=True,
    #             enable_thinking=False,
    #         )
    #     except TypeError:
    #         # tokenizer's template doesn't support enable_thinking; fall back
    #         return self._tokenizer.apply_chat_template(
    #             m, tokenize=False, add_generation_prompt=True,
    #         )

    def chat_batch(self, system: str, users: list[str]) -> list[str]:
        keys = [self.cache.key(self.model_name, system, u) for u in users]
        outputs: list[str | None] = [self.cache.get(k) for k in keys]
        todo = [i for i, o in enumerate(outputs) if o is None]

        if todo:
            self._ensure_engine()
            # Process the uncached prompts in mini-batches to bound memory.
            for start in range(0, len(todo), self.batch_size):
                chunk = todo[start : start + self.batch_size]
                messages_batch = [
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": users[i]},
                    ]
                    for i in chunk
                ]
                texts = self._generate(messages_batch)
                for i, text in zip(chunk, texts):
                    outputs[i] = text
                    self.cache.put(keys[i], text)

        return outputs  # type: ignore[return-value]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str, default: dict | None = None) -> dict:
    """Extract the first JSON object from an LLM response, tolerating
    code fences and surrounding prose."""
    try:
        m = _JSON_RE.search(text)
        if m:
            return json.loads(m.group(0))
    except json.JSONDecodeError:
        pass
    logger.debug("JSON parse failure on: %.200s", text)
    return default if default is not None else {}