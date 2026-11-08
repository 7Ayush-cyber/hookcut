"""Swappable LLM providers. One function: complete(prompt) -> str.

Pick with env vars (or the Streamlit sidebar):
  PROMO_LLM_PROVIDER = none | gemini | openai | anthropic | mock
  PROMO_LLM_MODEL    = e.g. gemini-2.5-flash, llama-3.3-70b-versatile, claude-sonnet-4-5
  PROMO_LLM_API_KEY  = key (or GEMINI_API_KEY / OPENAI_API_KEY / GROQ_API_KEY / ANTHROPIC_API_KEY)
  PROMO_LLM_BASE_URL = for "openai": any OpenAI-compatible endpoint
                       (Groq https://api.groq.com/openai/v1, OpenRouter, Ollama
                        http://localhost:11434/v1, vLLM, or an in-house gateway)

Only `requests` is needed, so adding a provider is ~15 lines.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import requests

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
    "mock": "mock",
}


@dataclass
class LLMConfig:
    provider: str = "none"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.2
    max_retries: int = 4
    min_interval: float = 6.5   # seconds between calls: stays under ~10 requests/min free tiers

    @classmethod
    def from_env(cls) -> "LLMConfig":
        p = os.getenv("PROMO_LLM_PROVIDER", "none").lower()
        key = os.getenv("PROMO_LLM_API_KEY") or {
            "gemini": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        }.get(p) or ""
        return cls(provider=p, model=os.getenv("PROMO_LLM_MODEL", DEFAULT_MODELS.get(p, "")),
                   api_key=key, base_url=os.getenv("PROMO_LLM_BASE_URL", ""))

    @property
    def enabled(self) -> bool:
        return self.provider not in ("", "none")


class LLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._last = 0.0
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def complete(self, prompt: str) -> str:
        for attempt in range(self.cfg.max_retries):
            wait = self.cfg.min_interval - (time.time() - self._last)
            if wait > 0 and self.cfg.provider != "mock":
                time.sleep(wait)
            self._last = time.time()
            try:
                out = getattr(self, "_" + self.cfg.provider)(prompt)
                self.calls += 1
                return out
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if code in (429, 500, 502, 503, 529) and attempt < self.cfg.max_retries - 1:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
        raise RuntimeError("LLM retries exhausted")

    # ---------------- providers ----------------
    def _gemini(self, prompt: str) -> str:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.cfg.model}:generateContent")
        r = requests.post(url, params={"key": self.cfg.api_key}, timeout=120, json={
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.cfg.temperature,
                                 "responseMimeType": "application/json"},
        })
        r.raise_for_status()
        j = r.json()
        u = j.get("usageMetadata", {})
        self.tokens_in += u.get("promptTokenCount", 0)
        self.tokens_out += u.get("candidatesTokenCount", 0)
        return "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"])

    def _openai(self, prompt: str) -> str:
        base = (self.cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        r = requests.post(f"{base}/chat/completions", timeout=120,
                          headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                          json={"model": self.cfg.model, "temperature": self.cfg.temperature,
                                "messages": [{"role": "user", "content": prompt}],
                                "response_format": {"type": "json_object"}})
        r.raise_for_status()
        j = r.json()
        u = j.get("usage", {})
        self.tokens_in += u.get("prompt_tokens", 0)
        self.tokens_out += u.get("completion_tokens", 0)
        return j["choices"][0]["message"]["content"]

    def _anthropic(self, prompt: str) -> str:
        base = (self.cfg.base_url or "https://api.anthropic.com").rstrip("/")
        r = requests.post(f"{base}/v1/messages", timeout=120,
                          headers={"x-api-key": self.cfg.api_key,
                                   "anthropic-version": "2023-06-01"},
                          json={"model": self.cfg.model, "max_tokens": 4000,
                                "temperature": self.cfg.temperature,
                                "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        j = r.json()
        u = j.get("usage", {})
        self.tokens_in += u.get("input_tokens", 0)
        self.tokens_out += u.get("output_tokens", 0)
        return "".join(b.get("text", "") for b in j["content"])

    def _mock(self, prompt: str) -> str:
        """Offline stand-in used by tests: returns valid JSON for every candidate id."""
        ids = re.findall(r'"id":\s*"([^"]+)"', prompt)
        return json.dumps({"candidates": [
            {"id": i, "hook_score": 6, "cliffhanger_score": 6, "standalone_score": 6,
             "hook": "mock hook", "cliffhanger": "mock cliff", "why": "mock",
             "keep": True, "start": None, "end": None} for i in ids]})


def parse_json(text: str) -> dict:
    """Models sometimes wrap JSON in prose or ``` fences; pull the first object out."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start:end + 1])
