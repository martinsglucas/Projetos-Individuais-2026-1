from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from contracts import LLMMetricExtractionResponse


class LLMExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResult:
    raw_text: str
    parsed: LLMMetricExtractionResponse
    model_name: str
    input_tokens: int | None = None
    output_tokens: int | None = None


UNIT_ALIASES = {
    "BRL_thousand": "thousand_BRL",
    "brl_thousand": "thousand_BRL",
    "BRL_millions": "BRL_million",
    "brl_million": "BRL_million",
    "BRL_bilion": "BRL_billion",
    "BRL_billion": "BRL_billion",
    "unit": "units",
}

RETRYABLE_STATUS_CODES = {500, 503, 504}


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or last < first:
        raise LLMExtractionError("LLM response does not contain a JSON object")
    return stripped[first : last + 1]


def normalize_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for metric in payload.get("metrics", []):
        if not isinstance(metric, dict):
            continue
        unit = metric.get("unit")
        if isinstance(unit, str):
            metric["unit"] = UNIT_ALIASES.get(unit, unit)
    return payload


def parse_llm_response(text: str) -> LLMMetricExtractionResponse:
    try:
        payload = json.loads(extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise LLMExtractionError(f"Invalid JSON returned by LLM: {exc}") from exc

    try:
        return LLMMetricExtractionResponse.model_validate(normalize_llm_payload(payload))
    except ValidationError as exc:
        raise LLMExtractionError(f"LLM response does not match semantic contract: {exc}") from exc


class GeminiProvider:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        fallback_model_names: list[str] | None = None,
        max_attempts_per_model: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.fallback_model_names = fallback_model_names if fallback_model_names is not None else self._fallback_models()
        self.max_attempts_per_model = max_attempts_per_model or int(os.getenv("GEMINI_MAX_ATTEMPTS", "2"))
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise LLMExtractionError("GEMINI_API_KEY is not configured")

        from google import genai

        self.client = genai.Client(api_key=self.api_key)

    def _fallback_models(self) -> list[str]:
        configured = os.getenv("GEMINI_FALLBACK_MODELS")
        if configured:
            return [model.strip() for model in configured.split(",") if model.strip()]
        if self.model_name != "gemini-2.5-flash-lite":
            return ["gemini-2.5-flash-lite"]
        return []

    def _candidate_models(self) -> list[str]:
        models = [self.model_name, *self.fallback_model_names]
        deduped: list[str] = []
        for model in models:
            if model not in deduped:
                deduped.append(model)
        return deduped

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code in RETRYABLE_STATUS_CODES:
            return True
        message = str(exc).lower()
        retryable_terms = ("503", "500", "504", "unavailable", "overloaded", "high demand")
        return any(token in message for token in retryable_terms)

    def _generate_content(self, *, model_name: str, prompt: str):
        from google.genai import types

        return self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )

    def extract(self, prompt: str) -> LLMResult:
        last_error: Exception | None = None
        response = None
        model_used = self.model_name

        for model_name in self._candidate_models():
            model_used = model_name
            for attempt in range(1, self.max_attempts_per_model + 1):
                try:
                    response = self._generate_content(model_name=model_name, prompt=prompt)
                    break
                except Exception as exc:
                    last_error = exc
                    if not self._is_retryable_error(exc) or attempt == self.max_attempts_per_model:
                        break
                    time.sleep(min(2 ** (attempt - 1), 8))
            if response is not None:
                break

        if response is None:
            raise LLMExtractionError(f"Gemini request failed after retries: {last_error}") from last_error

        raw_text = response.text
        parsed = parse_llm_response(raw_text)

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        return LLMResult(
            raw_text=raw_text,
            parsed=parsed,
            model_name=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
