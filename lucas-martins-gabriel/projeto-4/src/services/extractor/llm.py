from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from pydantic import ValidationError

from contracts import LLMMetricExtractionResponse


class LLMExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResult:
    raw_text: str
    parsed: LLMMetricExtractionResponse
    input_tokens: int | None = None
    output_tokens: int | None = None


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


def parse_llm_response(text: str) -> LLMMetricExtractionResponse:
    try:
        payload = json.loads(extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise LLMExtractionError(f"Invalid JSON returned by LLM: {exc}") from exc

    try:
        return LLMMetricExtractionResponse.model_validate(payload)
    except ValidationError as exc:
        raise LLMExtractionError(f"LLM response does not match semantic contract: {exc}") from exc


class GeminiProvider:
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str | None = None) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise LLMExtractionError("GEMINI_API_KEY is not configured")

        from google import genai

        self.client = genai.Client(api_key=self.api_key)

    def extract(self, prompt: str) -> LLMResult:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        raw_text = response.text
        parsed = parse_llm_response(raw_text)

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        return LLMResult(raw_text=raw_text, parsed=parsed, input_tokens=input_tokens, output_tokens=output_tokens)
