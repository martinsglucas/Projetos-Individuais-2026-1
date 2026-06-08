from services.extractor.llm import GeminiProvider, parse_llm_response


def test_parse_llm_response_accepts_json_fenced_response() -> None:
    response = parse_llm_response(
        """```json
        {
          "company": "MRV",
          "period": {"year": 2025, "quarter": 1},
          "report_type": "operational_preview",
          "metrics": [],
          "missing_relevant_fields": [],
          "extraction_notes": []
        }
        ```"""
    )

    assert response.company == "MRV"
    assert response.period.year == 2025


def test_parse_llm_response_normalizes_safe_unit_aliases() -> None:
    response = parse_llm_response(
        """
        {
          "company": "MRV",
          "period": {"year": 2025, "quarter": 1},
          "report_type": "operational_preview",
          "metrics": [
            {
              "company": "MRV",
              "period": {"year": 2025, "quarter": 1},
              "category": "sales",
              "metric_name": "Receita",
              "value": 1000,
              "unit": "BRL_thousand",
              "evidence": {"raw_text": "Receita 1000"}
            }
          ],
          "missing_relevant_fields": [],
          "extraction_notes": []
        }
        """
    )

    assert response.metrics[0].unit == "thousand_BRL"


def test_gemini_provider_falls_back_on_retryable_error(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        text = """
        {
          "company": "MRV",
          "period": {"year": 2025, "quarter": 1},
          "report_type": "operational_preview",
          "metrics": [],
          "missing_relevant_fields": [],
          "extraction_notes": []
        }
        """
        usage_metadata = None

    def fake_generate(self, *, model_name: str, prompt: str):
        del self, prompt
        calls.append(model_name)
        if model_name == "gemini-2.5-flash":
            raise RuntimeError("503 UNAVAILABLE high demand")
        return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr("services.extractor.llm.time.sleep", lambda _: None)
    monkeypatch.setattr(GeminiProvider, "_generate_content", fake_generate)

    provider = GeminiProvider(
        model_name="gemini-2.5-flash",
        fallback_model_names=["gemini-2.5-flash-lite"],
        max_attempts_per_model=1,
    )
    result = provider.extract("prompt")

    assert calls == ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    assert result.model_name == "gemini-2.5-flash-lite"
