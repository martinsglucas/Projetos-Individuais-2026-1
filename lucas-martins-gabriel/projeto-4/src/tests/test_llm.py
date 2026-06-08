from services.extractor.llm import parse_llm_response


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
