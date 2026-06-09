from services.extractor.prompts import SYSTEM_INSTRUCTIONS


def test_prompt_prioritizes_consolidated_totals() -> None:
    assert "total consolidado" in SYSTEM_INSTRUCTIONS
    assert "TOTAL INCORPORACAO" in SYSTEM_INSTRUCTIONS
