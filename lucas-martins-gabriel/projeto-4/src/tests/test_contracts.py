import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from contracts import (
    CompanyCode,
    ExtractedMetric,
    LLMMetricExtractionResponse,
    MetricCategory,
    MetricUnit,
    Period,
    SourceEvidence,
)


SRC_ROOT = Path(__file__).resolve().parents[1]


def test_period_from_label() -> None:
    period = Period.from_label("3T25")

    assert period.year == 2025
    assert period.quarter == 3
    assert period.label == "3T25"


def test_metric_requires_unit_when_value_is_present() -> None:
    with pytest.raises(ValidationError):
        ExtractedMetric(
            company=CompanyCode.MRV,
            period=Period.from_label("1T25"),
            category=MetricCategory.SALES,
            metric_name="VGV",
            value=Decimal("2167"),
            unit=None,
            evidence=SourceEvidence(raw_text="VGV 2.167"),
        )


def test_missing_value_cannot_have_high_confidence() -> None:
    with pytest.raises(ValidationError):
        ExtractedMetric(
            company=CompanyCode.MRV,
            period=Period.from_label("1T25"),
            category=MetricCategory.SALES,
            metric_name="VGV",
            value=None,
            unit=MetricUnit.BRL_MILLION,
            confidence=Decimal("0.95"),
            evidence=SourceEvidence(raw_text="Valor ausente"),
        )


def test_mrv_3t25_fixture_matches_llm_contract() -> None:
    fixture_path = SRC_ROOT / "data" / "validated" / "mrv_3t25_fixture_metrics.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    response = LLMMetricExtractionResponse.model_validate(payload)

    assert response.company == CompanyCode.MRV
    assert response.period == Period.from_label("3T25")
    assert len(response.metrics) == 12
    assert any(metric.metric_name == "VGV acumulado" for metric in response.metrics)
