from decimal import Decimal

import pytest
from pydantic import ValidationError

from contracts import CompanyCode, ExtractedMetric, MetricCategory, MetricUnit, Period, SourceEvidence


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
