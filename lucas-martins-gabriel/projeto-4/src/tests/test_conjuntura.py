from decimal import Decimal

from services.api.conjuntura import build_conjuntura_response, percent_change, previous_quarter


def test_previous_quarter_crosses_year_boundary() -> None:
    assert previous_quarter(2025, 1) == (2024, 4)


def test_percent_change_returns_null_without_base() -> None:
    assert percent_change(Decimal("100"), Decimal("0")) is None
    assert percent_change(None, Decimal("100")) is None


def test_build_conjuntura_response_calculates_company_variations() -> None:
    rows = [
        {"company_code": "MRV", "category": "launches", "metric_name": "VGV", "year": 2025, "quarter": 3, "value": 120},
        {"company_code": "MRV", "category": "launches", "metric_name": "VGV", "year": 2025, "quarter": 2, "value": 100},
        {"company_code": "MRV", "category": "launches", "metric_name": "VGV", "year": 2024, "quarter": 3, "value": 80},
        {"company_code": "MRV", "category": "launches", "metric_name": "VGV", "year": 2025, "quarter": 1, "value": 90},
        {"company_code": "MRV", "category": "launches", "metric_name": "VGV", "year": 2024, "quarter": 1, "value": 70},
        {"company_code": "MRV", "category": "launches", "metric_name": "VGV", "year": 2024, "quarter": 2, "value": 80},
    ]

    response = build_conjuntura_response(rows, year=2025, quarter=3)
    company = response["metricas"]["lancamentos"]["empresas"][0]

    assert company["x_trimestre_anterior_pct"] == Decimal("20.0")
    assert company["x_mesmo_trimestre_ano_anterior_pct"] == Decimal("50.0")
    assert company["acumulado_ano_atual_pct"] == Decimal("34.8")


def test_build_conjuntura_response_returns_null_for_incomplete_ytd() -> None:
    rows = [
        {"company_code": "MRV", "category": "sales", "metric_name": "VGV", "year": 2025, "quarter": 3, "value": 120},
        {"company_code": "MRV", "category": "sales", "metric_name": "VGV", "year": 2025, "quarter": 2, "value": 100},
        {"company_code": "MRV", "category": "sales", "metric_name": "VGV", "year": 2024, "quarter": 3, "value": 80},
    ]

    response = build_conjuntura_response(rows, year=2025, quarter=3)
    company = response["metricas"]["vendas"]["empresas"][0]

    assert company["acumulado_ano_atual_pct"] is None
