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
    assert "x_trimestre_anterior" not in company["missing_history"]


def test_build_conjuntura_response_returns_null_for_incomplete_ytd() -> None:
    rows = [
        {"company_code": "MRV", "category": "sales", "metric_name": "VGV", "year": 2025, "quarter": 3, "value": 120},
        {"company_code": "MRV", "category": "sales", "metric_name": "VGV", "year": 2025, "quarter": 2, "value": 100},
        {"company_code": "MRV", "category": "sales", "metric_name": "VGV", "year": 2024, "quarter": 3, "value": 80},
    ]

    response = build_conjuntura_response(rows, year=2025, quarter=3)
    company = response["metricas"]["vendas"]["empresas"][0]

    assert company["acumulado_ano_atual_pct"] is None
    assert company["missing_history"]["acumulado_ano_atual"] == ["1T24", "1T25", "2T24"]


def test_build_conjuntura_response_explains_missing_quarter_comparisons() -> None:
    rows = [
        {"company_code": "CURY", "category": "sales", "metric_name": "VGV", "year": 2025, "quarter": 3, "value": 1827.0},
    ]

    response = build_conjuntura_response(rows, year=2025, quarter=3)
    company = response["metricas"]["vendas"]["empresas"][0]

    assert company["x_trimestre_anterior_pct"] is None
    assert company["x_mesmo_trimestre_ano_anterior_pct"] is None
    assert company["missing_history"]["x_trimestre_anterior"] == ["2T25"]
    assert company["missing_history"]["x_mesmo_trimestre_ano_anterior"] == ["3T24"]


def test_build_conjuntura_response_uses_ytd_metrics_when_available() -> None:
    rows = [
        {"company_code": "CURY", "category": "sales", "metric_name": "VGV", "year": 2025, "quarter": 3, "value": 1827.0},
        {"company_code": "CURY", "category": "sales", "metric_name": "VGV", "year": 2025, "quarter": 2, "value": 2261.4},
        {"company_code": "CURY", "category": "sales", "metric_name": "VGV", "year": 2024, "quarter": 3, "value": 1437.2},
        {
            "company_code": "CURY",
            "category": "sales",
            "metric_name": "VGV acumulado",
            "year": 2025,
            "quarter": 3,
            "value": 6194.0,
            "source_text": "Vendas Líquidas (R$ milhões VGV) | 9M25 | 6.194,0",
        },
        {
            "company_code": "CURY",
            "category": "sales",
            "metric_name": "VGV acumulado",
            "year": 2024,
            "quarter": 3,
            "value": 4738.5,
        },
    ]

    response = build_conjuntura_response(rows, year=2025, quarter=3)
    company = response["metricas"]["vendas"]["empresas"][0]

    assert company["x_trimestre_anterior_pct"] == Decimal("-19.2")
    assert company["x_mesmo_trimestre_ano_anterior_pct"] == Decimal("27.1")
    assert company["acumulado_ano_atual_pct"] == Decimal("30.7")
    assert "acumulado_ano_atual" not in company["missing_history"]
    assert len(company["lineage"]) == 1
    assert company["accumulated_lineage"][0]["source_text"] == "Vendas Líquidas (R$ milhões VGV) | 9M25 | 6.194,0"


def test_build_conjuntura_response_keeps_quarter_and_ytd_lineage_separate() -> None:
    rows = [
        {
            "company_code": "CURY",
            "category": "launches",
            "metric_name": "VGV",
            "year": 2025,
            "quarter": 3,
            "value": 1986.4,
            "source_text": "Lançamentos | VGV (R$ milhões) | 3T25 | 1.986,4",
        },
        {
            "company_code": "CURY",
            "category": "launches",
            "metric_name": "VGV acumulado",
            "year": 2025,
            "quarter": 3,
            "value": 6995.2,
            "source_text": "Lançamentos | VGV (R$ milhões) | 9M25 | 6.995,2",
        },
    ]

    response = build_conjuntura_response(rows, year=2025, quarter=3)
    company = response["metricas"]["lancamentos"]["empresas"][0]

    assert company["lineage"][0]["source_text"] == "Lançamentos | VGV (R$ milhões) | 3T25 | 1.986,4"
    assert company["accumulated_lineage"][0]["source_text"] == "Lançamentos | VGV (R$ milhões) | 9M25 | 6.995,2"
