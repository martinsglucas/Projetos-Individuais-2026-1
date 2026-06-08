from decimal import Decimal

from scripts.validate_conjuntura_against_boletim import compare


def test_compare_marks_matching_boletim_values_as_ok() -> None:
    payload = {
        "metricas": {
            "lancamentos": {
                "empresas": [
                    {
                        "empresa": "MRV",
                        "x_trimestre_anterior_pct": -31.7,
                        "x_mesmo_trimestre_ano_anterior_pct": -19,
                        "acumulado_ano_anterior_pct": 96,
                        "acumulado_ano_atual_pct": 20,
                    }
                ]
            },
            "vendas": {"empresas": []},
        }
    }

    rows = compare(payload, tolerance=Decimal("0.6"))
    mrv_quarter = [
        row
        for row in rows
        if row["categoria"] == "lancamentos"
        and row["empresa"] == "MRV"
        and row["metrica"] == "x_trimestre_anterior_pct"
    ][0]

    assert mrv_quarter["status"] == "ok"


def test_compare_marks_missing_company_as_diff() -> None:
    rows = compare({"metricas": {}}, tolerance=Decimal("0.6"))

    assert all(row["status"] == "diff" for row in rows)
