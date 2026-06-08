from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any


BOLETIM_CATEGORIES = {
    "lancamentos": "launches",
    "vendas": "sales",
}


def previous_quarter(year: int, quarter: int) -> tuple[int, int]:
    if quarter == 1:
        return year - 1, 4
    return year, quarter - 1


def percent_change(current: Decimal | None, base: Decimal | None) -> Decimal | None:
    if current is None or base is None or base == 0:
        return None
    return ((current - base) / base * Decimal("100")).quantize(Decimal("0.1"))


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _metric_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        row["company_code"],
        row["category"],
        int(row["year"]),
        int(row["quarter"]),
    )


def _sum_period(
    values: dict[tuple[str, str, int, int], Decimal],
    *,
    company_code: str,
    category: str,
    year: int,
    quarters: range,
) -> Decimal | None:
    total = Decimal("0")
    for quarter in quarters:
        value = values.get((company_code, category, year, quarter))
        if value is None:
            return None
        total += value
    return total


def build_conjuntura_response(rows: list[dict[str, Any]], *, year: int, quarter: int) -> dict[str, Any]:
    values: dict[tuple[str, str, int, int], Decimal] = {}
    lineage: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if row.get("metric_name") != "VGV":
            continue
        value = _to_decimal(row.get("value"))
        if value is None:
            continue

        key = _metric_key(row)
        values[key] = values.get(key, Decimal("0")) + value
        lineage[key].append(
            {
                "document_hash": row.get("pdf_hash"),
                "source_url": row.get("source_url"),
                "source_page": row.get("source_page"),
                "source_text": row.get("source_text"),
            }
        )

    companies = sorted({key[0] for key in values})
    previous_year, previous_q = previous_quarter(year, quarter)
    period_label = f"{quarter}T{str(year)[-2:]}"
    ytd_quarters = range(1, quarter + 1)

    response: dict[str, Any] = {
        "ano": year,
        "trimestre": quarter,
        "periodo": period_label,
        "metricas": {},
    }

    for output_name, category in BOLETIM_CATEGORIES.items():
        company_rows: list[dict[str, Any]] = []
        for company_code in companies:
            current = values.get((company_code, category, year, quarter))
            previous = values.get((company_code, category, previous_year, previous_q))
            same_last_year = values.get((company_code, category, year - 1, quarter))
            ytd_current = _sum_period(
                values,
                company_code=company_code,
                category=category,
                year=year,
                quarters=ytd_quarters,
            )
            ytd_last_year = _sum_period(
                values,
                company_code=company_code,
                category=category,
                year=year - 1,
                quarters=ytd_quarters,
            )
            ytd_two_years_ago = _sum_period(
                values,
                company_code=company_code,
                category=category,
                year=year - 2,
                quarters=ytd_quarters,
            )

            if current is None and ytd_current is None:
                continue

            company_rows.append(
                {
                    "empresa": company_code,
                    "valor_atual": current,
                    "x_trimestre_anterior_pct": percent_change(current, previous),
                    "x_mesmo_trimestre_ano_anterior_pct": percent_change(current, same_last_year),
                    "acumulado_ano_anterior_pct": percent_change(ytd_last_year, ytd_two_years_ago),
                    "acumulado_ano_atual_pct": percent_change(ytd_current, ytd_last_year),
                    "lineage": lineage.get((company_code, category, year, quarter), []),
                }
            )

        response["metricas"][output_name] = {
            "empresas": company_rows,
            "total": _build_total(values, category=category, year=year, quarter=quarter),
        }

    return response


def _total_for(
    values: dict[tuple[str, str, int, int], Decimal],
    *,
    category: str,
    year: int,
    quarter: int,
) -> Decimal | None:
    total = Decimal("0")
    found = False
    for company_code, metric_category, metric_year, metric_quarter in values:
        if metric_category == category and metric_year == year and metric_quarter == quarter:
            total += values[(company_code, metric_category, metric_year, metric_quarter)]
            found = True
    return total if found else None


def _total_ytd(
    values: dict[tuple[str, str, int, int], Decimal],
    *,
    category: str,
    year: int,
    quarters: range,
) -> Decimal | None:
    total = Decimal("0")
    for quarter in quarters:
        value = _total_for(values, category=category, year=year, quarter=quarter)
        if value is None:
            return None
        total += value
    return total


def _build_total(
    values: dict[tuple[str, str, int, int], Decimal],
    *,
    category: str,
    year: int,
    quarter: int,
) -> dict[str, Decimal | None]:
    previous_year, previous_q = previous_quarter(year, quarter)
    ytd_quarters = range(1, quarter + 1)
    current = _total_for(values, category=category, year=year, quarter=quarter)
    previous = _total_for(values, category=category, year=previous_year, quarter=previous_q)
    same_last_year = _total_for(values, category=category, year=year - 1, quarter=quarter)
    ytd_current = _total_ytd(values, category=category, year=year, quarters=ytd_quarters)
    ytd_last_year = _total_ytd(values, category=category, year=year - 1, quarters=ytd_quarters)
    ytd_two_years_ago = _total_ytd(values, category=category, year=year - 2, quarters=ytd_quarters)

    return {
        "x_trimestre_anterior_pct": percent_change(current, previous),
        "x_mesmo_trimestre_ano_anterior_pct": percent_change(current, same_last_year),
        "acumulado_ano_anterior_pct": percent_change(ytd_last_year, ytd_two_years_ago),
        "acumulado_ano_atual_pct": percent_change(ytd_current, ytd_last_year),
    }
