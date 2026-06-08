from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any


BOLETIM_CATEGORIES = {
    "lancamentos": "launches",
    "vendas": "sales",
}
QUARTERLY_VGV_METRIC = "VGV"
YTD_VGV_METRIC = "VGV acumulado"


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


def _lineage_key(row: dict[str, Any]) -> tuple[str, str, int, int, str]:
    company_code, category, year, quarter = _metric_key(row)
    return (company_code, category, year, quarter, row["metric_name"])


def _is_ytd_metric(row: dict[str, Any]) -> bool:
    return row.get("metric_name") == YTD_VGV_METRIC


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


def _period_label(year: int, quarter: int) -> str:
    return f"{quarter}T{str(year)[-2:]}"


def _missing_periods_for_company(
    values: dict[tuple[str, str, int, int], Decimal],
    ytd_values: dict[tuple[str, str, int, int], Decimal],
    *,
    company_code: str,
    category: str,
    year: int,
    quarter: int,
) -> dict[str, list[str]]:
    previous_year, previous_q = previous_quarter(year, quarter)
    ytd_quarters = range(1, quarter + 1)

    missing = {
        "x_trimestre_anterior": [],
        "x_mesmo_trimestre_ano_anterior": [],
        "acumulado_ano_anterior": [],
        "acumulado_ano_atual": [],
    }

    if (company_code, category, previous_year, previous_q) not in values:
        missing["x_trimestre_anterior"].append(_period_label(previous_year, previous_q))
    if (company_code, category, year - 1, quarter) not in values:
        missing["x_mesmo_trimestre_ano_anterior"].append(_period_label(year - 1, quarter))

    if (company_code, category, year - 1, quarter) not in ytd_values:
        for ytd_quarter in ytd_quarters:
            if (company_code, category, year - 1, ytd_quarter) not in values:
                missing["acumulado_ano_anterior"].append(_period_label(year - 1, ytd_quarter))
                missing["acumulado_ano_atual"].append(_period_label(year - 1, ytd_quarter))
    if (company_code, category, year - 2, quarter) not in ytd_values:
        for ytd_quarter in ytd_quarters:
            if (company_code, category, year - 2, ytd_quarter) not in values:
                missing["acumulado_ano_anterior"].append(_period_label(year - 2, ytd_quarter))
    if (company_code, category, year, quarter) not in ytd_values:
        for ytd_quarter in ytd_quarters:
            if (company_code, category, year, ytd_quarter) not in values:
                missing["acumulado_ano_atual"].append(_period_label(year, ytd_quarter))

    return {key: sorted(set(periods)) for key, periods in missing.items() if periods}


def build_conjuntura_response(rows: list[dict[str, Any]], *, year: int, quarter: int) -> dict[str, Any]:
    values: dict[tuple[str, str, int, int], Decimal] = {}
    ytd_values: dict[tuple[str, str, int, int], Decimal] = {}
    lineage: dict[tuple[str, str, int, int, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if row.get("metric_name") not in {QUARTERLY_VGV_METRIC, YTD_VGV_METRIC}:
            continue
        value = _to_decimal(row.get("value"))
        if value is None:
            continue

        key = _metric_key(row)
        if _is_ytd_metric(row):
            ytd_values[key] = ytd_values.get(key, Decimal("0")) + value
        else:
            values[key] = values.get(key, Decimal("0")) + value
        lineage[_lineage_key(row)].append(
            {
                "document_hash": row.get("pdf_hash"),
                "source_url": row.get("source_url"),
                "source_page": row.get("source_page"),
                "source_text": row.get("source_text"),
            }
        )

    companies = sorted({key[0] for key in values} | {key[0] for key in ytd_values})
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
            ytd_current = ytd_values.get((company_code, category, year, quarter)) or _sum_period(
                values,
                company_code=company_code,
                category=category,
                year=year,
                quarters=ytd_quarters,
            )
            ytd_last_year = ytd_values.get((company_code, category, year - 1, quarter)) or _sum_period(
                values,
                company_code=company_code,
                category=category,
                year=year - 1,
                quarters=ytd_quarters,
            )
            ytd_two_years_ago = ytd_values.get((company_code, category, year - 2, quarter)) or _sum_period(
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
                    "missing_history": _missing_periods_for_company(
                        values,
                        ytd_values,
                        company_code=company_code,
                        category=category,
                        year=year,
                        quarter=quarter,
                    ),
                    "lineage": lineage.get((company_code, category, year, quarter, QUARTERLY_VGV_METRIC), []),
                    "accumulated_lineage": lineage.get((company_code, category, year, quarter, YTD_VGV_METRIC), []),
                }
            )

        response["metricas"][output_name] = {
            "empresas": company_rows,
            "total": _build_total(values, ytd_values, category=category, year=year, quarter=quarter),
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
    ytd_values: dict[tuple[str, str, int, int], Decimal],
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
    ytd_current = _total_for(ytd_values, category=category, year=year, quarter=quarter) or _total_ytd(
        values, category=category, year=year, quarters=ytd_quarters
    )
    ytd_last_year = _total_for(ytd_values, category=category, year=year - 1, quarter=quarter) or _total_ytd(
        values, category=category, year=year - 1, quarters=ytd_quarters
    )
    ytd_two_years_ago = _total_for(ytd_values, category=category, year=year - 2, quarter=quarter) or _total_ytd(
        values, category=category, year=year - 2, quarters=ytd_quarters
    )

    return {
        "x_trimestre_anterior_pct": percent_change(current, previous),
        "x_mesmo_trimestre_ano_anterior_pct": percent_change(current, same_last_year),
        "acumulado_ano_anterior_pct": percent_change(ytd_last_year, ytd_two_years_ago),
        "acumulado_ano_atual_pct": percent_change(ytd_current, ytd_last_year),
        "missing_history": _missing_periods_for_total(
            values,
            ytd_values,
            category=category,
            year=year,
            quarter=quarter,
        ),
    }


def _has_total_for(
    values: dict[tuple[str, str, int, int], Decimal],
    *,
    category: str,
    year: int,
    quarter: int,
) -> bool:
    return _total_for(values, category=category, year=year, quarter=quarter) is not None


def _missing_periods_for_total(
    values: dict[tuple[str, str, int, int], Decimal],
    ytd_values: dict[tuple[str, str, int, int], Decimal],
    *,
    category: str,
    year: int,
    quarter: int,
) -> dict[str, list[str]]:
    previous_year, previous_q = previous_quarter(year, quarter)
    ytd_quarters = range(1, quarter + 1)

    missing = {
        "x_trimestre_anterior": [],
        "x_mesmo_trimestre_ano_anterior": [],
        "acumulado_ano_anterior": [],
        "acumulado_ano_atual": [],
    }

    if not _has_total_for(values, category=category, year=previous_year, quarter=previous_q):
        missing["x_trimestre_anterior"].append(_period_label(previous_year, previous_q))
    if not _has_total_for(values, category=category, year=year - 1, quarter=quarter):
        missing["x_mesmo_trimestre_ano_anterior"].append(_period_label(year - 1, quarter))

    if not _has_total_for(ytd_values, category=category, year=year - 1, quarter=quarter):
        for ytd_quarter in ytd_quarters:
            if not _has_total_for(values, category=category, year=year - 1, quarter=ytd_quarter):
                missing["acumulado_ano_anterior"].append(_period_label(year - 1, ytd_quarter))
                missing["acumulado_ano_atual"].append(_period_label(year - 1, ytd_quarter))
    if not _has_total_for(ytd_values, category=category, year=year - 2, quarter=quarter):
        for ytd_quarter in ytd_quarters:
            if not _has_total_for(values, category=category, year=year - 2, quarter=ytd_quarter):
                missing["acumulado_ano_anterior"].append(_period_label(year - 2, ytd_quarter))
    if not _has_total_for(ytd_values, category=category, year=year, quarter=quarter):
        for ytd_quarter in ytd_quarters:
            if not _has_total_for(values, category=category, year=year, quarter=ytd_quarter):
                missing["acumulado_ano_atual"].append(_period_label(year, ytd_quarter))

    return {key: sorted(set(periods)) for key, periods in missing.items() if periods}
