from __future__ import annotations

import argparse
import json
from decimal import Decimal
from urllib.parse import urlencode
from urllib.request import urlopen


EXPECTED_BOLETIM_3T25 = {
    "lancamentos": {
        "MRV": {
            "x_trimestre_anterior_pct": Decimal("-32"),
            "x_mesmo_trimestre_ano_anterior_pct": Decimal("-19"),
            "acumulado_ano_anterior_pct": Decimal("96"),
            "acumulado_ano_atual_pct": Decimal("20"),
        },
        "CURY": {
            "x_trimestre_anterior_pct": Decimal("3"),
            "x_mesmo_trimestre_ano_anterior_pct": Decimal("32"),
            "acumulado_ano_anterior_pct": Decimal("33"),
            "acumulado_ano_atual_pct": Decimal("35"),
        },
    },
    "vendas": {
        "MRV": {
            "x_trimestre_anterior_pct": Decimal("-12"),
            "x_mesmo_trimestre_ano_anterior_pct": Decimal("-10"),
            "acumulado_ano_anterior_pct": Decimal("9"),
            "acumulado_ano_atual_pct": Decimal("-5"),
        },
        "CURY": {
            "x_trimestre_anterior_pct": Decimal("-15"),
            "x_mesmo_trimestre_ano_anterior_pct": Decimal("32"),
            "acumulado_ano_anterior_pct": Decimal("30"),
            "acumulado_ano_atual_pct": Decimal("27"),
        },
    },
}


def fetch_conjuntura(base_url: str, *, year: int, quarter: int) -> dict:
    query = urlencode({"ano": year, "trimestre": quarter})
    with urlopen(f"{base_url.rstrip('/')}/api/conjuntura?{query}", timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def as_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value)).quantize(Decimal("0.1"))


def compare(payload: dict, *, tolerance: Decimal) -> list[dict[str, object]]:
    rows = []
    for category, expected_companies in EXPECTED_BOLETIM_3T25.items():
        actual_companies = {
            company["empresa"].upper(): company
            for company in payload.get("metricas", {}).get(category, {}).get("empresas", [])
        }
        for company, expected_metrics in expected_companies.items():
            actual = actual_companies.get(company)
            for metric_name, expected in expected_metrics.items():
                actual_value = as_decimal(actual.get(metric_name)) if actual else None
                diff = None if actual_value is None else (actual_value - expected).copy_abs()
                matches = diff is not None and diff <= tolerance
                rows.append(
                    {
                        "categoria": category,
                        "empresa": company,
                        "metrica": metric_name,
                        "esperado_boletim": expected,
                        "api": actual_value,
                        "status": "ok" if matches else "diff",
                    }
                )
    return rows


def print_rows(rows: list[dict[str, object]]) -> None:
    for row in rows:
        print(
            " ".join(
                [
                    f"status={row['status']}",
                    f"categoria={row['categoria']}",
                    f"empresa={row['empresa']}",
                    f"metrica={row['metrica']}",
                    f"esperado_boletim={row['esperado_boletim']}",
                    f"api={row['api']}",
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare /api/conjuntura with boletim_conjuntura_3t25 reference values."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--quarter", type=int, default=3)
    parser.add_argument("--tolerance", type=Decimal, default=Decimal("0.6"), help="Accepted percentage-point tolerance.")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero status when differences are found.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = fetch_conjuntura(args.base_url, year=args.year, quarter=args.quarter)
    rows = compare(payload, tolerance=args.tolerance)
    print_rows(rows)
    diffs = [row for row in rows if row["status"] != "ok"]
    print(f"checked={len(rows)} diffs={len(diffs)}")
    if args.strict and diffs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
