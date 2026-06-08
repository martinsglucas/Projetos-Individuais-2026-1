import pytest

pytest.importorskip("fastapi")

from services.api import main


class FakeRepository:
    def list_companies(self) -> list[dict]:
        return [{"code": "MRV", "name": "MRV Engenharia", "ri_base_url": "https://example.com"}]

    def list_documents(self, *, company_code=None, year=None, quarter=None) -> list[dict]:
        return [{"company_code": company_code or "MRV", "year": year or 2025, "quarter": quarter or 1}]

    def list_metrics(self, *, company_code=None, year=None, quarter=None, category=None) -> list[dict]:
        return [{"company_code": company_code or "MRV", "year": year or 2025, "quarter": quarter or 1, "category": category or "sales"}]

    def list_conjuntura_metric_rows(self, *, year: int, quarter: int, company_code=None) -> list[dict]:
        return [
            {"company_code": "MRV", "category": "launches", "metric_name": "VGV", "year": year, "quarter": quarter, "value": 120},
            {"company_code": "MRV", "category": "launches", "metric_name": "VGV", "year": year, "quarter": quarter - 1, "value": 100},
            {"company_code": "MRV", "category": "launches", "metric_name": "VGV", "year": year - 1, "quarter": quarter, "value": 80},
        ]


def test_health() -> None:
    assert main.health() == {"status": "ok"}


def test_conjuntura_endpoint_uses_repository(monkeypatch) -> None:
    monkeypatch.setattr(main, "UdaRepository", lambda: FakeRepository())

    response = main.get_conjuntura(empresa="MRV", ano=2025, trimestre=3)

    assert response.status_code == 200
    payload = response.body.decode("utf-8")
    assert '"empresa":"MRV"' in payload
    assert '"x_trimestre_anterior_pct":20.0' in payload
    assert '"x_mesmo_trimestre_ano_anterior_pct":50.0' in payload
