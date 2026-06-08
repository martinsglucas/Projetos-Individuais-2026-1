from __future__ import annotations

from decimal import Decimal
from typing import Any

from fastapi import FastAPI, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from db import UdaRepository
from services.api.conjuntura import build_conjuntura_response

app = FastAPI(title="UDA Conjuntura API", version="0.1.0")


def json_response(payload: Any) -> JSONResponse:
    return JSONResponse(
        content=jsonable_encoder(
            payload,
            custom_encoder={Decimal: lambda value: float(value)},
        )
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/companies")
def list_companies() -> JSONResponse:
    repo = UdaRepository()
    return json_response({"companies": repo.list_companies()})


@app.get("/api/documents")
def list_documents(
    empresa: str | None = Query(default=None),
    ano: int | None = Query(default=None, ge=2000, le=2100),
    trimestre: int | None = Query(default=None, ge=1, le=4),
) -> JSONResponse:
    repo = UdaRepository()
    documents = repo.list_documents(company_code=empresa, year=ano, quarter=trimestre)
    return json_response({"documents": documents})


@app.get("/api/metrics")
def list_metrics(
    empresa: str | None = Query(default=None),
    ano: int | None = Query(default=None, ge=2000, le=2100),
    trimestre: int | None = Query(default=None, ge=1, le=4),
    categoria: str | None = Query(default=None),
) -> JSONResponse:
    repo = UdaRepository()
    metrics = repo.list_metrics(company_code=empresa, year=ano, quarter=trimestre, category=categoria)
    return json_response({"metrics": metrics})


@app.get("/api/conjuntura")
def get_conjuntura(
    ano: int = Query(ge=2000, le=2100),
    trimestre: int = Query(ge=1, le=4),
    empresa: str | None = Query(default=None),
) -> JSONResponse:
    repo = UdaRepository()
    rows = repo.list_conjuntura_metric_rows(year=ano, quarter=trimestre, company_code=empresa)
    response = build_conjuntura_response(rows, year=ano, quarter=trimestre)
    response["metadata"] = {
        "empresa": empresa,
        "observacao": "Percentuais calculados a partir dos valores absolutos extraidos dos PDFs processados.",
    }
    return json_response(response)
