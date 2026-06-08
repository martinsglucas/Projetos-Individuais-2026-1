from __future__ import annotations

from decimal import Decimal
from typing import Any

from fastapi import FastAPI, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from db import UdaRepository
from services.api.conjuntura import build_conjuntura_response

app = FastAPI(
    title="UDA Conjuntura API",
    version="0.1.0",
    description=(
        "API REST/JSON para consultar documentos, metricas extraidas de PDFs de RI "
        "e a visao de conjuntura habitacional por empresa e periodo."
    ),
    openapi_tags=[
        {"name": "health", "description": "Verificacao simples de disponibilidade da API."},
        {"name": "catalog", "description": "Consultas ao catalogo de empresas e documentos processados."},
        {"name": "metrics", "description": "Metricas operacionais extraidas com linhagem."},
        {"name": "conjuntura", "description": "Visao agregada no formato do boletim de conjuntura."},
    ],
)


def json_response(payload: Any) -> JSONResponse:
    return JSONResponse(
        content=jsonable_encoder(
            payload,
            custom_encoder={Decimal: lambda value: float(value)},
        )
    )


@app.get("/health", tags=["health"], summary="Verifica se a API esta disponivel")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/companies", tags=["catalog"], summary="Lista empresas cadastradas")
def list_companies() -> JSONResponse:
    repo = UdaRepository()
    return json_response({"companies": repo.list_companies()})


@app.get("/api/documents", tags=["catalog"], summary="Lista documentos processados")
def list_documents(
    empresa: str | None = Query(default=None, description="Codigo da empresa, por exemplo MRV ou CURY."),
    ano: int | None = Query(default=None, ge=2000, le=2100, description="Ano do periodo do documento."),
    trimestre: int | None = Query(default=None, ge=1, le=4, description="Trimestre do periodo do documento."),
) -> JSONResponse:
    repo = UdaRepository()
    documents = repo.list_documents(company_code=empresa, year=ano, quarter=trimestre)
    return json_response({"documents": documents})


@app.get("/api/metrics", tags=["metrics"], summary="Lista metricas extraidas")
def list_metrics(
    empresa: str | None = Query(default=None, description="Codigo da empresa, por exemplo MRV ou CURY."),
    ano: int | None = Query(default=None, ge=2000, le=2100, description="Ano da metrica."),
    trimestre: int | None = Query(default=None, ge=1, le=4, description="Trimestre da metrica."),
    categoria: str | None = Query(default=None, description="Categoria da metrica, como launches, sales ou landbank."),
) -> JSONResponse:
    repo = UdaRepository()
    metrics = repo.list_metrics(company_code=empresa, year=ano, quarter=trimestre, category=categoria)
    return json_response({"metrics": metrics})


@app.get("/api/conjuntura", tags=["conjuntura"], summary="Retorna a visao de conjuntura por periodo")
def get_conjuntura(
    ano: int = Query(ge=2000, le=2100, description="Ano de referencia do boletim."),
    trimestre: int = Query(ge=1, le=4, description="Trimestre de referencia do boletim."),
    empresa: str | None = Query(default=None, description="Filtro opcional por empresa."),
) -> JSONResponse:
    repo = UdaRepository()
    rows = repo.list_conjuntura_metric_rows(year=ano, quarter=trimestre, company_code=empresa)
    response = build_conjuntura_response(rows, year=ano, quarter=trimestre)
    response["metadata"] = {
        "empresa": empresa,
        "observacao": "Percentuais calculados a partir dos valores absolutos extraidos dos PDFs processados.",
    }
    return json_response(response)
