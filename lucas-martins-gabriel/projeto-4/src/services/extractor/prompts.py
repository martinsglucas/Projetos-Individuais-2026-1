from __future__ import annotations

import json
from typing import Any

from contracts import LLMMetricExtractionResponse, MetricCategory, MetricUnit

PROMPT_VERSION = "operational_metrics_v1"

SYSTEM_INSTRUCTIONS = """\
Voce e um extrator de dados operacionais de relatorios de incorporadoras brasileiras.

Regras obrigatorias:
- Retorne somente JSON valido, sem markdown e sem texto explicativo.
- Extraia valores absolutos do periodo solicitado, nao percentuais de variacao de colunas Var.
- Use null quando o valor nao estiver presente no documento.
- Nao invente metricas, segmentos, paginas ou unidades.
- Preserve evidencias curtas no campo evidence.raw_text com o trecho que justifica o valor.
- Para valores entre parenteses, retorne numero negativo.
- Converta virgula decimal brasileira para ponto decimal no JSON.
- Mantenha pontos de milhares fora dos numeros. Exemplo: "2.167" em R$ milhoes significa 2167.
- Use units para unidades, BRL_million para R$ milhoes, BRL_billion para R$ bilhoes, percent para %, percentage_points para p.p.
- Quando a tabela trouxer total consolidado e segmentos abertos, extraia o total consolidado e nao repita os subsegmentos para a mesma metrica.
- Para MRV, prefira TOTAL INCORPORACAO quando existir; para Cury, prefira TOTAL ou a linha principal da tabela.
"""

TARGET_METRICS = """\
Priorize metricas operacionais de conjuntura:
- landbank: VGV, Unidades, Ticket Medio
- launches: VGV, Unidades, Ticket Medio
- sales: VGV, Unidades, Ticket Medio
- transfers: Repasses Unidades
- financing: Vendas com financiamento direto Unidades
- production: Producao Unidades
- cash_generation: Geracao de caixa, Geracao de caixa ajustada, Geracao de caixa sem cessao de carteira
- vso: VSO Liquida

Segmentos comuns: TOTAL INCORPORACAO, MRV, SENSIA, RESIA, LUGGO, URBA.
"""


def response_schema_json() -> str:
    schema = LLMMetricExtractionResponse.model_json_schema()
    return json.dumps(schema, ensure_ascii=False, indent=2)


def enum_help() -> str:
    categories = ", ".join(item.value for item in MetricCategory)
    units = ", ".join(item.value for item in MetricUnit)
    return f"Categorias permitidas: {categories}\nUnidades permitidas: {units}"


def format_chunks(chunks: list[dict[str, Any]]) -> str:
    formatted: list[str] = []
    for chunk in chunks:
        formatted.append(
            "\n".join(
                [
                    f"[chunk_id={chunk['id']} ordinal={chunk['ordinal']} heading={chunk.get('heading') or ''}]",
                    chunk["content"],
                ]
            )
        )
    return "\n\n---\n\n".join(formatted)


def build_metric_extraction_prompt(
    *,
    company: str,
    year: int,
    quarter: int,
    report_type: str,
    chunks: list[dict[str, Any]],
) -> str:
    period_label = f"{quarter}T{str(year)[-2:]}"
    return f"""\
{SYSTEM_INSTRUCTIONS}

Empresa: {company}
Periodo alvo: {period_label} ({year}, trimestre {quarter})
Tipo de relatorio: {report_type}

{TARGET_METRICS}

{enum_help()}

Formato obrigatorio de resposta JSON conforme este schema Pydantic/JSON Schema:
{response_schema_json()}

Chunks relevantes do documento:
{format_chunks(chunks)}
"""
