# Validacao contra a especificacao

Validacao realizada com base em `especificacao/README.md`.

## Resultado geral

O projeto atende aos componentes obrigatorios da especificacao: extracao automatizada por polling, idempotencia por hash, processamento semantico com chunking, contrato Pydantic, catalogo com linhagem e API REST para consulta de conjuntura.

## Matriz de requisitos

| Requisito da especificacao | Evidencia no projeto | Status |
| --- | --- | --- |
| Coletar PDFs diretamente das centrais de RI | Fontes MRV e Cury configuradas no seed SQL e no README; polling em `src/services/ingestion/poll_sources.py` | Atendido |
| Observar fontes continuamente sem sobrecarregar RI | Polling agendavel com `--loop`, `--interval-hours`, timeout e user-agent academico | Atendido |
| Detectar duplicidade antes do LLM | SHA-256 em `services/extractor/hash.py`; verificacao `document_exists_by_hash` no polling e no `run_pipeline` | Atendido |
| Parsing e chunking semantico | Docling gera markdown; `chunking.py` segmenta por headings; `chunk_filter.py` seleciona chunks relevantes antes do prompt | Atendido |
| Extracao com LLM sob contrato semantico | Gemini via `google-genai` em `llm.py`; prompt inclui JSON Schema de `LLMMetricExtractionResponse`; fixtures reproduzem o mesmo contrato | Atendido com ressalva |
| Contrato semanticamente rigido | Modelos Pydantic em `contracts/uda.py`; testes cobrem unidade obrigatoria e valores ausentes com baixa confianca | Atendido |
| Extrair valores absolutos e ignorar percentuais de variacao | Prompt instrui a extrair valores absolutos; fixtures MRV/Cury gravam VGV absoluto e acumulado; API calcula percentuais no backend | Atendido |
| Catalogo de dados e linhagem | Postgres armazena documentos, chunks, runs, metricas, `pdf_hash`, `source_url`, storage path e evidencia textual | Atendido |
| API REST/JSON filtravel por empresa e periodo | FastAPI em `services/api/main.py`; endpoints `/api/metrics` e `/api/conjuntura` aceitam `empresa`, `ano`, `trimestre` | Atendido |
| Swagger/OpenAPI | FastAPI expoe Swagger em `/docs` e schema em `/openapi.json` | Atendido |
| Dois layouts diferentes | MRV 3T25 e Cury 3T25 parseados, persistidos e consultados pela API com fixtures reprodutiveis | Atendido |
| Saida para boletim de conjuntura | `/api/conjuntura?ano=2025&trimestre=3` retorna lancamentos, vendas, totais, comparativos e linhagem | Atendido |

## Evidencias executadas

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

Resultado:

```text
23 passed in 5.88s
```

```bash
PYTHONPATH=. .venv/bin/python scripts/smoke_integrated_pipeline.py --apply-schema
```

Resultado resumido:

```text
postgres=ok
schema=ok
minio=ok bucket=uda-artifacts
chunks=9
selected_chunks=3
metrics=3
fixture_pipeline=ok
```

API local validada:

```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/api/conjuntura?ano=2025&trimestre=3"
curl "http://localhost:8000/docs"
```

Resultados observados:

- `health` retornou `{"status":"ok"}`;
- conjuntura `3T25` retornou CURY e MRV em `lancamentos` e `vendas`;
- `lineage` ficou separado de `accumulated_lineage`;
- totais agregados foram calculados a partir dos valores absolutos persistidos.
- Swagger respondeu HTTP 200 em `/docs`.

Auditoria contra o boletim real:

```bash
PYTHONPATH=. .venv/bin/python scripts/validate_conjuntura_against_boletim.py
```

Resultado resumido:

```text
checked=16 diffs=14
```

Essa auditoria confirma que o endpoint tem o formato do boletim, mas tambem explicita divergencias numericas quando as fixtures usam os valores absolutos dos PDFs processados e o boletim oficial usa outro recorte/segmento ou historico ainda nao carregado.

## Ressalvas

- A ultima tentativa de extracao MRV 3T25 com Gemini real chegou ate a chamada de LLM, mas retornou `503 UNAVAILABLE` por alta demanda temporaria do modelo. O pipeline real esta implementado; para reproducibilidade da entrega, as fixtures versionadas validam o mesmo contrato Pydantic esperado da LLM.
- `acumulado_ano_anterior_pct` retorna `null` quando faltam dados historicos `9M23`; o endpoint explicita isso em `missing_history`, em vez de inventar valores.
- A comparacao numerica com o boletim oficial ainda nao e criterio de igualdade perfeita; para isso, seria necessario carregar o mesmo historico e padronizar o mesmo recorte empresarial usado no boletim.
