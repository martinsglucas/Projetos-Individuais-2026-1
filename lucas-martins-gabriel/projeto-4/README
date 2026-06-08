# Desafio de Processamento de Dado Não Estruturado

O plano de conclusão do projeto está documentado em [PLANO.md](PLANO.md).

## Visão geral

Pipeline UDA para coletar PDFs de centrais de resultados de incorporadoras, extrair métricas operacionais com LLM e servir os dados em uma API de conjuntura habitacional.

Fluxo principal:

```text
Central de Resultados RI
-> polling
-> download do PDF
-> SHA-256/idempotencia
-> MinIO
-> Docling/markdown
-> chunking e filtro semantico
-> LLM com contrato Pydantic
-> Postgres com lineage
-> API REST
```

## Fontes configuradas

- MRV: https://ri.mrv.com.br/informacoes-financeiras/central-de-resultados/
- Cury: https://ri.cury.net/informacoes-aos-investidores/central-de-resultados/

## Atendimento aos requisitos

- Extração automatizada: `services.ingestion.poll_sources` consulta as fontes cadastradas, descobre PDFs via HTML ou API MZIQ e pode rodar uma vez, por empresa, ou em loop agendável.
- Idempotência: o pipeline calcula SHA-256 do PDF antes da extração e consulta o catálogo para evitar reprocessar documentos já extraídos, salvo quando `--force` é usado.
- Processamento semântico: PDFs são convertidos para markdown com Docling, segmentados por headings e filtrados por relevância antes do prompt, reduzindo o contexto enviado ao LLM.
- Contrato semântico: a resposta da LLM é validada por Pydantic em `contracts/uda.py`; valores ausentes devem ser `null` e métricas com valor precisam de unidade.
- Catálogo e linhagem: Postgres registra empresas, fontes, documentos, chunks, execuções e métricas. Cada métrica guarda hash, URL, storage path e evidência textual.
- API REST: FastAPI expõe consultas por empresa e período, incluindo `/api/conjuntura`.
- Dois layouts validados: MRV `3T25` e Cury `3T25` foram parseados, persistidos e consultados pela API usando fixtures reprodutíveis.

## Setup local

```bash
cd lucas-martins-gabriel/projeto-4/src
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d
psql postgresql://admin:admin@localhost:5432/uda -f db/001_initial_schema.sql
```

Variáveis esperadas no `.env`:

```text
DATABASE_URL=postgresql://admin:admin@localhost:5432/uda
GEMINI_API_KEY=...
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=uda-artifacts
ARTIFACT_STORAGE_BACKEND=minio
GEMINI_FALLBACK_MODELS=gemini-2.5-flash-lite
GEMINI_MAX_ATTEMPTS=2
```

Use `src/.env.example` como base para criar o `.env` local.

Para desenvolvimento sem MinIO, use:

```bash
export ARTIFACT_STORAGE_BACKEND=filesystem
```

## MinIO

O MinIO e usado como storage de artefatos do pipeline. O bucket padrao e `uda-artifacts`, criado automaticamente quando o storage e inicializado.

Console web:

```text
http://localhost:9001
```

Credenciais locais do `docker-compose.yml`:

```text
usuario: minioadmin
senha: minioadmin
```

Estrutura de objetos gravada pelo pipeline:

```text
raw/<EMPRESA>/<ANO>/<PERIODO>/<pdf_hash>.pdf
parsed/<EMPRESA>/<ANO>/<PERIODO>/<pdf_hash>.md
validated/<EMPRESA>/<ANO>/<PERIODO>/llm_response_<pdf_hash>.json
```

Esses caminhos tambem ficam registrados no Postgres em campos como `documents.storage_path`, `documents.parsed_storage_path` e `extraction_runs.raw_response_storage_path`.

## LLM real

O provider Gemini usa `google-genai`. Por padrao, o modelo primario e `gemini-2.5-flash`.

Para reduzir falhas temporarias de capacidade (`503 UNAVAILABLE`), o provider faz retry e pode tentar modelos de fallback:

```bash
export GEMINI_FALLBACK_MODELS=gemini-2.5-flash-lite
export GEMINI_MAX_ATTEMPTS=2
```

O contrato Pydantic continua sendo o filtro de qualidade. Aliases seguros de unidade retornados pela LLM, como `BRL_thousand`, sao normalizados para o enum aceito `thousand_BRL`; respostas fora do contrato continuam falhando e ficam registradas como execucao com status `failed`.

Smoke integrado com Postgres, MinIO e fixture offline:

```bash
PYTHONPATH=. python scripts/smoke_integrated_pipeline.py --apply-schema
```

Esse smoke valida Postgres, MinIO e pipeline com fixture sem sobrescrever os arquivos `llm_response_*.json` versionados.

## Pipeline local

Com LLM real:

```bash
PYTHONPATH=. python -m services.extractor.run_pipeline \
  --pdf data/raw/mrv_1t25.pdf \
  --company MRV \
  --period 1T25 \
  --source-url https://ri.mrv.com.br/informacoes-financeiras/central-de-resultados/
```

Com fixture offline para teste sem chamada externa:

```bash
PYTHONPATH=. python -m services.extractor.run_pipeline \
  --pdf data/raw/mrv_1t25.pdf \
  --company MRV \
  --period 1T25 \
  --fixture data/validated/mrv_1t25_fixture_metrics.json \
  --no-persist-raw
```

MRV 3T25, no formato usado para a conjuntura:

```bash
PYTHONPATH=. python -m services.extractor.run_pipeline \
  --pdf data/raw/mrv_3t25.pdf \
  --company MRV \
  --period 3T25 \
  --source-url "https://api.mziq.com/mzfilemanager/v2/d/4b56353d-d5d9-435f-bf63-dcbf0a6c25d5/2c084655-23f7-7c55-5ac7-f4b2ed930448?origin=2" \
  --fixture data/validated/mrv_3t25_fixture_metrics.json \
  --no-persist-raw \
  --force
```

Segundo layout, Cury 3T25:

```bash
PYTHONPATH=. python -m services.extractor.run_pipeline \
  --pdf data/raw/cury_3t25.pdf \
  --company CURY \
  --period 3T25 \
  --source-url https://ri.cury.net/informacoes-aos-investidores/central-de-resultados/ \
  --fixture data/validated/cury_3t25_fixture_metrics.json \
  --no-persist-raw
```

Para atualizar o banco depois de alterar a fixture, use `--force`:

```bash
PYTHONPATH=. python -m services.extractor.run_pipeline \
  --pdf data/raw/cury_3t25.pdf \
  --company CURY \
  --period 3T25 \
  --source-url https://ri.cury.net/informacoes-aos-investidores/central-de-resultados/ \
  --fixture data/validated/cury_3t25_fixture_metrics.json \
  --no-persist-raw \
  --force
```

## Polling

Executa uma varredura das fontes cadastradas no banco:

```bash
PYTHONPATH=. python -m services.ingestion.poll_sources
```

O polling suporta páginas MZIQ como MRV e Cury usando a API pública de file manager quando os PDFs não aparecem diretamente no HTML.

Filtrar por empresa:

```bash
PYTHONPATH=. python -m services.ingestion.poll_sources --company MRV
```

Rodar continuamente com intervalo diário:

```bash
PYTHONPATH=. python -m services.ingestion.poll_sources --loop --interval-hours 24
```

## API

```bash
PYTHONPATH=. uvicorn services.api.main:app --reload --port 8000
```

Endpoints principais:

```text
GET /health
GET /api/companies
GET /api/documents
GET /api/metrics
GET /api/conjuntura?ano=2025&trimestre=3
GET /api/conjuntura?empresa=MRV&ano=2025&trimestre=3
```

A API de conjuntura calcula percentuais a partir dos valores absolutos gravados no banco. Quando há métricas `VGV acumulado`, elas são usadas para os comparativos `9M` e aparecem em `accumulated_lineage`; quando falta histórico, o campo correspondente retorna `null` e `missing_history` lista os períodos ausentes.

Exemplos após carregar as fixtures:

```bash
curl "http://localhost:8000/api/metrics?empresa=MRV&ano=2025&trimestre=3"
curl "http://localhost:8000/api/metrics?empresa=CURY&ano=2025&trimestre=3"
curl "http://localhost:8000/api/conjuntura?ano=2025&trimestre=3"
```

Validação integrada já executada para `3T25`: Cury e MRV retornam em `lancamentos` e `vendas`, com totais agregados, linhagem trimestral em `lineage` e linhagem acumulada `9M` em `accumulated_lineage`.

## Testes

```bash
PYTHONPATH=. python -m pytest tests -q
```

Na última validação local, a suíte retornou `19 passed`.
