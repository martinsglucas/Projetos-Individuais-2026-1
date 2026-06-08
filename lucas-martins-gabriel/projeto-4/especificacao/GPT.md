# Desafio UDA — Contexto Arquitetural e Recomendações Técnicas

## Objetivo do Projeto

Construir uma pipeline resiliente de Engenharia e Análise de Dados Não Estruturados (UDA - Unstructured Data Analysis) para coletar, processar, estruturar e servir dados extraídos de PDFs de prévias operacionais e relatórios financeiros das construtoras brasileiras.

O pipeline deve:

- Detectar automaticamente novos PDFs publicados nas páginas de RI;
- Extrair informações semânticas usando LLMs;
- Estruturar os dados em contratos semânticos rígidos;
- Persistir dados com rastreabilidade (lineage);
- Disponibilizar os dados via API REST.

---

# Requisitos Críticos da Especificação

## O sistema DEVE:

- Monitorar continuamente páginas de RI;
- Detectar novos PDFs automaticamente;
- Implementar idempotência via hash;
- Utilizar LLMs;
- NÃO depender de regex/coordenadas fixas;
- Implementar chunking semântico;
- Possuir catálogo de dados;
- Registrar data lineage;
- Expor API REST.

---

# Filosofia da Solução

O projeto NÃO deve ser tratado como:

```text
PDF -> Regex -> Tabela
```

O projeto deve ser tratado como:

```text
PDF -> Estrutura Semântica -> Contrato -> Banco
```

A IA deve interpretar significado/contexto e NÃO apenas extrair texto.

---

# Arquitetura Recomendada

## Fluxo Geral

```text
RI Website
 ↓
Polling Scheduler
 ↓
Detectar novos PDFs
 ↓
Hash SHA-256
 ↓
Verificar catálogo
 ↓
Armazenar PDF bruto
 ↓
Parsing semântico
 ↓
Chunking semântico
 ↓
LLM Extraction
 ↓
Validação via contrato semântico
 ↓
Persistência
 ↓
API REST
```

---

# Stack Tecnológica Recomendada

## Parsing de PDF

### MinerU
https://github.com/opendatalab/MinerU

### PyMuPDF

Motivos:
- parsing semântico;
- extração robusta;
- alinhado com a especificação;
- mais resiliente a mudanças de layout.

---

## LLM

### Gemini Flash

Motivos:
- menor custo operacional;
- menos dependência de infraestrutura local;
- ótimo contexto para documentos longos;
- desenvolvimento mais rápido;
- restrição de armazenamento local inviabiliza Ollama local.

A arquitetura deve permitir troca futura de provider.

Exemplo:

```python
class LLMProvider:
    pass

class GeminiProvider(LLMProvider):
    pass

class OllamaProvider(LLMProvider):
    pass
```

---

## Banco de Dados

### PostgreSQL

Responsabilidades:
- catálogo de documentos;
- hashes;
- métricas extraídas;
- contratos semânticos;
- lineage;
- auditoria.

---

## Object Storage

### MinIO

Responsabilidades:
- armazenar PDFs originais;
- armazenar markdown parseado;
- armazenar JSONs intermediários;
- permitir reprocessamento.

---

## API

### FastAPI

---

## Infraestrutura

### Docker Compose

Inicialmente NÃO usar:
- Kubernetes;
- Airflow;
- vector databases;
- pipelines multi-agent;
- OCR avançado;
- RAG complexo.

---

# Estratégia de Ingestão

## Modelo escolhido

### Polling/CronJobs

Motivos:
- simples;
- robusto;
- explicitamente aceito na especificação;
- fácil de demonstrar.

---

## Fluxo de ingestão

```text
Página RI
 ↓
Extrair links
 ↓
Detectar novos PDFs
 ↓
Calcular hash
 ↓
Verificar duplicidade
 ↓
Enfileirar processamento
```

---

# Idempotência

## Requisito obrigatório

Antes do processamento:

1. Calcular SHA-256 do PDF;
2. Verificar existência no catálogo.

Se já existir:
- ignorar.

Se for novo:
- processar.

---

# Estratégia de Parsing

## NÃO usar

- regex rígida;
- coordenadas fixas;
- XPath acoplado;
- regras hardcoded por layout.

---

## Estratégia correta

### Parsing semântico

Fluxo:

```text
PDF
 ↓
MinerU
 ↓
Markdown estruturado
 ↓
Headings
 ↓
Semantic chunks
```

---

# Chunking Semântico

## Estratégia escolhida

Separar documento por:

- headings;
- seções;
- contexto semântico;
- blocos operacionais.

---

## Exemplos reais do PDF da MRV

- DESTAQUES
- VENDAS LÍQUIDAS
- GERAÇÃO DE CAIXA
- DADOS OPERACIONAIS

Exemplo:

```markdown
## VENDAS LÍQUIDAS

Vendas líquidas MRV Incorporação...
```

---

# Estratégia de Extração

## NÃO permitir respostas livres do LLM

O modelo deve preencher contratos rígidos.

---

# Contrato Semântico

## Usar Pydantic

```python
from pydantic import BaseModel
from datetime import datetime

class Metric(BaseModel):
    company: str
    report_type: str
    period: str
    metric_name: str
    metric_value: float | None
    unit: str | None
    source_page: int
    source_pdf_url: str
    extracted_at: datetime
```

---

# Regras do Prompt

## O prompt deve:

- retornar JSON válido;
- respeitar schema;
- retornar NULL quando necessário;
- não inventar valores;
- focar apenas em métricas operacionais.

---

# Catálogo de Dados

## documents

```sql
id
company
report_type
period
pdf_url
pdf_hash
storage_path
processed_at
```

---

## chunks

```sql
id
document_id
heading
content
page
```

---

## metrics

```sql
id
document_id
metric_name
metric_value
unit
page
confidence
raw_text
```

---

# Data Lineage

Cada métrica deve registrar:

- PDF original;
- URL original;
- página;
- chunk de origem;
- timestamp;
- confiança da extração.

---

# API REST

## Endpoints sugeridos

```text
GET /documents
GET /metrics
GET /companies
GET /conjuntura
```

---

## Endpoint principal

```text
GET /api/conjuntura?empresa=MRV&ano=2025&trimestre=1
```

---

# Estratégia de Desenvolvimento

## Sprint 1

Objetivo:
- processar 1 PDF ponta a ponta.

Resultado esperado:

```bash
python process_pdf.py mrv_1t25.pdf
```

Gerando:

```json
{
  "company": "MRV",
  "period": "1T25",
  "metrics": [...]
}
```

---

## Sprint 2

Adicionar:
- hashing;
- persistência;
- lineage.

---

## Sprint 3

Adicionar:
- polling automático.

---

## Sprint 4

Adicionar:
- API REST.

---

# Estrutura de Pastas Recomendada

```text
project/
├── docker-compose.yml
├── .env
├── services/
│   ├── api/
│   ├── extractor/
│   ├── ingestion/
│   └── workers/
├── contracts/
├── scripts/
├── data/
│   ├── raw/
│   ├── parsed/
│   ├── normalized/
│   └── validated/
├── notebooks/
└── tests/
```

---

# Docker Compose Inicial

```yaml
version: "3.9"

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: admin
      POSTGRES_DB: document_ai
    ports:
      - "5432:5432"

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: admin123
    ports:
      - "9000:9000"
      - "9001:9001"
```

---

# Dependências Python

```bash
pip install \
  pymupdf \
  mineru \
  google-generativeai \
  pydantic \
  pandas \
  sqlalchemy \
  psycopg2-binary
```

---

# Objetivo Arquitetural Final

A solução deve parecer um:

- semantic-first pipeline;
- schema-driven system;
- lineage-aware UDA engine;
- resilient document intelligence workflow.

E NÃO apenas:
- um parser de PDF;
- um scraper;
- um OCR simples.