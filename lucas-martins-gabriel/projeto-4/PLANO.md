# Plano para completar o Projeto 4 - Pipeline UDA

## 1. Objetivo de entrega

Completar um pipeline UDA para PDFs de RI de incorporadoras que:

- monitora fontes de RI por polling;
- detecta PDFs novos sem duplicar processamento;
- calcula SHA-256 do PDF antes de acionar LLM;
- faz parsing e chunking semântico do documento;
- extrai métricas operacionais com LLM usando contrato Pydantic rígido;
- persiste catálogo, chunks, execuções, métricas e linhagem em Postgres;
- expõe uma API REST/JSON para consulta por empresa e período.

O foco da entrega deve ser robustez arquitetural e rastreabilidade, não interface gráfica.

## 1.1. Decisões assumidas após alinhamento

- Empresas de demonstração: MRV e Cury. Pacaembu fica como alternativa caso a Cury tenha portal/PDF mais difícil de coletar ou processar.
- Fontes de ingestão:
  - MRV: `https://ri.mrv.com.br/informacoes-financeiras/central-de-resultados/`;
  - Cury: `https://ri.cury.net/informacoes-aos-investidores/central-de-resultados/`.
- LLM: o pipeline deve usar uma LLM real na extração, com Gemini como provider inicial.
- SDK Gemini: usar `google-genai`, importado como `from google import genai`. A biblioteca antiga `google-generativeai`/`google.generativeai` está depreciada.
- Fixtures offline: serão mantidas apenas para testes e demonstração reprodutível sem chamada externa. Uma fixture é um JSON salvo no formato esperado pela resposta da LLM; ela não substitui a integração real.
- MinIO: será tratado como componente necessário da solução, não apenas opcional.
- Polling: implementar como comando agendável que carrega/coleta documentos das fontes cadastradas. Se houver tempo, adicionar modo `--loop` com intervalo configurável.
- Saída da API: deve retornar dados no formato necessário para montar o boletim `boletim_conjuntura_3t25.pdf`, mas apenas para as empresas processadas.

## 1.2. Status atual

Implementado nesta branch:

- plano de execução e README operacional;
- commits incrementais na branch `projeto-4`;
- contratos Pydantic e schema Postgres;
- seed de empresas e fontes MRV/Cury;
- repository para catálogo, chunks, métricas, fontes e consultas da API;
- parsing Docling, chunking e filtro semântico de chunks;
- provider Gemini migrado para `google-genai`;
- orquestrador local `services.extractor.run_pipeline`;
- storage de artefatos com MinIO como backend principal e filesystem como fallback;
- polling agendável em `services.ingestion.poll_sources`;
- API FastAPI com `/health`, `/api/companies`, `/api/documents`, `/api/metrics` e `/api/conjuntura`;
- cálculo dos comparativos do boletim a partir de valores absolutos;
- testes unitários de contratos, LLM, filtro de chunks e cálculo de conjuntura.

Validação atual:

- `pytest`: 14 testes passaram.
- O ambiente Codex atual não conseguiu acessar o socket Docker, então a validação integrada Postgres/MinIO precisa ser rodada no terminal local do usuário.

Próximos itens críticos:

- subir Postgres e MinIO para testar o fluxo ponta a ponta;
- processar MRV com fixture e depois com LLM real;
- coletar/processar Cury ou Pacaembu para validar segundo layout;
- completar evidências e README final de submissão.

## 2. Estado atual do projeto

### Já implementado

- Estrutura inicial em `src/` com camadas de banco, contratos e extração.
- `docker-compose.yml` com Postgres e MinIO.
- Schema SQL inicial em `src/db/001_initial_schema.sql`.
- Catálogo relacional com:
  - `companies`;
  - `documents`;
  - `extraction_runs`;
  - `document_chunks`;
  - `metrics`;
  - `ingestion_sources`.
- Contratos semânticos Pydantic em `src/contracts/uda.py`, incluindo:
  - metadados do documento;
  - período;
  - chunks;
  - métricas extraídas;
  - evidências;
  - execução da extração;
  - resposta estruturada do LLM.
- Repositório Postgres em `src/db/repositories.py` para documentos, chunks, runs e métricas.
- Cálculo de SHA-256 em `src/services/extractor/hash.py`.
- Parsing de PDF para markdown com Docling em `src/services/extractor/parser.py`.
- Chunking por headings markdown em `src/services/extractor/chunking.py`.
- Prompt semântico versionado em `src/services/extractor/prompts.py`.
- Integração de LLM com Gemini em `src/services/extractor/llm.py`.
- CLI de parsing e persistência de chunks em `src/services/extractor/process_pdf.py`.
- CLI de extração de métricas e persistência em `src/services/extractor/extract_metrics.py`.
- Orquestrador local com idempotência em `src/services/extractor/run_pipeline.py`.
- Filtro semântico de chunks em `src/services/extractor/chunk_filter.py`.
- Storage de artefatos em `src/services/storage/`.
- Polling agendável em `src/services/ingestion/poll_sources.py`.
- API REST em `src/services/api/main.py`.
- Cálculo de conjuntura em `src/services/api/conjuntura.py`.
- Testes automatizados em `src/tests/`.
- Fixture validada para MRV 1T25 em `src/data/validated/mrv_1t25_fixture_metrics.json`.
- PDFs e markdowns de MRV já armazenados em `src/data/`.

### Parcialmente implementado

- Idempotência: o orquestrador local e o polling consultam hash antes de reprocessar, mas ainda falta validar esse comportamento com Postgres real rodando.
- Linhagem: o schema e os contratos suportam URL, hash, chunk, evidência, storage e execução; ainda falta validar preenchimento em execução ponta a ponta com PDFs reais.
- MinIO: a camada foi implementada, mas ainda falta teste integrado com o container MinIO.
- API REST: implementada e coberta por teste unitário direto dos endpoints.

### Ainda não implementado

- Demonstração com pelo menos dois layouts diferentes de empresas/documentos.
- Execução ponta a ponta completa com Postgres, MinIO e LLM real.
- Teste de polling contra MRV/Cury em rede.
- Evidências finais para submissão.

## 2.1. Comandos de validação integrada pendentes

Rodar no terminal local, onde Docker está acessível:

```bash
cd lucas-martins-gabriel/projeto-4/src
docker compose up -d
psql postgresql://admin:admin@localhost:5432/uda -f db/001_initial_schema.sql
PYTHONPATH=. python -m services.extractor.run_pipeline \
  --pdf data/raw/mrv_1t25.pdf \
  --company MRV \
  --period 1T25 \
  --fixture data/validated/mrv_1t25_fixture_metrics.json
PYTHONPATH=. uvicorn services.api.main:app --port 8000
```

Depois consultar:

```bash
curl "http://localhost:8000/api/metrics?empresa=MRV&ano=2025&trimestre=1"
curl "http://localhost:8000/api/conjuntura?empresa=MRV&ano=2025&trimestre=1"
```

## 3. Plano de implementação

### Fase 1 - Consolidar o pipeline local ponta a ponta

Objetivo: transformar as CLIs atuais em um fluxo confiável para um PDF local.

Tarefas:

- Adicionar comando orquestrador `run_pipeline.py` ou função equivalente que execute:
  - cálculo do hash;
  - consulta ao catálogo;
  - parsing;
  - chunking;
  - extração LLM ou fixture;
  - persistência;
  - atualização de status.
- Implementar idempotência explícita:
  - se `pdf_hash` já existir com status `extracted` ou `validated`, não chamar o LLM;
  - permitir `--force` apenas para reprocessamento manual.
- Melhorar o vínculo de métricas com chunks:
  - usar `evidence.chunk_id` retornado pelo LLM quando houver;
  - resolver esse valor para o UUID real do chunk no banco;
  - manter `source_text` e `source_heading` sempre preenchidos.
- Registrar falhas em `documents.error_message` e `extraction_runs.error_message`.
- Remover ou substituir `src/test.py`, que hoje é apenas um teste manual do Gemini.

Critério de aceite:

- Rodar um PDF MRV local e persistir documento, chunks, run e métricas.
- Rodar o mesmo PDF novamente e confirmar que o LLM não é chamado.

### Fase 2 - Implementar ingestão por polling

Objetivo: cumprir o requisito de extração automatizada e contínua.

Tarefas:

- Criar `src/services/ingestion/`.
- Implementar modelo simples de fonte:
  - empresa;
  - URL da central de resultados;
  - seletor/estratégia de descoberta;
  - frequência recomendada de polling.
- Criar scraper conservador:
  - baixar HTML da página de RI;
  - encontrar links PDF;
  - normalizar URLs relativas;
  - filtrar candidatos de "prévia operacional" quando possível;
  - registrar candidatos no catálogo.
- Evitar sobrecarga:
  - polling diário por padrão;
  - timeout;
  - user-agent identificável;
  - sem loops agressivos.
- Baixar PDFs novos para `src/data/raw/` ou MinIO.
- Calcular SHA-256 após download.
- Se o hash já existir, marcar/ignorar como duplicado antes do LLM.

Critério de aceite:

- Executar um comando de polling para uma fonte configurada e processar apenas PDFs novos.
- Registrar `last_checked_at` em `ingestion_sources`.

Decisão pendente:

- Se Cury bloquear ou dificultar a coleta, usar Pacaembu como segunda empresa.

### Fase 3 - Camada de storage

Objetivo: alinhar o projeto ao desenho com catálogo e reprocessamento.

Tarefas:

- Criar `src/services/storage/`.
- Implementar interface mínima de storage:
  - salvar PDF bruto;
  - salvar markdown parseado;
  - salvar resposta JSON bruta do LLM;
  - retornar caminho usado no catálogo.
- Começar com filesystem local para simplicidade.
- Conectar MinIO como backend principal de artefatos.
- Manter filesystem local apenas como fallback para desenvolvimento.
- Atualizar `documents.storage_path`, `parsed_storage_path`, `validated_storage_path` e `extraction_runs.raw_response_storage_path`.

Critério de aceite:

- Todos os artefatos gerados pelo pipeline possuem caminho registrado no banco.
- PDFs brutos, markdowns parseados e respostas JSON da LLM são salvos em buckets/prefixos do MinIO.

### Fase 4 - API REST

Objetivo: disponibilizar dados estruturados para o boletim de conjuntura.

Tarefas:

- Adicionar FastAPI às dependências.
- Criar `src/services/api/main.py`.
- Criar queries no repository para consulta.
- Implementar endpoints:
  - `GET /health`;
  - `GET /api/companies`;
  - `GET /api/documents`;
  - `GET /api/metrics`;
  - `GET /api/conjuntura?empresa=MRV&ano=2025&trimestre=1`.
- Retornar JSON com:
  - empresa;
  - ano;
  - trimestre;
  - métricas;
  - origem do PDF;
  - hash;
  - evidências.
- Implementar uma visão de boletim para lançamentos e vendas:
  - variação do trimestre atual contra o trimestre anterior;
  - variação do trimestre atual contra o mesmo trimestre do ano anterior;
  - variação acumulada do ano até o trimestre contra o mesmo acumulado do ano anterior;
  - totais agregados por conjunto de empresas.
- Documentar como rodar com `uvicorn`.

Critério de aceite:

- Consultar métricas da MRV 1T25 via API depois de processar o PDF.
- Consultar `GET /api/conjuntura?ano=2025&trimestre=3` e receber estrutura suficiente para reproduzir as tabelas do boletim.
- A resposta pode conter apenas MRV e Cury se essas forem as empresas processadas.

### Fase 4.1 - Modelagem dos dados do boletim

Objetivo: transformar métricas absolutas extraídas dos PDFs em indicadores comparativos iguais aos do boletim de conjuntura.

O PDF de referência `boletim_conjuntura_3t25.pdf` apresenta dois blocos principais:

- `LANÇAMENTOS 3T25`;
- `VENDAS 3T25`.

Para cada empresa, o boletim compara:

- trimestre atual contra trimestre anterior, por exemplo `3T25 x 2T25`;
- trimestre atual contra mesmo trimestre do ano anterior, por exemplo `3T25 x 3T24`;
- acumulado do ano anterior contra o ano anterior a ele, por exemplo `9m 24/23`;
- acumulado do ano atual contra o ano anterior, por exemplo `9m 25/24`.

Implicação para o pipeline:

- O banco deve guardar valores absolutos por empresa, métrica, ano e trimestre.
- A API deve calcular percentuais derivados a partir dos valores absolutos persistidos.
- O contrato semântico deve priorizar valores absolutos de `launches` e `sales`, principalmente `VGV`, porque eles alimentam as tabelas comparativas.
- O filtro de chunks deve priorizar tabelas como a seção `DADOS OPERACIONAIS` do markdown da MRV, porque os destaques iniciais podem citar vendas/lançamentos sem conter a tabela completa.
- Para calcular o boletim de `3T25`, o banco precisa ter, no mínimo:
  - valores de `3T25`;
  - valores de `2T25`;
  - valores de `3T24`;
  - valores de `1T24`, `2T24`, `3T24`;
  - valores de `1T25`, `2T25`, `3T25`.

Formato recomendado de resposta para o endpoint:

```json
{
  "ano": 2025,
  "trimestre": 3,
  "periodo": "3T25",
  "metricas": {
    "lancamentos": {
      "empresas": [
        {
          "empresa": "MRV",
          "valor_atual": 0,
          "x_trimestre_anterior_pct": null,
          "x_mesmo_trimestre_ano_anterior_pct": null,
          "acumulado_ano_anterior_pct": null,
          "acumulado_ano_atual_pct": null,
          "lineage": []
        }
      ],
      "total": {
        "x_trimestre_anterior_pct": null,
        "x_mesmo_trimestre_ano_anterior_pct": null,
        "acumulado_ano_anterior_pct": null,
        "acumulado_ano_atual_pct": null
      }
    },
    "vendas": {
      "empresas": [],
      "total": {}
    }
  }
}
```

Valores `null` devem ser retornados quando faltar histórico suficiente para calcular algum comparativo.

### Fase 5 - Testes e validação

Objetivo: reduzir risco de regressão e demonstrar robustez.

Tarefas:

- Criar testes unitários para:
  - `Period.from_label`;
  - validação de SHA-256;
  - `parse_llm_response`;
  - contrato rejeitando campos extras;
  - contrato rejeitando valor sem unidade;
  - idempotência.
- Criar teste de repository com banco local ou fixture controlada.
- Criar teste de API com FastAPI `TestClient`.
- Criar teste de fixture LLM sem chamar Gemini.
- Validar pelo menos dois documentos/layouts.
- Testar cálculo de variação percentual e acumulados usados pelo boletim.

Critério de aceite:

- `pytest` executa sem chamadas externas obrigatórias.
- Existe pelo menos um caminho de demonstração sem depender de crédito/API do LLM, usando fixture.
- O endpoint de conjuntura retorna `null`, e não inventa percentual, quando algum período base não existe no banco.

### Fase 6 - Documentação final

Objetivo: tornar a submissão reproduzível.

Tarefas:

- Expandir `README` do projeto com:
  - visão da arquitetura;
  - requisitos;
  - configuração de `.env`;
  - subida do Postgres/MinIO;
  - aplicação do schema;
  - execução do pipeline local;
  - execução do polling;
  - execução da API;
  - exemplos de chamadas `curl`;
  - estratégia de idempotência;
  - estratégia de chunking;
  - explicação do contrato semântico;
  - limitações conhecidas.
- Incluir diagrama textual do fluxo.
- Explicar por que não usa regex/coordenadas fixas para extrair métricas.
- Documentar que percentuais de variação são ignorados e que valores absolutos do período são priorizados.

Critério de aceite:

- Uma pessoa consegue subir o projeto e consultar a API seguindo apenas o README.

## 4. Ordem recomendada de execução

1. Criar testes mínimos dos contratos e parsing de resposta LLM.
2. Ajustar dependências e imports para execução consistente.
3. Criar orquestrador local com idempotência.
4. Completar storage MinIO e lineage.
5. Implementar queries e API REST.
6. Implementar cálculo do boletim de conjuntura.
7. Implementar polling simples agendável.
8. Processar segundo documento/layout.
9. Atualizar README final.

Essa ordem prioriza uma entrega demonstrável rapidamente: primeiro o pipeline local e a API, depois automação contínua.

## 4.1. Estratégia de commits incrementais

O histórico do repositório usa commits curtos, majoritariamente em português, com mensagens como `feat: ...`, `docs: ...` e `test: ...`. Quando necessário, o corpo do commit explica o racional da mudança. Seguir esse padrão e não adicionar `Co-authored-by`.

Como a pasta `lucas-martins-gabriel/projeto-4/` ainda aparece como não rastreada no git, o primeiro commit deve ser cuidadosamente montado para não incluir `.env`, `.venv`, caches ou artefatos temporários. A `.gitignore` local já ignora venv e `__pycache__`, mas o staging deve ser revisado com `git status --short --ignored` antes de cada commit.

Sequência sugerida:

1. `docs: registra plano de execução do projeto 4`
   - `PLANO.md`;
   - atualização mínima do `README`.
2. `feat: ajusta extração semântica com google-genai e filtro de chunks`
   - migração do provider Gemini para `google-genai`;
   - módulo `chunk_filter.py`;
   - testes do filtro e parsing de resposta;
   - atualização de dependências.
3. `feat: cadastra fontes de ingestão MRV e Cury`
   - URLs de RI no schema;
   - seed de `ingestion_sources`.
4. `feat: implementa storage MinIO para artefatos`
   - camada de storage;
   - integração com pipeline.
5. `feat: implementa API de conjuntura`
   - FastAPI;
   - endpoints de métricas/documentos/conjuntura;
   - cálculo dos comparativos do boletim.
6. `feat: implementa polling das centrais de resultados`
   - descoberta de PDFs;
   - download;
   - idempotência antes do LLM.
7. `docs: consolida instruções de execução e entrega`
   - README completo;
   - exemplos de comandos;
   - limitações e critérios de aceite.

Antes de cada commit:

- rodar os testes disponíveis;
- revisar `git diff --staged`;
- confirmar que não há chave de API, `.env`, `.venv`, cache ou arquivo local indevido;
- usar mensagem autoral simples, sem `Co-authored-by`.

## 5. Riscos técnicos

- O pacote instalado em `requirements.txt` é `google-genai`, mas o código atual importava `google.generativeai`. O provider deve ser migrado para a API nova.
- `docling` pode exigir dependências pesadas e demorar no primeiro processamento.
- `source_page` pode ficar incompleto porque o markdown exportado nem sempre preserva página de forma estruturada.
- O chunking atual por heading funciona para o markdown da MRV, mas pode precisar de reforço para relatórios em slides com poucas headings.
- A API do Gemini não deve ser chamada em testes automatizados; usar fixtures offline.
- A entrega exige pelo menos dois layouts. Hoje há evidência forte apenas no fluxo MRV.
- O boletim calcula variações percentuais, então a extração de apenas um trimestre não basta para reproduzir a saída final.
- Se não houver documentos históricos suficientes, o endpoint deve retornar `null` nos comparativos ausentes e explicar a falta de base no campo de metadados.

## 6. Dúvidas abertas

Antes de implementar as partes que dependem de decisão externa, ainda é preciso confirmar:

- Se os documentos históricos necessários para calcular 9m 24/23 e 9m 25/24 serão coletados totalmente pelo polling ou também carregados manualmente como amostra inicial.

## 7. Definição de pronto

O projeto estará pronto quando:

- um PDF novo for descoberto ou informado;
- o hash for calculado antes da extração;
- duplicatas forem ignoradas sem custo de LLM;
- o PDF for parseado em markdown;
- chunks relevantes forem selecionados;
- o LLM retornar JSON validado por Pydantic;
- métricas forem gravadas com lineage;
- a API retornar dados por empresa, ano e trimestre;
- a API retornar uma visão de conjuntura com comparativos de lançamentos e vendas no formato do boletim;
- houver documentação de execução e testes mínimos.
