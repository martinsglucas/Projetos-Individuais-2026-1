# Relatório de Entrega — Projeto Individual 2: Sistema de ML com MLflow

> **Aluno(a):** Guilherme Westphall, Lucas Martins Gabriel, Leonardo Padre
> **Matrícula:** 211061805, 221022088, XXXXXXX
> **Data de entrega:** 15/04/2026

---

## 1. Resumo do Projeto

Este projeto implementa um sistema de classificação binária de sentimento (positivo / negativo) sobre resenhas de filmes do Stanford Large Movie Review Dataset (aclImdb). O modelo utilizado é o `distilbert/distilbert-base-uncased-finetuned-sst-2-english`, disponível no HuggingFace, aplicado diretamente sem fine-tuning adicional — o que é viável porque o modelo já foi treinado no SST-2, domínio próximo ao IMDb. O pipeline cobre ingestão, pré-processamento, inferência e coleta de métricas, com rastreamento completo via MLflow. Oito experimentos foram executados variando tamanho de amostra, comprimento máximo de tokens e semente aleatória. O melhor resultado obtido foi 91,4% de acurácia (F1 positivo de 0,9083) com 500 amostras e `max_length=512`. O deploy é suportado por uma stack Docker Compose com PostgreSQL como backend do MLflow.


## 2. Escolha do Problema, Dataset e Modelo

### 2.1 Problema

A análise de sentimento em textos é uma das tarefas mais consolidadas de NLP, com aplicações diretas em monitoramento de opinião, filtragem de conteúdo e pesquisa de mercado. A formulação binária -- classificar um texto como expressando sentimento positivo ou negativo -- reduz o problema a uma classificação supervisionada bem definida, com ground truth confiável quando o dado vem de avaliações escritas por usuários.

Resenhas de filmes são um domínio particularmente rico para essa tarefa: os textos costumam ser extensos, com argumentação subjetiva, uso de ironia e variação considerável de vocabulário. Isso torna o problema mais desafiador do que análise de tweets curtos, por exemplo, e exige um modelo com boa capacidade de compreensão contextual.

A escolha do aclImdb como dataset de avaliação é direta: é um benchmark público consolidado, balanceado entre classes e com volume suficiente para estimativas estatisticamente confiáveis. A proximidade de domínio com o SST-2, ambos cobrem sentimento em inglês, justifica a abordagem zero-shot com o modelo pré-treinado.

### 2.2 Dataset

| Item | Descrição |
||--|
| **Nome do dataset** | Stanford Large Movie Review Dataset (aclImdb) |
| **Fonte** | Disco local (aclImdb baixado manualmente) |
| **Tamanho** | 25.000 resenhas no split de teste, balanceado (12.500 positivas / 12.500 negativas) |
| **Tipo de dado** | Texto em inglês com rótulo binário (positivo / negativo) |
| **Link** | |

### 2.3 Modelo pré-treinado

| Item | Descrição |
||--|
| **Nome do modelo** | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| **Fonte** (ex: Hugging Face) | HuggingFace Model Hub |
| **Tipo** (ex: classificação, NLP) | Classificação de texto / NLP |
| **Fine-tuning realizado?** | Não — o modelo é usado diretamente como disponibilizado |
| **Link** | |


## 3. Pré-processamento

- Remoção de tags HTML `<br />` via expressão regular, presentes com frequência nas resenhas do aclImdb por conta do formato original dos arquivos.
- Normalização de whitespace: sequências de espaços, tabs e quebras de linha são colapsadas em um único espaço.
- Sem tokenização manual: o tokenizador do DistilBERT é invocado internamente pelo pipeline HuggingFace, mantendo consistência com o vocabulário original do modelo.
- Truncamento no final para textos com mais de 512 tokens, respeitando o limite arquitetural do DistilBERT. A alternativa de split-and-aggregate foi descartada por adicionar complexidade desnecessária ao pipeline.
- A versão do pré-processamento é registrada como parâmetro (`preprocess_version = "v1"`) no MLflow para rastreabilidade entre runs.



## 4. Estrutura do Pipeline

O pipeline segue um fluxo linear de quatro estágios, orquestrado por `src/pipeline.py`. A ingestão carrega o aclImdb do disco em um DataFrame `[text, label]`; o pré-processamento aplica a limpeza de HTML e normalização; o carregamento do modelo constrói o pipeline HuggingFace com DistilBERT; e a avaliação executa a inferência em batches e computa as métricas. Quando a flag `--track` está ativa, o módulo `src/tracking.py` envolve todo o fluxo em um MLflow run, logando parâmetros no início e métricas/artefatos ao final.

```
aclImdb (disco)
      │
      ▼
┌─────────────┐
│  1. Ingest  │  src/data/ingest.py
│  load_imdb  │  → DataFrame [text, label]
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  2. Preprocess   │  src/data/preprocess.py
│  strip HTML,     │  → DataFrame limpo
│  normalize WS    │
└──────┬───────────┘
       │
       ▼
┌───────────────────┐
│  3. Load Model    │  src/model/loader.py
│  DistilBERT       │  → HuggingFace pipeline
│  classifier       │
└──────┬────────────┘
       │
       ▼
┌───────────────────┐
│  4. Evaluate      │  src/model/evaluate.py
│  inference +      │  → métricas, preds, confs
│  compute metrics  │
└──────┬────────────┘
       │
       ▼
┌───────────────────┐
│  5. Tracking      │  src/tracking.py  (opcional, --track)
│  MLflow log       │  → params, metrics, artefatos
└───────────────────┘
```

### Estrutura do código

```
sentimental-analysis-on-movie-reviews/
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── tracking.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   └── preprocess.py
│   └── model/
│       ├── __init__.py
│       ├── loader.py
│       └── evaluate.py
├── data/
│   └── raw/
│       └── aclImdb/
├── mlruns/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── EXPERIMENTS.md
```



## 5. Uso do MLflow

### 5.1 Rastreamento de experimentos

O MLflow é utilizado para registrar cada execução do pipeline dentro do experimento `sentiment-imdb`. Os parâmetros são logados no início da run, antes da inferência começar, de modo que um crash intermediário ainda preserva o contexto da execução.

- **Parâmetros registrados:** `data_dir`, `split`, `sample_size`, `batch_size`, `max_length`, `random_seed`, `model_name`, `preprocess_version`
- **Métricas registradas:** `accuracy`, `precision`, `recall`, `f1`
- **Artefatos salvos:** `classification_report.txt` (precisão, recall e F1 por classe), `confusion_matrix.txt` (tabela 2×2 com labels), `predictions.csv` (texto original, label real, label predito e confiança do modelo)

### 5.2 Versionamento e registro

O modelo é serializado e registrado no MLflow Model Registry via `mlflow.transformers.log_model` com `registered_model_name="sentiment-imdb"`. A versão 1 foi registrada a partir do exp-04 (baseline de 1.000 amostras, acurácia 0,900). O backend de artefatos é file-based local, armazenado em `mlruns/` na raiz do projeto e versionado junto com o código no git.

### 5.3 Evidências



## 6. Deploy

O deploy é realizado via Docker Compose com três serviços em rede interna. O serviço `db` sobe um PostgreSQL 15 como backend de metadados do MLflow. O serviço `mlflow` executa o servidor MLflow na porta 5000, apontando para o PostgreSQL e montando o diretório `mlruns/` local como volume de artefatos. O serviço `pipeline` constrói a imagem a partir do `Dockerfile` (Python 3.10-slim) e executa o pipeline apontando para o MLflow via variável de ambiente `MLFLOW_TRACKING_URI=http://mlflow:5000`.

- **Método de deploy:** Docker Compose com imagem construída localmente a partir do `Dockerfile`
- **Como executar inferência:** subindo o stack com `docker compose up`, o container `pipeline` executa automaticamente `python -m src.pipeline --data-dir src/data/raw/aclImdb --sample-size 200 --track`; para execuções customizadas, o comando pode ser sobrescrito via `docker compose run pipeline python -m src.pipeline [args]`

```bash
# Executar o pipeline via Docker Compose
docker compose up --build
```

## 7. Guardrails e Restrições de Uso



## 8. Observabilidade

O MLflow UI permite comparar todas as execuções do experimento `sentiment-imdb` lado a lado, filtrando e ordenando por qualquer parâmetro ou métrica registrada.

- **Comparação de execuções:** oito runs foram registradas cobrindo três dimensões independentes de variação: tamanho de amostra (`sample_size` de 100 a 1.000), comprimento máximo de tokens (`max_length` de 128 a 512) e semente aleatória (`random_seed` 42, 43 e 44). Cada dimensão usa as outras como âncora, permitindo isolar o efeito de cada variável.
- **Análise de métricas:** a variação de `sample_size` mostra que a acurácia estabiliza próximo a 0,90–0,91 com amostras acima de 250, sem ganho relevante ao dobrar para 1.000. A variação de `max_length` evidencia degradação clara ao truncar para 128 tokens (acurácia 0,854 vs. 0,914 com 512), confirmando que resenhas longas carregam informação discriminativa relevante. A variação de seed (42, 43, 44) produziu acurácia idêntica (0,914 nos três casos), indicando que a amostragem não introduz variância detectável nessa escala.
- **Capacidade de inspeção:** cada run armazena três artefatos — o relatório de classificação por classe, a matriz de confusão e o CSV de predições individuais com confiança — permitindo auditar casos específicos de erro sem precisar re-executar o pipeline.


## 9. Limitações e Riscos

- Resenhas com mais de 512 tokens são truncadas no final da sequência. O veredito do crítico costuma aparecer na conclusão do texto, o que significa que parte da informação mais discriminativa pode ser descartada antes da inferência.
- O modelo opera exclusivamente em inglês, por ter sido treinado no SST-2. Resenhas em outros idiomas produzirão predições sem validade.
- O modelo é estático: não foi retreinado com dados do IMDb e pode apresentar queda de desempenho em subdomínios com ironia densa, jargão técnico de crítica cinematográfica ou construções linguísticas pouco representadas no SST-2.
- A avaliação foi feita apenas no split de teste do aclImdb. Não há validação em dados externos, distribuição de produção ou resenhas coletadas após o período de criação do dataset.
- Não existe guardrail de entrada ou saída implementado: entradas malformadas, textos vazios ou conteúdo fora do domínio são processados sem rejeição ou aviso.



## 10. Como executar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar o pipeline sem rastreamento
python -m src.pipeline --data-dir data/raw/aclImdb --sample-size 200

# 3. Executar o pipeline com rastreamento MLflow
python -m src.pipeline --data-dir data/raw/aclImdb --sample-size 200 --track

# 4. Executar o pipeline com rastreamento e registro do modelo
python -m src.pipeline --data-dir data/raw/aclImdb --sample-size 200 --track --register-model

# 5. Iniciar o MLflow UI
mlflow ui --backend-store-uri mlruns/

# 6. Executar via Docker Compose
docker compose up --build
```



## 11. Referências

1. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv:1910.01108*.
2. MLflow. (2024). MLflow: A platform for the machine learning lifecycle. https://mlflow.org


## 12. Checklist de entrega

- [x] Código-fonte completo
- [x] Pipeline funcional
- [x] Configuração do MLflow
- [ ] Evidências de execução (logs, prints ou UI)
- [x] Modelo registrado
- [ ] Script ou endpoint de inferência
- [ ] Guardrails
- [] Pull Request aberto
