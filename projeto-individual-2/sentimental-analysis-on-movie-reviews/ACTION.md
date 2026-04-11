# Action Plan — Sentimental Analysis on Movie Reviews

- **Theme:** binary sentiment analysis on movie reviews
- **Dataset:** IMDb Large Movie Review Dataset
- **Model:** distilbert-base-uncased-finetuned-sst-2-english

---

## Stage 0 — Local Environment Setup

**Goal:** Get every tool installed and verified before writing any project code. Half of all "ML systems" failures happen here.

**Install:**
- **Python 3.10 or 3.11** — avoid 3.12+ to dodge transformers/torch wheel issues on Windows
- **A virtual environment** — use `venv` or `conda`. Pick one and stick with it
- **Git** — already present since this is in a repo
- **Core Python libraries:** `transformers`, `torch` (CPU is fine), `datasets`, `mlflow`, `scikit-learn`, `pandas`, `fastapi`, `uvicorn`, `langdetect`, `pydantic`

**Local infrastructure:**
- **MLflow tracking server** — runs locally with `mlflow ui` on `localhost:5000`. No remote backend. The default file-based store (`mlruns/`) is enough
- **HuggingFace cache** — `~/.cache/huggingface/` will hold the model (~250MB) and IMDb dataset (~80MB). Make sure you have ~2GB free
- **Disk:** Reserve ~3GB total for caches, data, mlruns, and Python packages
- **No GPU required** — DistilBERT inference on CPU is fast enough for IMDb test set evaluation

**Validation gate before moving on:**
- `python -c "import torch, transformers, mlflow, datasets; print('ok')"` runs cleanly
- `mlflow ui` starts and you can open `localhost:5000` in your browser
- A 5-line script that loads the DistilBERT pipeline and classifies one sentence works
- A 3-line script that calls `load_dataset("imdb")` downloads successfully

**Risks to avoid early:**
- **Don't install `tensorflow`** — `transformers` will pick it up by default and bloat your environment. Stick to PyTorch
- **Don't pin overly modern versions** — pick stable mid-range versions (transformers 4.40, torch 2.2, mlflow 2.12) that have known-good Windows wheels
- **Don't use system Python** — always work inside a venv. You'll thank yourself when something breaks
- **Don't skip pinning** — write `requirements.txt` immediately with exact versions so the project is reproducible

---

## Stage 1 — Smoke Test the Whole Stack

**Goal:** Prove the full chain works end-to-end before structuring anything. One throwaway script: load IMDb → load DistilBERT → predict on 10 reviews → print accuracy.

**Why first:** This is your sanity check. If the model and dataset don't speak to each other (label encoding mismatch, tokenizer issues, memory limits), you find out in 20 minutes instead of after building the pipeline.

**Decisions to make here:**
- **Label mapping:** DistilBERT returns `POSITIVE`/`NEGATIVE`. IMDb uses `1`/`0`. Lock in this mapping and document it now
- **Truncation strategy:** IMDb reviews can exceed 512 tokens. Decide: truncate at 512 (simplest, recommended) or split-and-aggregate (overkill). Pick truncate
- **Sample size for development:** Use a small sample (200-500 reviews) during dev. Save full evaluation for final runs

**Outputs before moving on:**
- A working script that produces a real accuracy number on a small IMDb sample
- Confidence that the model+dataset pair behaves sensibly (expect high accuracy, ~88-93%)

**Pitfalls:**
- **Don't fight CUDA setup on Windows.** Force CPU explicitly with `device=-1` in the pipeline. CPU is fast enough
- **Don't skip this stage to save time** — building structure on top of an unverified stack is the most common waste

---

## Stage 2 — Modularize Into a Real Pipeline

**Goal:** Take the smoke test and break it into clean stages that match the rubric: ingest → preprocess → load model → evaluate. Each stage is a separate file with a single responsibility.

**Decisions to make:**
- **Where the pipeline orchestrator lives** — one entry point (`pipeline.py` or `run.py`) that calls each stage in order. CLI args for things you'll vary (sample size, experiment name)
- **Determinism:** Set a `random_state` everywhere sampling happens. Reproducibility is grading-critical
- **Preprocessing scope:** Movie reviews from IMDb contain HTML `<br />` tags. Decide what to clean and document why. Don't over-engineer — strip HTML, normalize whitespace, done

**Outputs before moving on:**
- A single command runs the entire pipeline from raw data to printed metrics
- Each stage is callable independently (you can import and run `load_imdb()` alone for debugging)
- The orchestrator works from a clean state (delete cached data, run again, still works)

**Pitfalls:**
- **Don't put everything in a notebook.** Notebooks are not pipelines and reviewers will downgrade you
- **Don't hardcode paths.** Use relative paths from project root or a small config
- **Don't preprocess into an undocumented format.** Whatever shape the data takes between stages, write down the contract

---

## Stage 3 — Add MLflow Tracking

**Goal:** Wrap the pipeline so every run logs params, metrics, and artifacts. This is the highest-leverage stage of the project — 25% of the grade comes from here, and adding MLflow well takes only an afternoon.

**Decisions to make:**
- **Experiment naming:** Pick one experiment name and stick to it (`sentiment-imdb`). All your runs go inside it for easy comparison
- **What to log:**
  - **Params:** dataset name, sample size, model name, max_length, preprocessing version, random seed
  - **Metrics:** accuracy, F1, precision, recall — at minimum
  - **Artifacts:** classification report, confusion matrix image, a CSV sample of predictions
- **Whether to register the model:** Yes. Use `mlflow.transformers.log_model(..., registered_model_name=...)`. The Model Registry is part of the versioning criterion

**Outputs before moving on:**
- Multiple runs visible in the MLflow UI under one experiment
- At least one run has params + metrics + artifacts + a registered model version
- You can open `mlflow ui`, click into a run, and see everything

**Pitfalls:**
- **Don't log only at the end.** Log params at the start of the run, metrics as soon as they're computed. Crashes mid-run still produce useful evidence
- **Don't forget to commit `mlruns/`.** Reviewers should be able to clone, run `mlflow ui`, and see your experiments without re-running anything
- **Don't register the model 50 times during dev.** Only register on intentional, named runs

---

## Stage 4 — Run Multiple Experiments

**Goal:** Generate evidence of comparison and observability by running the pipeline 3-5 times with varied parameters.

**Decisions to make:**
- **What to vary:** Sample size is the easiest meaningful variation. You could also vary preprocessing (with/without HTML strip), max_length, or batch_size. Pick 2 dimensions
- **What to compare:** Accuracy stability across sample sizes shows model robustness. This becomes a finding in your report

**Outputs before moving on:**
- 3-5 distinct runs in the MLflow experiment
- A short comparison table (CSV or printed) showing how metrics shift across runs
- A clear "best run" identified by metric

**Pitfalls:**
- **Don't run identical configurations multiple times.** Each run should differ in at least one logged param, otherwise comparison is meaningless
- **Don't delete failed runs.** Failed runs are observability evidence too — mark them and keep them

---

## Stage 5 — Build Guardrails

**Goal:** Implement concrete validation logic that prevents misuse and false confidence. This is 15% of the grade and most students half-ass it. Doing it well is easy and high-leverage.

**Decisions to make:**
- **Input guardrails to implement:** empty input, too short, too long, non-English, numeric/symbol-only. Pick at least 4
- **Output guardrails:** confidence threshold (refuse predictions below ~65%), scope disclaimer attached to every response
- **How they fail:** Raise a custom exception with a clear, user-facing message — not a generic error
- **Where they live:** In their own module, isolated from model code. The inference layer calls them; the model itself stays clean

**Outputs before moving on:**
- A guardrails module with each check as a small, testable function
- A test script that triggers each check with an example input and prints `[PASS] reason`
- Guardrails are wired into the inference path (not just defined and ignored)

**Pitfalls:**
- **Don't put guardrails only in the README.** Reviewers want implemented mechanisms, not stated intentions
- **Don't use vague messages.** "Invalid input" is bad. "Input appears to be in 'es', not English" is good
- **Don't skip output guardrails.** The "false confidence" requirement specifically asks for this

---

## Stage 6 — Local Inference Endpoint

**Goal:** Expose the model as a local REST API so the deploy requirement is satisfied with something runnable, not just a script.

**Decisions to make:**
- **Framework:** FastAPI is the standard, has automatic docs at `/docs`, and is trivial to set up. Skip Flask
- **Model loading strategy:** Load once at app startup, not per request. This is non-negotiable — per-request loading would make the API unusable
- **Endpoints:** `/predict` (POST, returns label + confidence + disclaimer) and `/health` (GET, sanity check)
- **Where guardrails fire:** Before inference for input checks, after inference for confidence checks. Both return HTTP 422 with the guardrail message

**Local infrastructure note:**
- The API runs with `uvicorn` on `localhost:8000`. No exposure to the internet. No reverse proxy. No Docker required (though you could add one if you want extra polish)

**Outputs before moving on:**
- The API starts cleanly with one command
- A `curl` example in the README that returns a real prediction
- Hitting `/predict` with bad input returns the guardrail message
- Hitting `/predict` with low-confidence input is rejected by the output guardrail

**Pitfalls:**
- **Don't try to serve via `mlflow models serve`.** It's clunky, hard to wire guardrails into, and worse than FastAPI for this project
- **Don't load the model inside the request handler.** Cold start every request will make the API feel broken
- **Don't return the raw HuggingFace dictionary.** Define a clean response schema with `pydantic`

---

## Stage 7 — Observability Layer

**Goal:** Show the system can be inspected — not just run. Most students stop at "I have MLflow runs"; you should explicitly demonstrate comparison and inspection.

**Decisions to make:**
- **What inspection means here:** MLflow UI run comparison + a small inference log file (one line per prediction served by the API)
- **What to capture in the inference log:** timestamp, input length, predicted label, confidence. This becomes your "production monitoring" surrogate
- **What evidence to commit:** Screenshots of the MLflow UI (experiment list, comparison view, single-run detail), the comparison CSV, the inference log

**Outputs before moving on:**
- MLflow runs are comparable in the UI
- A `runs_comparison.csv` exists, generated from MLflow's API
- Inference log accumulates entries when you hit the API
- 3-4 screenshots saved as evidence

**Pitfalls:**
- **Don't pretend "MLflow is installed" is observability.** Observability is the act of *using* it to compare and inspect. Show that act happening
- **Don't try to install Prometheus/Grafana locally.** Out of scope, time sink, no grading benefit

---

## Stage 8 — Documentation and Report

**Goal:** Make the engineering work legible to a reviewer who will spend 10-15 minutes on your project.

**The README must answer four questions immediately:**
1. What problem does this solve?
2. How do I set it up and run it?
3. How do I run the pipeline, the API, and the MLflow UI?
4. What guardrails exist and how do I see them fire?

**The technical report (`relatorio-entrega.md` template) must:**
- Justify the choice of problem, dataset, and model in 1-2 paragraphs
- Include a pipeline diagram (ASCII is fine)
- List logged params and metrics explicitly
- Show MLflow screenshots
- Describe each guardrail with the input that triggers it
- Reference the comparison artifact
- Acknowledge real limitations (truncation, English-only, domain-specific training)

**Pitfalls:**
- **Don't leave any `[placeholder]` in the report.** Reviewers grep for these
- **Don't write generic limitations.** "May not always be accurate" is filler. "Reviews longer than 512 tokens are truncated and may lose late-occurring sentiment cues" is real

---

## Stage 9 — Final Validation Pass

Before opening the PR, do a clean dry-run from scratch:

- Delete your local virtualenv. Recreate it. Reinstall from `requirements.txt`. Run the pipeline. It should work
- Run the API. Hit it with three example requests: a good one, a non-English one, an empty one. Each should produce the expected behavior
- Open `mlflow ui`. Confirm the experiments and registered model are visible
- Read your own README as if you'd never seen the project. Are the run instructions complete?
- Check that nothing outside your student folder was modified

---

## Big-Picture Order of Operations

```
0. Environment      → tools installed, smoke imports work
1. Smoke test       → end-to-end script, ~20 lines, proves stack
2. Modularize       → real pipeline, one entry point, clean stages
3. MLflow           → params, metrics, artifacts, registered model
4. Multi-runs       → 3-5 runs with varied params for comparison
5. Guardrails       → 4+ input checks, 1+ output check, tests
6. Inference API    → FastAPI, model loaded once, guardrails wired
7. Observability    → comparison CSV, inference log, screenshots
8. Documentation    → README + report, no placeholders
9. Clean rerun      → reproducibility check, then PR
```

**The only stage that's truly optional is fine-tuning.** Skip it. You don't need it, the rubric doesn't reward it, and the pretrained model already performs well on IMDb.

---

## Highest-Risk Early Decisions to Get Right

1. **Python and library versions** — pin them in Stage 0, never touch them again
2. **CPU vs GPU** — commit to CPU, set `device=-1`, never look back
3. **Truncation strategy** — truncate at 512 tokens, document it, move on
4. **Where guardrails live** — separate module, called from API, not inlined
5. **MLflow store location** — default `mlruns/` in project root, committed to git
6. **One orchestration entry point** — not three scripts, not a notebook

Get these six right early and the rest of the project is straightforward execution. Get them wrong and you'll be refactoring during the last 24 hours.
