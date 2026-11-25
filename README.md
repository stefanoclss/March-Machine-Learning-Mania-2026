# March Machine Learning Mania 2026

[![Autonomous Synthesis](https://img.shields.io/badge/Synthesized%20By-sovLoop%20(Stealth)-0A66C2?style=for-the-badge)](#how-this-pipeline-was-built)
[![Foundation LLM](https://img.shields.io/badge/Model-qwen3.5%3A9b-111827?style=for-the-badge)](#how-this-pipeline-was-built)
[![Validation](https://img.shields.io/badge/5--Fold%20OOF-BRIER%200.16730-059669?style=for-the-badge)](#verified-validation-benchmark)
[![Security](https://img.shields.io/badge/Zero--Pickle-CWE--502%20Compliant-374151?style=for-the-badge)](#governance--production-guarantees)
[![Kaggle Public Notebook](https://img.shields.io/badge/Kaggle%20Notebook-Public%20Kernel-20BEFF?style=for-the-badge)](https://www.kaggle.com/code/stefanoclss/march-machine-learning-mania-2026)
[![Kaggle Challenge](https://img.shields.io/badge/Challenge-Benchmark-0284C7?style=for-the-badge)](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)

> **Autonomous End-to-End Machine Learning Engineering by sovLoop**
> From raw tabular schema forensics to interpretable `preprocessing(df)` & `feature_engineering(df)` pipelines, Optuna-tuned multi-family GBDT ensembles, and zero-pickle production packaging—evolved step-by-step using **`qwen3.5:9b`**.

---

## Executive Overview

Predict `Pred` (binary classification) across `91` structured predictors on `march-machine-learning-mania-2026` under strict leak-free 5-fold out-of-fold validation to minimize `BRIER`.

Production data science teams need pipelines that are not only accurate, but **readable, explainable, and maintainable**. Every transformation in this repository is organized by pipeline topology stage (`preprocessing`, `feature_engineering`, `objective`, `models`, `postprocess`) and validated incrementally via **sovLoop** (operating in private stealth mode).

---

## How This Pipeline Was Built

1. **Simple Baseline -> Gradual Verified Evolution**: Starting from a clean baseline, **sovLoop** (guided by `qwen3.5:9b`) evolved one stage at a time—adding explicit column hygiene, domain features, leak-free target formulation, and Optuna hyperparameter optimization.
2. **Strict Out-of-Fold Scientific Gatekeeping**: Candidate modifications to preprocessing, feature engineering, estimator hyperparameters, and post-processing were evaluated across **100% of 5-fold out-of-fold validation rows**. Any change that failed to improve CV was rejected.
3. **Atomic Single-Module Git Lineage**: After initial scaffolding, **each commit in this repository modifies strictly one atomic stage file** (`pipeline/models.py`, `pipeline/features.py`, `pipeline/preprocess.py`, `pipeline/objective.py`, or `pipeline/postprocess.py`). Inspect `git log -p` to audit every single lift.
4. **Single-Notebook End-to-End Reproducibility (Zero-Pickle / No Git LFS)**: Place raw tables in `data/` and run **`pipeline.ipynb`** top-to-bottom to reproduce visual EDA (`eda_pipeline.py`), K-Fold LightGBM/Optuna training (`training_pipeline.py` -> `artifacts/model_artifact.json`), and pure artifact holdout inferencing (`inference_pipeline.py` -> `submission.csv`).

---

## Verified Validation Benchmark

| Benchmark Dimension | Verified Result |
| :--- | :--- |
| **Challenge / Dataset** | **March Machine Learning Mania 2026** (`march-machine-learning-mania-2026`) |
| **Target Variable (`y`)** | `Pred` (`binary_classification`) |
| **Evaluation KPI** | `BRIER` (Lower is better) |
| **Baseline 5-Fold OOF Score** | `0.17154` |
| **Champion 5-Fold OOF Score** | **`0.16730`** (`+0.00424` verified lift vs. baseline) |
| **Cross-Fold Stability (Std. Dev.)** | `±0.01123` across `5` strict folds |
| **Synthesis LLM** | `qwen3.5:9b` (via **sovLoop**) |

---

## Individual Model Benchmarks vs. Unrestricted Ensemble

Every individual model inside the unrestricted ensemble is benchmarked independently for standalone validation accuracy, training time, and inference latency—so production engineering teams can either deploy the **full unrestricted SOTA ensemble** or select the **fastest / best solo model** (`SELECTED_MODEL_KEY`) for strict SLA budgets:

| Model Family (`key`) | Standalone Val Score | Mean Fold Fit | Mean Fold Inference | Ensemble Weight | Deployment Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Optuna LightGBM** (`lgbm`) | `0.16523 ± 0.01122` | `1865.6 ms` | `3.96 ms` | `43.8%` | `Best Solo + Fastest SLA` |
| **CatBoost Symmetric** (`catboost`) | `0.16554 ± 0.01183` | `3297.1 ms` | `4.63 ms` | `22.9%` | `Active Ensemble Member` |
| **Optuna XGBoost** (`xgb`) | `0.16581 ± 0.01141` | `5899.1 ms` | `3.96 ms` | `12.1%` | `Active Ensemble Member` |
| **TabArena RealMLP** (`realmlp`) | `0.16789 ± 0.01099` | `308.5 ms` | `5.10 ms` | `21.1%` | `Active Ensemble Member` |

---

## Autonomous Build Ledger (Commit-by-Commit Scientific Progression)

Every row below corresponds to a verified out-of-fold promotion recorded as an **isolated single-module commit** in `git log`:

| Iteration | Module Evolved | Active Synthesis Operations | 5-Fold OOF `BRIER` | Lift | Engineering Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gen 0** | `scaffold` | Baseline initialization | `0.17154` | `baseline` | Deterministic 5-fold leak-free baseline anchor |
| **Gen 1** | `pipeline/models.py` | `realmlp_tuned_head`, `rff_mlp_neural_head` | `0.17070` | `+0.00084` | Replacing the saturated L2 Ridge baseline with RealMLP and RFF MLP neural heads will capture non-linear feature interactions in high-value quantiles... |
| **Gen 2** | `pipeline/preprocess.py` | `median_impute`, `winsorize_iqr` | `0.17025` | `+0.00045` | Replacing mean imputation with median imputation and IQR winsorization will reduce the influence of extreme values in high-residual columns... |
| **Gen 3** | `pipeline/postprocess.py` | `tail_shrinkage`, `isotonic_pav_calibration` | `0.17012` | `+0.00013` | Applying James-Stein tail shrinkage followed by isotonic PAV calibration will reduce the Brier score from 0.3316 to below 0.25 by correcting the... |
| **Gen 4** | `pipeline/models.py` | `catboost_tuned`, `lgbm_optuna_tuned`, `xgb_optuna_tuned` | `0.16754` | `+0.00258` | Replacing neural heads with tuned GBDTs will reduce under-prediction bias for high-margin teams by leveraging native non-linear splitting on ordinal and... |
| **Gen 5** | `pipeline/models.py` | `catboost_tuned`, `lgbm_optuna_tuned`, `xgb_optuna_tuned`, `realmlp_tuned_head` | `0.16730` | `+0.00024` | Coordinate-ascent probe on `model`: Test 4-model ensemble adding `TabArena RealMLP (T+E) Parametric-Mish MLP` |

---

## Modular Architecture & Repository Layout

```mermaid
flowchart LR
    DATA["Raw Data (91 predictors)"] --> SPLIT["5-Fold CV Split"]
    SPLIT --> PRE["1. Preprocessing: median_impute, winsorize_iqr"]
    PRE --> FEAT["2. Features: raw_passthrough"]
    FEAT --> OBJ["3. Objective: standard_task_loss"]
    OBJ --> M0["4. Ensemble: Optuna LightGBM (44%) + CatBoost Symmetric (23%) + Optuna XGBoost (12%) + TabArena RealMLP (21%)"]
    M0 --> POST["5. Post-Processing: tail_shrinkage, isotonic_pav_calibration"]
    POST --> OUT["5-Fold OOF & Holdout (BRIER = 0.16730)"]
```

| Deliverable | Role in Production Workflow |
| :--- | :--- |
| **`pipeline.ipynb`** | **Primary Topology Walkthrough Notebook**: explains every stage with visual charts and reproduces EDA, K-Fold training, and holdout scoring. |
| `eda.ipynb` / `eda_pipeline.py` | Schema diagnostics, missingness bar charts, and target distribution plots via `pipeline/eda.py`. |
| `train.ipynb` / `training_pipeline.py` | Leak-free K-Fold LightGBM + Optuna CV orchestrator exporting zero-pickle `artifacts/model_artifact.json` + `fold_*.lgb`. |
| `inference.ipynb` / `inference_pipeline.py` | Pure artifact-driven holdout scoring (`--artifact artifacts/model_artifact.json`) with zero re-training. |
| `pipeline/*.py` | Readable stage modules (`preprocessing(df)`, `feature_engineering(df)`, `models.py`) evolved one file per commit. |

---

## Governance & Production Guarantees

- **100% Validation Fold Integrity**: Imputation statistics, IQR bounds, Optuna inner splits, and target transforms fit strictly inside training folds.
- **Zero-Pickle Security (`CWE-502`)**: Models serialize to human-auditable LightGBM text/JSON (`fold_*.lgb`, `model_artifact.json`) — zero unsafe `.pkl` binaries.
- **Data-Scientist Readability**: Explicit column-by-column `pandas` transformations that any engineer can inspect, explain to stakeholders, and extend.

---

## Quickstart (1-Notebook Reproduction & CLI)

```bash
pip install -r requirements.txt

# Option A — Reproduce Training + Inferencing in ONE Notebook
jupyter execute pipeline.ipynb

# Option B — Modular CLI Orchestrators
python eda_pipeline.py --input data/train.csv
python training_pipeline.py --train data/train.csv --artifact-out artifacts/model_artifact.json --oof-out oof_predictions.csv
python inference_pipeline.py --artifact artifacts/model_artifact.json --test data/test.csv --output submission.csv
```

---

## About the Author

### Stefano Claes — AI Systems Architect

Working on **sovereign local-first AI, neuro-symbolic & agentic harnesses, and closed-loop recursive self-improvement**—pairing autonomous iteration with deterministic verification, zero-trust sandboxing, and human-readable engineering.

| Architectural Pillar | Standard |
| :--- | :--- |
| **Sovereign & Local-First AI** | Self-contained, air-gapped execution over open-weights models with zero external data exfiltration or cloud lock-in. |
| **Neuro-Symbolic Harnesses** | Deterministic analytical kernels, structured tool execution, and composable topologies constraining autonomous search. |
| **Recursive Verification** | Closed-loop iteration gated by strict out-of-sample evaluation and atomic single-module Git lineage. |
| **Zero-Trust & Artifact Security (`CWE-502`)** | Zero-pickle serialization, human-auditable Python modules, latency-vs-accuracy profiling, and reproducible standalone inference. |

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Stefano%20Claes-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/stefano-claes-266435140/) [![GitHub](https://img.shields.io/badge/GitHub-%40stefanoclss-111827?style=for-the-badge&logo=github&logoColor=white)](https://github.com/stefanoclss)

- **Author**: **Stefano Claes** (AI Systems Architect)
- **LinkedIn**: [https://www.linkedin.com/in/stefano-claes-266435140/](https://www.linkedin.com/in/stefano-claes-266435140/)
- **GitHub**: [https://github.com/stefanoclss](https://github.com/stefanoclss)
