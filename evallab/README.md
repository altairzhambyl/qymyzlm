# evallab — Kazakh benchmarking lab (`kazeval`)

Target: no model claim exists until a runner here reproduces it. Today that holds
for the `measured` KazQAD rows (`reported` rows are other people's numbers, and the
`qymyz-embed` model itself is not public). The only KazMMLU row is older: it comes from the legacy
`scripts/benchmark_baselines.py` with no raw output retained, and no committed record
comes from `kazeval.run_kazmmlu` yet. Every leaderboard table is rendered from a
committed JSON record in `results/`, so nothing below the markers is edited by hand.

## Tracks

| Track | Tasks | Runner |
|---|---|---|
| Embedding / retrieval | `KazQADRetrieval` (full 825,309-passage corpus, comparable to arXiv:2404.04487), `KazQADReranking` + `KazQAD-HardNeg` (deterministic `kazqad-hardneg-bm25-v1` protocol, `kazeval.hardneg`) | `python -m kazeval.run_retrieval` |
| Generative | `KazMMLU-kk` — Kazakh subset of MBZUAI/KazMMLU, 12 subjects, 9,870 test questions, **3-shot** (dev holds only 3 exemplars/subject; historical "5-shot" numbers were effectively 3-shot); dataset revision pinned in the task YAML | `python -m kazeval.run_kazmmlu` (lm-eval loglikelihood, a different harness from the legacy record's next-token letter logit) |
| Safety | `Qorgau-kk` (arXiv:2502.13640) — **no license declared upstream ⇒ eval-only, never train on it**; loaders in `kazeval.qorgau`, judge protocol not wired yet | — |

## Usage

```bash
# kk-MTEB retrieval + reranking (gated dataset: accept conditions on
# https://huggingface.co/datasets/issai/kazqad-retrieval first)
python -m kazeval.run_retrieval --model intfloat/multilingual-e5-large

# KazMMLU Kazakh subset, 3-shot (the dev-split maximum)
python -m kazeval.run_kazmmlu --model Qwen/Qwen3-0.6B-Base

# re-render the leaderboard below from results/*.json
python -m kazeval.leaderboard
# CI-style staleness check (exit 2 if the section is out of date)
python -m kazeval.leaderboard --check
```

**GPU runs happen on Kaggle** (compute policy 2026-07-04: free plan, T4×2/P100 16 GB fp16,
30 GPU-h/week): upload `kaggle/kazeval_kaggle.ipynb`, add `HF_TOKEN` to Kaggle Secrets,
push `main` first (the notebook clones this repo from GitHub). Download the produced
JSON records into `results/` and re-render the leaderboard locally.

Both runners write one validated record per (task, split) into `results/`
(`provenance: measured`); externally sourced numbers are committed as
`provenance: reported` and stay flagged until re-measured in-lab.

Tests are fully offline (`fixtures/` only): `pytest evallab/tests -q`.

## Leaderboard

<!-- LEADERBOARD:START -->
_Auto-generated from `evallab/results/*.json` by `python -m kazeval.leaderboard` — do not edit by hand._

### At a glance

_One comparable metric per track, best first. `measured` = produced in-lab, by a kazeval runner or the legacy baseline script (see Source below); `reported` = external number, shown for context until re-measured in-lab. Full per-record metrics below._

**Embedding / retrieval (KazQAD)**

| Model | Task | nDCG@10 | Provenance | Date |
|---|---|---|---|---|
| BM25 + fine-tuned reranker (KazQAD paper best pipeline) | KazQADRetrieval | 0.3890 | reported | 2024-04-06 |
| intfloat/multilingual-e5-large | KazQADRetrieval | 0.3294 | measured | 2026-07-04 |
| intfloat/multilingual-e5-base | KazQADRetrieval | 0.2785 | measured | 2026-07-05 |
| kekeront/qymyz-embed-lim-v0-souped | KazQADRetrieval | 0.2077 | measured | 2026-07-05 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | KazQADRetrieval | 0.0005 | measured | 2026-07-05 |
| intfloat/multilingual-e5-large | KazQADReranking | 0.6943 | measured | 2026-07-04 |
| intfloat/multilingual-e5-large | KazQAD-HardNeg | — | reported | 2025-12-04 |

**Generative (KazMMLU)**

| Model | Task | Accuracy | Provenance | Date |
|---|---|---|---|---|
| Qwen/Qwen3-0.6B-Base | KazMMLU-kk | 0.3280 | measured | 2026-04-06 |

### Full records

### Embedding / retrieval (KazQAD)

| Model | Revision | Task | Split | ndcg_at_10 | map_at_1000 | mrr | hits_at_1 | hits_at_5 | mrr_at_10 | ndcg_at_100 | recall_at_10 | recall_at_100 | Provenance | Source | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 + fine-tuned reranker (KazQAD paper best pipeline) | — | KazQADRetrieval | test | 0.3890 | — | 0.3820 | — | — | — | — | — | — | reported | arXiv:2404.04487 (KazQAD, LREC-COLING 2024) | 2024-04-06 |
| intfloat/multilingual-e5-large | ab10c1a7 | KazQADRetrieval | test | 0.3294 | 0.2772 | — | — | — | 0.3045 | 0.3831 | 0.4836 | 0.7202 | measured | kazeval.run_retrieval (mteb 2.16.3, encode dtype=float16) | 2026-07-04 |
| intfloat/multilingual-e5-base | d13f1b27 | KazQADRetrieval | test | 0.2785 | 0.2321 | — | — | — | 0.2568 | 0.3301 | 0.4176 | 0.6449 | measured | kazeval.run_retrieval (mteb 2.16.3, encode dtype=float16) | 2026-07-05 |
| kekeront/qymyz-embed-lim-v0-souped | dacc18a5 | KazQADRetrieval | test | 0.2077 | 0.1736 | — | — | — | 0.1911 | 0.2676 | 0.3211 | 0.5876 | measured | kazeval.run_retrieval (mteb 2.16.3, encode dtype=float16) | 2026-07-05 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | bf3bf13a | KazQADRetrieval | test | 0.0005 | 0.0005 | — | — | — | 0.0005 | 0.0006 | 0.0005 | 0.0008 | measured | kazeval.run_retrieval (mteb 2.16.3, encode dtype=float16) | 2026-07-05 |
| intfloat/multilingual-e5-large | ab10c1a7 | KazQADReranking | test | 0.6943 | 0.6226 | — | — | — | 0.6545 | 0.7222 | 0.8917 | 1.0000 | measured | kazeval.run_retrieval (mteb 2.16.3, encode dtype=float16) | 2026-07-04 |
| intfloat/multilingual-e5-large | — | KazQAD-HardNeg | test | — | — | 0.9090 | 0.8500 | 0.9900 | — | — | — | — | reported | https://huggingface.co/Nurlykhan/kazembed-v5 | 2025-12-04 |

### Generative (KazMMLU)

| Model | Revision | Task | Split | acc | Provenance | Source | Date |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B-Base | — | KazMMLU-kk | test | 0.3280 | measured | scripts/benchmark_baselines.py at commit 78abafb (first committed there; no raw output) | 2026-04-06 |

### Safety (Qorgau)

_No results yet._
<!-- LEADERBOARD:END -->
