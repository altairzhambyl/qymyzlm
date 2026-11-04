# evallab/ — benchmarking lab (leaderboard records; runner-backed for measured KazQAD only)

## Goal
Target: no model claim exists until a runner here reproduces it. Today that holds for `measured`
KazQAD records only; the KazMMLU record is legacy-script and `reported` records are others' numbers. Tracks:
1. **Embedding/retrieval** — KazQAD (planka: off-the-shelf mE5-large, re-measured by our runners; the cited card "MRR 0.909" is [UNVERIFIED], protocol-sensitive),
   kk-MTEB task classes (KazQADRetrieval, KazQADReranking) written here first, then PR'd
   upstream to embeddings-benchmark/mteb — the upstream merge is the deliverable.
2. **Generative** — KazMMLU 3-shot (dev split holds only 3 exemplars/subject; runs previously
   labeled 5-shot were effectively 3-shot). Baseline Qwen3-0.6B 32.8% (n=9,870, legacy script
   on RTX 2070, raw output not retained; not yet reproduced by a runner here), ceiling
   Sherkala-8B 41.4% (reported, cross-protocol until re-run here). A 4-bit T4 run measured the
   same model at 30.7% (n=9,870), so compare only within one harness. See root README.
3. **Safety** — Qorgau kk-ru guardrails (arXiv 2502.13640; NO license — eval-only,
   never train on its annotations).

## Package
`kazeval`, import `kazeval`, workspace member of the qymyzlm monorepo.

## Hard constraints
- Every leaderboard row regenerable by one command; no hand-edited numbers.
- Results = committed JSON per run in `evallab/results/` (model, revision, task, numbers, date);
  README leaderboard auto-rendered from them — never edited by hand.
- Tests run offline on fixtures in `evallab/tests/fixtures/`; real dataset pulls only behind
  explicit CLI commands (documented per runner).

## Done means
- `ruff check` clean; `pytest evallab/tests` green offline; mteb tasks instantiate;
  leaderboard renders from sample results JSON.
