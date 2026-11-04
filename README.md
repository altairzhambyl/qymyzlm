# QymyzLM

**Reproducible Kazakh language models.** KazQAD numbers here come from an open runner with the protocol saved in the record, so you can re-run them. KazMMLU numbers are older script runs and are labelled as such. Every number is `measured` (we ran it) or `reported` (someone else's number, shown for context, never as our own).

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?logo=pytorch&logoColor=white)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue)
![Kazakh](https://img.shields.io/badge/Language-Kazakh-green)

Monorepo of the Kazakh model campaign — one uv workspace, three packages:

| Package | Dir | What it is |
|---------|-----|------------|
| `kazeval` | `evallab/` | Benchmarking lab with a [leaderboard](evallab/) generated from committed records (KazQAD retrieval/reranking, KazMMLU). |
| `qymyz-embed` | `embed/` | Kazakh text-embedding model — **active track**: goal is to beat off-the-shelf mE5-large on KazQAD under evallab's pinned protocol. The first fine-tune regressed (see Track 1). |
| `qymyzlm` (import `kazllm`) | `src/` | Kazakh SLM engine — **exploratory track**: ≤600M active params, Engram n-gram memory, QLoRA continual pretraining. April 2026 runs have no established gain (see Track 2). |

Sibling project: [kazakh-nlp-atlas](https://github.com/kekeront/kazakh-nlp-atlas) — data-driven survey of Kazakh NLP + frontier knowledge base.

```bash
uv sync --all-packages --all-extras   # one env for the whole workspace
```

---

## Reproducibility first

This is the point of the whole repo. Every measured KazQAD number comes from a `kazeval` runner: a **pinned dataset revision**, the **exact protocol saved in the record**, fp16 on free Kaggle GPUs. The KazMMLU numbers do not meet this bar yet (see Track 2). Full leaderboard → [`evallab/`](evallab/).

**First measured results (2026-07) — off-the-shelf `intfloat/multilingual-e5-large` on KazQAD:**

| Task | Metric | mE5-large (`measured`) | reference (`reported`) |
|------|--------|:----:|--------|
| KazQAD retrieval — full 825,309-passage corpus | NDCG@10 | **0.329** | 0.389 — KazQAD paper's tuned pipeline |
| KazQAD reranking — pinned hard-neg protocol | MRR@10 | **0.654** | 0.909 — kazembed-v5 card — **unverifiable** (protocol unpublished) |

The widely-cited "mE5-large MRR 0.909 on KazQAD" is **unverifiable**: the candidate pools behind it were never published, and the successor model card reports mE5-large at **0.449** under a same-named setting. A number without a reproducible protocol isn't wrong — it's uncheckable, which for a benchmark is worse. Under our open, pinned protocol (`kazqad-hardneg-bm25-v1`: 100 BM25 candidates, fixed corpus revision), off-the-shelf mE5-large scores **MRR@10 0.654**. Turning uncheckable folklore into numbers you can re-run is why this repo exists. (Full-corpus NDCG@10 0.329 sits honestly *below* the paper's tuned pipeline; off-the-shelf, untuned, reported as-is.)

---

## Track 1 — Kazakh embeddings (active)

**Goal:** the best Kazakh text-embedding model. **Bar:** beat off-the-shelf mE5-large on KazQAD under evallab's pinned protocol, then #1 on kk-MTEB. The first fine-tune (Less-is-More on mE5-base, 0.5/0.5 weight soup) targeted **≥ +2.0 NDCG@10** and **regressed**: full-corpus KazQAD NDCG@10 **0.2077**, below its own base mE5-base at 0.2785 and mE5-large at 0.3294 (`measured`, `kazeval.run_retrieval`, records in [`evallab/results/`](evallab/results/)). Leading hypothesis, untested: KazParC translation pairs train cross-lingual alignment rather than Kazakh passage retrieval.

## Track 2 — Kazakh SLM ≤600M (exploratory)

**Goal:** a sub-1B-active model competitive on KazMMLU. **Levers:** QLoRA continual pretraining · Engram sparse n-gram memory (0 extra FLOPs) · Kazakh SFT · test-time scaling. Reachability on free compute is an **open question**, gated by a token-budget study before any paid run.

Generative baselines — KazMMLU 3-shot, 9,870 Kazakh questions ([MBZUAI/KazMMLU](https://huggingface.co/datasets/MBZUAI/KazMMLU)). Unmarked rows are `measured` in April 2026 with the legacy `scripts/benchmark_baselines.py` (fp16, RTX 2070, unpinned revisions), **not a `kazeval` runner**. Raw output was not retained, and the current script truncates long prompts differently, so these rows are not exactly reproducible. *Italic* = reported (different harness, context only). The dev split holds only 3 exemplars/subject, so earlier "5-shot" runs were effectively **3-shot**.

| Model | Params | KazMMLU 3-shot | Tok/word |
|-------|--------|:----:|:----:|
| Qwen2.5-1.5B | 1.54B | 34.3% | 4.88 |
| **Qwen3-0.6B-Base** (primary target) | 0.6B | **32.8%** | 4.88 |
| Qwen2.5-0.5B | 0.49B | 28.8% | 4.88 |
| Gemma3-1B-it | 1.0B | 28.7% | 3.27 |
| Llama-3.2-1B | 1.24B | 25.1% | 4.80 |
| Ekitil-Qwen3-600M (Kazakh-adapted Qwen) | 0.67B | 23.7% | 1.43 |
| Random baseline | — | 25.0% | — |
| *Sherkala-Chat-8B* | *8B* | *41.4%* | *2.04* |
| *Llama-3.1-70B* | *70B* | *55.2%* | *4.73* |

**Takeaways (n=9,870, legacy harness):** Qwen3-0.6B scores highest below 1.5B parameters; Qwen2.5-1.5B (1.54B) is higher at 34.3%. Ekitil — a Kazakh-adapted Qwen with an elite tokenizer (1.43 tok/word) — still scores below the 25% random reference: **a good tokenizer is not knowledge**. Changing the harness moved the same base model by 2.1pp (next section), so small gaps in this table are not established differences. The gap to Sherkala-8B is cross-protocol until Sherkala is re-run through our runner.

### Generative track: April 2026 runs

Exploratory QLoRA and Engram runs on Qwen3-0.6B-Base. **No gain from them is established.** Rows use different sample sizes and harnesses, so compare only rows with the same n and harness. All are 3-shot in effect; the kernels print "5-shot". Intervals are Wilson 95%, computed from the rounded accuracy.

| Config | KazMMLU | n | Questions | 95% CI | Harness | Evidence | Status |
|--------|:------:|:-:|-----------|:-----:|---------|----------|--------|
| Base | 32.8% | 9,870 | all | 31.9-33.7 | fp16, RTX 2070, legacy script | first committed in `78abafb`; raw output **unavailable** | measured, not exactly reproducible |
| Base | 30.7% | 9,870 | all | 29.8-31.6 | 4-bit, T4, notebook | log of kernel `engram-module` | measured |
| Base | 31.5% | **200** | **first 200 rows, Biology only** | 25.5-38.2 | 4-bit, T4, notebook | log of kernel `qlora-kazakh-continual-pretraining-2500` | measured, **200-question subset, not representative** |
| + session 1 `step_2500` | 35.5% | **200** (inferred) | **first 200 rows, Biology only** (inferred) | 29.2-42.3 | 4-bit, T4, notebook (inferred) | output **unavailable**; n inferred from notebook code | **unverifiable, 200-question subset** |
| + sessions 1+2 `step_2500` | 30.0% | 9,870 (inferred) | all (inferred) | 29.1-30.9 | 4-bit, T4, notebook (inferred) | output **unavailable**; n inferred from notebook code | **unverifiable** |
| + session 1 `step_2500` + Engram | 31.0% | 9,870 | all | 30.1-31.9 | 4-bit, T4, notebook | log and `results.json` of kernel `engram-dream` | measured |
| + 9:1 replay `step_2500` | not evaluated | — | — | — | — | adapter in kernel `engram-module`; its log has no evaluation | no number |

How to read this:

- **35.5% vs 31.5% is not a KazMMLU result.** Both cover one subject of twelve, their intervals overlap, and the 35.5% output did not survive. Do not cite it as a +4pp lift.
- **The "catastrophic forgetting" story and the Engram delta are withdrawn.** The kernels' printed "-4.5pp" compares 31.0% (n=9,870) with 35.5% (n=200), and the 30.0% vs 31.5% drop mixes the same two sample sizes. The ablation block in `engram-dream`'s `results.json` hard-codes 31.5 and 35.5; only its 31.0 was measured in that run. Among the full-set 4-bit numbers (30.7, 30.0, 31.0) the spread is 1.0pp with intervals near ±0.9pp, so neither a gain nor a loss is established (inferred, because 30.0% has no surviving output). Session 1 has no full-set number at all.
- **The harness matters.** The same base model differs by 2.1pp between 32.8% and 30.7%, which is more than the spread above.
- **Perplexity is not reported as a result.** The 8.87 and 3.89 figures come from the first 50 long Kazakh Wikipedia articles, the same split session 1 streamed for training, so overlap is very likely (inferred from code). The 7.68 base figure uses 200 HPLT documents and the 3.8 Engram figure uses 200 Wikipedia articles, so those are not comparable either.

Checkpoints. Sessions 1 and 2 stopped near step 2,950 of 3,051 at about 12 h, having processed about 48M tokens each, but the last save was `step_2500`. The evaluated weights therefore saw **about 41M tokens per session**, not the "48M" and "96M" of earlier labels. Adapters sit in private Kaggle kernel output and are not published here.

| Adapter | Tokens seen | Corpus | Kernel | `adapter_model.safetensors` sha256 |
|---------|:----------:|--------|--------|------------------------------------|
| Session 1 `step_2500` | ~41M | `wikimedia/wikipedia` 20231101.kk | `qlora-kazakh-continual-pretraining-2500` | `4056d32ffbbd8175715fedf6fe5359e51cb2b8a5a5858eff79ac2bf9b2239550` |
| Session 2 `step_2500` | ~82M cumulative (~41M new) | `kz-transformers/multidomain-kazakh-dataset` only; later sources never reached | `50-100m-upgrade` | `4d75417e9cd3eb37465821e4485c190ee73fed22bdb584ee7f0278606b62d901` |
| 9:1 replay `step_2500` | ~41M | HPLT new data with replay ratio 0.1 | `engram-module` | not downloaded |

Session 2 is not "web data", and it also restarted with a fresh optimizer at LR 1e-4, so corpus is confounded with schedule.

To replace all of the above: [`scripts/eval_kazmmlu_full.py`](scripts/eval_kazmmlu_full.py) scores the base and each adapter on all 9,870 questions in one harness and saves per-subject and per-question output. It needs about 30 minutes per model on a T4 and **has not been run**; `--dry-run` rehearses it on CPU.

---

## Roadmap

- [x] KazMMLU generative baselines (6 models `measured` with the legacy script, 2 `reported`)
- [x] **First reproducible KazQAD embedding numbers (mE5-large) — the 0.909 claim is unverifiable**
- [ ] KazQAD leaderboard v0 — BGE-M3, Qwen3-Embedding, LaBSE (MiniLM measured: NDCG@10 0.0005)
- [x] First `qymyz-embed` fine-tune (Less-is-More) — ran and regressed; the mE5-large baseline is **not** beaten
- [ ] Generative track: full-set same-harness re-evaluation of the April adapters (script ready, not run)
- [ ] Generative track: token-budget go/no-go → QLoRA continual PT

## References

- [KazQAD](https://arxiv.org/abs/2404.04487) — Kazakh open-domain QA + retrieval (LREC-COLING 2024)
- [KazMMLU](https://arxiv.org/abs/2502.12829) — Kazakh MMLU, ACL 2025 (23K questions)
- [Sherkala](https://arxiv.org/abs/2503.01493) — 8B Kazakh continual PT (ISSAI)
- [SozKZ](https://arxiv.org/abs/2603.20854) — 600M Kazakh from scratch
- [Engram](https://arxiv.org/abs/2601.07372) — conditional N-gram memory (DeepSeek-AI)

## License

Apache 2.0
