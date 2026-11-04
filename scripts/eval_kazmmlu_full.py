#!/usr/bin/env python3
"""Same-harness KazMMLU evaluation of the base model and QLoRA adapters, all 9,870 questions.

Why this exists: the April 2026 numbers mix sample sizes and harnesses (see "Generative
track: April 2026 runs" in README.md). This scores every model in one run, one harness:
Qwen3-0.6B-Base, 4-bit nf4 with fp16 compute (as the April kernels), 3-shot from dev,
next-token letter-logit via scripts/benchmark_baselines.py, pinned dataset and model.
Prompts over budget drop their earliest shots, whereas the April kernels cut the prompt's
end, so comparisons must stay inside one run of this script.

Output in --output; finished models are skipped, so a killed session can be resumed:
    <name>.summary.json      accuracy, Wilson 95% interval, per-subject accuracy, adapter sha256
    <name>.questions.jsonl   one raw prediction per question, for paired comparison
    comparison.json          every adapter against base, with discordant-pair counts

There is deliberately no question-limit flag: a run that does not score exactly 9,870
questions fails instead of producing another partial number.

Cost: 29-32 min per model on one Kaggle T4 (measured, April kernels), so about 2 h for
base + three adapters. Needs a GPU, bitsandbytes and peft; downloads the base model and
dataset. Nothing here launches a kernel.

Usage on a Kaggle T4, with the adapter kernels attached as inputs:
    python scripts/eval_kazmmlu_full.py \\
        --adapter s1_step2500=/kaggle/input/<s1-kernel>/qlora_qwen3-0.6b_100m/step_2500 \\
        --adapter s2_step2500=/kaggle/input/<s2-kernel>/qlora_qwen3-0.6b_100m_s2/step_2500 \\
        --adapter dream_9_1_step2500=/kaggle/input/<dream-kernel>/run1_dream_9_1/step_2500

CPU-safe rehearsal, no downloads, GPU or model weights (adapter paths, if given, are really
checked; "s1_step2500" and "s2_step2500" must also match their known sha256):
    python scripts/eval_kazmmlu_full.py --dry-run
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections import Counter
from math import sqrt
from pathlib import Path
from types import SimpleNamespace

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent


def _load_harness():
    """Import the sibling benchmark script by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "benchmark_baselines", _SCRIPT_DIR / "benchmark_baselines.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bb = _load_harness()

BASE_NAME = "base"
BASE_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
BASE_MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"  # hub lastModified 2025-07-26
EXPECTED_N_QUESTIONS = 9870
N_SHOT = 3
MAX_LENGTH = 2048

# Kaggle kernel-output hashes (2026-11-04). The dream adapter's weights were not downloaded,
# so it has no known hash; its sha256 is still recorded in the summary.
KNOWN_ADAPTER_SHA256 = {
    "s1_step2500": "4056d32ffbbd8175715fedf6fe5359e51cb2b8a5a5858eff79ac2bf9b2239550",
    "s2_step2500": "4d75417e9cd3eb37465821e4485c190ee73fed22bdb584ee7f0278606b62d901",
}


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(correct: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a proportion, as fractions."""
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return centre - half, centre + half


def parse_adapter(spec: str) -> tuple[str, Path]:
    """Parse a NAME=PATH adapter argument."""
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {spec!r}")
    if name == BASE_NAME or not name.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError(f"bad adapter name {name!r} (reserved or unsafe)")
    return name, Path(path)


def check_adapter(name: str, path: Path) -> str:
    """Fail fast on an unusable adapter directory; return the weights' sha256."""
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    missing = [str(p) for p in (config_path, weights_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"adapter {name!r}: missing {missing}")
    base = json.loads(config_path.read_text(encoding="utf-8")).get("base_model_name_or_path")
    if base != BASE_MODEL_ID:
        raise ValueError(f"adapter {name!r}: base model is {base!r}, expected {BASE_MODEL_ID!r}")
    digest = sha256_file(weights_path)
    expected = KNOWN_ADAPTER_SHA256.get(name)
    if expected is not None and digest != expected:
        raise ValueError(f"adapter {name!r}: sha256 {digest} does not match known {expected}")
    return digest


def summarize(name: str, questions: list[dict], adapter_sha256: str | None) -> dict:
    """Overall and per-subject accuracy from raw per-question records."""
    total = len(questions)
    correct = sum(q["pred"] == q["gold"] for q in questions)
    low, high = wilson_interval(correct, total)
    subject_totals = Counter(q["config"] for q in questions)
    subject_correct = Counter(q["config"] for q in questions if q["pred"] == q["gold"])
    return {
        "name": name,
        "n_questions": total,
        "n_correct": correct,
        "accuracy": correct / total,
        "wilson_95": [low, high],
        "per_subject": {
            config: {"n": n, "accuracy": subject_correct[config] / n}
            for config, n in sorted(subject_totals.items())
        },
        "adapter_sha256": adapter_sha256,
    }


def compare_to_base(base: list[dict], other: list[dict]) -> dict:
    """Paired comparison on the same questions; inputs must share question order."""
    if len(base) != len(other) or any(
        a["config"] != b["config"] or a["gold"] != b["gold"] for a, b in zip(base, other)
    ):
        raise ValueError("per-question records are not aligned with the base run")
    base_only = sum(a["pred"] == a["gold"] and b["pred"] != b["gold"] for a, b in zip(base, other))
    other_only = sum(a["pred"] != a["gold"] and b["pred"] == b["gold"] for a, b in zip(base, other))
    return {
        "delta_pp": 100 * (other_only - base_only) / len(base),
        "base_only_correct": base_only,
        "other_only_correct": other_only,
    }


def evaluate_and_save(
    name: str,
    model,
    tokenizer,
    test_data: list[dict],
    dev_data: list[dict],
    output_dir: Path,
    adapter_sha256: str | None,
    metadata: dict,
    expected_n: int = EXPECTED_N_QUESTIONS,
) -> dict:
    """Score one model, refuse any partial run, write the raw records then the summary."""
    questions: list[dict] = []
    _, n_questions, shot_stats = bb.evaluate_kazmmlu(
        model,
        tokenizer,
        test_data,
        dev_data,
        n_shot=N_SHOT,
        max_length=MAX_LENGTH,
        per_question=questions,
    )
    if n_questions != expected_n:
        raise RuntimeError(f"{name}: scored {n_questions} questions, expected {expected_n}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{name}.questions.jsonl", "w", encoding="utf-8") as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + "\n")
    summary = summarize(name, questions, adapter_sha256)
    summary["shot_stats"] = shot_stats
    summary["metadata"] = metadata
    # Summary last: its presence marks the model as finished for resume.
    (output_dir / f"{name}.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    low, high = summary["wilson_95"]
    print(
        f"  {name}: {summary['accuracy'] * 100:.1f}% "
        f"(n={n_questions}, 95% CI {low * 100:.1f}-{high * 100:.1f})"
    )
    return summary


def write_comparison(names: list[str], output_dir: Path) -> dict:
    """Write comparison.json from the raw per-question files of a finished run."""

    def read(name: str) -> list[dict]:
        path = output_dir / f"{name}.questions.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    base = read(BASE_NAME)
    comparison = {name: compare_to_base(base, read(name)) for name in names if name != BASE_NAME}
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    return comparison


def _load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, revision=BASE_MODEL_REVISION, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(adapter_path: Path | None):
    """Fresh 4-bit base (plus adapter) on GPU 0, matching the April kernels' model setup."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
        trust_remote_code=True,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    return model.eval()


# ── Dry run: exercises the real harness and outputs on synthetic rows, CPU only ───────────


class _DryRunTokenizer:
    """Whitespace tokenizer producing ids a small embedding can take."""

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        ids = torch.tensor([[len(word) % 8 for word in text.split()] or [0]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(text.strip()[0]) - ord("A")]


class _DryRunModel(torch.nn.Module):
    """Tiny CPU model with the (input_ids, attention_mask) -> .logits interface."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = torch.nn.Embedding(8, 8)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.embed(input_ids))


def _dry_run_rows() -> tuple[list[dict], list[dict]]:
    def row(config: str, i: int) -> dict:
        return {
            "_config": config,
            "Question": f"Сұрақ {i} {'ұзын ' * (i % 3)}?",
            "Option A": "бір",
            "Option B": "екі",
            "Option C": "үш",
            "Option D": "төрт",
            "Option E": "",
            "Answer Key": "ABCD"[i % 4],
        }

    configs = ("Biology (High School in kaz)", "Math (High School in kaz)")
    test = [row(config, i) for config in configs for i in range(10)]
    dev = [row(config, i) for config in configs for i in range(3)]
    return test, dev


def _summaries_by_kind(output_dir: Path) -> tuple[set[str], set[str]]:
    """Names of finished models in a directory: (real, dry-run)."""
    real, rehearsal = set(), set()
    for path in output_dir.glob("*.summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        kind = rehearsal if summary["metadata"].get("dry_run") else real
        kind.add(summary["name"])
    return real, rehearsal


def dry_run(adapters: list[tuple[str, Path]], output_dir: Path) -> dict:
    """Rehearse the full flow with no network, GPU or model weights."""
    real, _ = _summaries_by_kind(output_dir)
    if real:
        raise RuntimeError(f"{output_dir} holds real results {sorted(real)}; not overwriting")
    digests = {name: check_adapter(name, path) for name, path in adapters}
    test, dev = _dry_run_rows()
    names = [BASE_NAME, *(digests or ["dry_adapter"])]
    print(f"DRY RUN: synthetic data, scores are meaningless. Models: {names}")
    for seed, name in enumerate(names):
        evaluate_and_save(
            name,
            _DryRunModel(seed),
            _DryRunTokenizer(),
            test,
            dev,
            output_dir,
            digests.get(name),
            {"dry_run": True},
            expected_n=len(test),
        )
    comparison = write_comparison(names, output_dir)
    print(f"DRY RUN OK: {output_dir} (a real run costs ~{30 * len(names)} GPU-min on a T4)")
    return comparison


def run(adapters: list[tuple[str, Path]], output_dir: Path) -> None:
    """The real evaluation. Requires a GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit evaluation needs a CUDA GPU; use --dry-run on CPU")
    finished, rehearsal = _summaries_by_kind(output_dir)
    if rehearsal:
        raise RuntimeError(f"{output_dir} holds dry-run output; use an empty directory")
    digests = {name: check_adapter(name, path) for name, path in adapters}
    paths: dict[str, Path | None] = {BASE_NAME: None, **dict(adapters)}
    metadata = {
        "date": time.strftime("%Y-%m-%d"),
        "repo_commit": bb.get_repo_commit(),
        "device": torch.cuda.get_device_name(0),
        "model": BASE_MODEL_ID,
        "model_revision": BASE_MODEL_REVISION,
        "dataset": "MBZUAI/KazMMLU (Kazakh subset)",
        "dataset_revision": bb.KAZMMLU_REVISION,
        "quantization": "4-bit nf4, fp16 compute, double quant",
        "n_shot": N_SHOT,
        "max_length": MAX_LENGTH,
    }
    test_data, dev_data = bb.load_kazmmlu_kazakh()
    tokenizer = _load_tokenizer()
    for name, path in paths.items():
        if name in finished:
            print(f"  {name}: already finished, skipping")
            continue
        print(f"Evaluating {name} ...")
        model = _load_model(path)
        evaluate_and_save(
            name, model, tokenizer, test_data, dev_data, output_dir, digests.get(name), metadata
        )
        del model
        torch.cuda.empty_cache()
    print(json.dumps(write_comparison(list(paths), output_dir), indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        type=parse_adapter,
        metavar="NAME=PATH",
        help="PEFT adapter directory to evaluate; repeatable. The base model always runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: results/kazmmlu_full, or a new temporary directory with --dry-run",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="synthetic rehearsal: CPU, offline, no weights"
    )
    args = parser.parse_args(argv)
    names = [name for name, _ in args.adapter]
    if len(set(names)) != len(names):
        parser.error(f"duplicate adapter names: {names}")
    if args.dry_run:
        os.environ["HF_HUB_OFFLINE"] = "1"
        output = args.output or Path(tempfile.mkdtemp(prefix="kazmmlu_full_dry_"))
        dry_run(args.adapter, output)
    else:
        run(args.adapter, args.output or Path("results/kazmmlu_full"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
