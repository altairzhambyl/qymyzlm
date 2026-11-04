"""CPU-safe tests for scripts/eval_kazmmlu_full.py: no downloads, GPU or real weights."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_kazmmlu_full.py"


@pytest.fixture(scope="module")
def full():
    spec = importlib.util.spec_from_file_location("eval_kazmmlu_full", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _no_network(module, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("must not touch the hub or load real weights")

    monkeypatch.setattr(module.bb, "load_dataset", fail)
    monkeypatch.setattr(module.bb, "get_dataset_config_names", fail)
    monkeypatch.setattr(module, "_load_model", fail)
    monkeypatch.setattr(module, "_load_tokenizer", fail)


def _adapter_dir(path: Path, base: str = "Qwen/Qwen3-0.6B-Base") -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    (path / "adapter_model.safetensors").write_bytes(b"not real weights")
    return path


def test_dry_run_writes_all_outputs_offline(full, tmp_path, monkeypatch) -> None:
    _no_network(full, monkeypatch)
    assert full.main(["--dry-run", "--output", str(tmp_path)]) == 0

    summary = json.loads((tmp_path / "base.summary.json").read_text(encoding="utf-8"))
    assert summary["n_questions"] == 20
    assert summary["metadata"] == {"dry_run": True}
    assert set(summary["per_subject"]) == {
        "Biology (High School in kaz)",
        "Math (High School in kaz)",
    }
    assert summary["per_subject"]["Biology (High School in kaz)"]["n"] == 10
    assert summary["shot_stats"]["min_shots_used"] == 3
    lines = (tmp_path / "dry_adapter.questions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    assert {"config", "pred", "gold", "shots_used"} <= set(json.loads(lines[0]))
    assert set(json.loads((tmp_path / "comparison.json").read_text())) == {"dry_adapter"}


def test_dry_run_verifies_adapter_directories(full, tmp_path, monkeypatch) -> None:
    _no_network(full, monkeypatch)
    adapter = _adapter_dir(tmp_path / "adapter")
    out = tmp_path / "out"
    assert full.main(["--dry-run", "--output", str(out), "--adapter", f"mine={adapter}"]) == 0
    summary = json.loads((out / "mine.summary.json").read_text(encoding="utf-8"))
    assert len(summary["adapter_sha256"]) == 64


def test_real_run_refuses_cpu(full, tmp_path, monkeypatch) -> None:
    _no_network(full, monkeypatch)
    monkeypatch.setattr(full.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        full.main(["--output", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def test_dry_and_real_output_never_mix(full, tmp_path, monkeypatch) -> None:
    _no_network(full, monkeypatch)
    monkeypatch.setattr(full.torch.cuda, "is_available", lambda: True)
    assert full.main(["--dry-run", "--output", str(tmp_path)]) == 0
    with pytest.raises(RuntimeError, match="dry-run output"):
        full.main(["--output", str(tmp_path)])

    real = tmp_path / "real"
    real.mkdir()
    (real / "base.summary.json").write_text(json.dumps({"name": "base", "metadata": {}}))
    with pytest.raises(RuntimeError, match="not overwriting"):
        full.main(["--dry-run", "--output", str(real)])


def test_partial_run_is_refused(full, tmp_path) -> None:
    test, dev = full._dry_run_rows()
    with pytest.raises(RuntimeError, match="scored 20 questions, expected 9870"):
        full.evaluate_and_save(
            "base",
            full._DryRunModel(0),
            full._DryRunTokenizer(),
            test,
            dev,
            tmp_path,
            None,
            {},
        )
    assert not (tmp_path / "base.summary.json").exists()


def test_check_adapter_rejects_bad_directories(full, tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        full.check_adapter("x", tmp_path)
    wrong_base = _adapter_dir(tmp_path / "wrong", base="Qwen/Qwen2.5-0.5B")
    with pytest.raises(ValueError, match="base model"):
        full.check_adapter("x", wrong_base)
    good = _adapter_dir(tmp_path / "good")
    with pytest.raises(ValueError, match="sha256"):
        full.check_adapter("s1_step2500", good)  # known name, wrong bytes


def test_parse_adapter(full) -> None:
    assert full.parse_adapter("s1=/a/b=c") == ("s1", Path("/a/b=c"))
    for bad in ("no-equals", "=/path", "name=", "base=/path", "../x=/path"):
        with pytest.raises(argparse.ArgumentTypeError):
            full.parse_adapter(bad)


def test_duplicate_adapter_names_rejected(full, capsys) -> None:
    with pytest.raises(SystemExit):
        full.main(["--dry-run", "--adapter", "a=/x", "--adapter", "a=/y"])
    assert "duplicate" in capsys.readouterr().err


def test_wilson_matches_documented_intervals(full) -> None:
    def pct(correct: int, total: int) -> list[float]:
        return [round(100 * bound, 1) for bound in full.wilson_interval(correct, total)]

    assert pct(63, 200) == [25.5, 38.2]  # 31.5%, n=200
    assert pct(71, 200) == [29.2, 42.3]  # 35.5%, n=200


def test_compare_counts_discordant_pairs(full) -> None:
    def question(pred: str) -> dict:
        return {"config": "c", "gold": "A", "pred": pred}

    base = [question("A"), question("B"), question("B"), question("A")]
    other = [question("B"), question("A"), question("A"), question("A")]
    assert full.compare_to_base(base, other) == {
        "delta_pp": 25.0,
        "base_only_correct": 1,
        "other_only_correct": 2,
    }
    with pytest.raises(ValueError, match="aligned"):
        full.compare_to_base(base, other[:-1])
