"""Tests for pipeline ordering and SFT reward-exclusion wiring."""

from types import SimpleNamespace

from data_preprocessing import pipeline


def test_all_stage_runs_reward_before_sft(monkeypatch):
    call_order = []

    monkeypatch.setattr(pipeline, "run_parse", lambda: call_order.append("parse"))
    monkeypatch.setattr(pipeline, "run_synthesize", lambda **kwargs: call_order.append("synthesize"))
    monkeypatch.setattr(pipeline, "run_format_reward", lambda: call_order.append("format_reward"))
    monkeypatch.setattr(pipeline, "run_format_sft", lambda: call_order.append("format_sft"))
    monkeypatch.setattr(
        pipeline,
        "run_format_grpo",
        lambda eval_ratio, seed: call_order.append("format_grpo"),
    )

    monkeypatch.setattr(
        "argparse.ArgumentParser.parse_args",
        lambda self: SimpleNamespace(
            stage="all",
            n_headline=200,
            n_keyword=100,
            eval_ratio=0.2,
            seed=42,
        ),
    )

    pipeline.main()
    assert call_order == ["parse", "format_reward", "format_sft", "format_grpo"]


def test_sft_reads_reward_files_when_available(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        pipeline,
        "_load_unified_datasets",
        lambda: {"dummy": "unified"},
    )

    def _fake_format_sft(unified_datasets, synthesized_dir=None, **kwargs):
        calls["exclude_texts"] = kwargs.get("exclude_texts")
        return {"train": [], "validation": []}

    monkeypatch.setattr(pipeline, "format_sft", _fake_format_sft)
    monkeypatch.setattr(
        pipeline,
        "_load_reward_used_texts",
        lambda: {"text-a", "text-b"},
    )

    class DummyDir:
        def mkdir(self, parents=False, exist_ok=False):
            return None

        def __truediv__(self, _):
            return DummyFile()

    class DummyFile:
        def __str__(self):
            return "dummy.jsonl"

    monkeypatch.setattr(pipeline, "SFT_DIR", DummyDir())
    monkeypatch.setattr(pipeline, "SYNTHESIZED_DIR", "dummy-synth")

    class DummySplit:
        def __init__(self):
            self.saved = 0

        def to_json(self, _):
            self.saved += 1

        def __len__(self):
            return 0

    monkeypatch.setattr(
        pipeline,
        "format_sft",
        lambda unified_datasets, synthesized_dir=None, **kwargs: {
            "train": DummySplit(),
            "validation": DummySplit(),
        },
    )

    # capture kwargs passed to format_sft by wrapping after replacement
    seen = {}

    def _capture_format_sft(unified_datasets, synthesized_dir=None, **kwargs):
        seen["exclude_texts"] = kwargs.get("exclude_texts")
        return {"train": DummySplit(), "validation": DummySplit()}

    monkeypatch.setattr(pipeline, "format_sft", _capture_format_sft)
    pipeline.run_format_sft()
    assert seen["exclude_texts"] == {"text-a", "text-b"}
