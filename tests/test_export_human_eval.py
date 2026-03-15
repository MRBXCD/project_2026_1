"""Tests for evaluation/export_human_eval.py"""

import json

import pytest

from evaluation.export_human_eval import export_blind_table, run


# ============================================================
# Helpers
# ============================================================

def _make_results(n: int, model: str, lang: str = "en") -> list[dict]:
    return [
        {
            "headline": f"Headline {i}",
            "lang": lang,
            "best_response": f"{model} response {i}",
        }
        for i in range(n)
    ]


# ============================================================
# export_blind_table
# ============================================================

class TestExportBlindTable:
    def test_basic_output_shape(self):
        results_a = _make_results(10, "base")
        results_b = _make_results(10, "grpo")
        rows, key = export_blind_table(results_a, results_b, "base", "grpo", n_samples=5)
        assert len(rows) == 5
        assert len(key) == 5

    def test_row_fields(self):
        results_a = _make_results(5, "base")
        results_b = _make_results(5, "grpo")
        rows, key = export_blind_table(results_a, results_b, "base", "grpo", n_samples=3)

        for row in rows:
            assert "id" in row
            assert "headline" in row
            assert "lang" in row
            assert "response_a" in row
            assert "response_b" in row
            assert "your_verdict" in row
            assert row["your_verdict"] == ""

    def test_answer_key_fields(self):
        results_a = _make_results(5, "base")
        results_b = _make_results(5, "grpo")
        rows, key = export_blind_table(results_a, results_b, "base", "grpo", n_samples=3)

        for entry in key:
            assert "id" in entry
            assert "response_a_source" in entry
            assert "response_b_source" in entry
            assert entry["response_a_source"] in ("base", "grpo")
            assert entry["response_b_source"] in ("base", "grpo")
            assert entry["response_a_source"] != entry["response_b_source"]

    def test_randomization_with_seed(self):
        results_a = _make_results(20, "base")
        results_b = _make_results(20, "grpo")
        rows1, _ = export_blind_table(results_a, results_b, "base", "grpo", n_samples=10, seed=42)
        rows2, _ = export_blind_table(results_a, results_b, "base", "grpo", n_samples=10, seed=42)
        assert rows1 == rows2

    def test_different_seeds_differ(self):
        results_a = _make_results(20, "base")
        results_b = _make_results(20, "grpo")
        rows1, _ = export_blind_table(results_a, results_b, "base", "grpo", n_samples=10, seed=42)
        rows2, _ = export_blind_table(results_a, results_b, "base", "grpo", n_samples=10, seed=99)
        assert rows1 != rows2

    def test_stratified_sampling(self):
        results_a = _make_results(10, "base", "en") + _make_results(10, "base", "zh")
        results_b = _make_results(10, "grpo", "en") + _make_results(10, "grpo", "zh")
        rows, _ = export_blind_table(results_a, results_b, "base", "grpo", n_samples=10)

        langs = [row["lang"] for row in rows]
        assert "en" in langs
        assert "zh" in langs

    def test_n_samples_capped_by_data(self):
        results_a = _make_results(3, "base")
        results_b = _make_results(3, "grpo")
        rows, _ = export_blind_table(results_a, results_b, "base", "grpo", n_samples=100)
        assert len(rows) <= 3

    def test_unequal_results_uses_min(self):
        results_a = _make_results(10, "base")
        results_b = _make_results(5, "grpo")
        rows, _ = export_blind_table(results_a, results_b, "base", "grpo", n_samples=8)
        assert len(rows) <= 5


# ============================================================
# run() programmatic entry point
# ============================================================

class TestRun:
    def _setup_outputs(self, tmp_path, n=10):
        output_dir = tmp_path / "outputs"
        results_dir = tmp_path / "results"
        output_dir.mkdir()
        for model in ["base", "grpo"]:
            data = _make_results(n, model)
            with open(output_dir / f"{model}.jsonl", "w") as f:
                for d in data:
                    f.write(json.dumps(d) + "\n")
        return output_dir, results_dir

    def test_run_exports_csv_and_key(self, tmp_path):
        output_dir, results_dir = self._setup_outputs(tmp_path)
        result = run(
            pair=("base", "grpo"), n_samples=5,
            output_dir=output_dir, results_dir=results_dir,
        )
        assert result is not None
        assert result["n_exported"] == 5
        assert (results_dir / "human_eval_samples.csv").exists()
        assert (results_dir / "human_eval_answer_key.json").exists()

    def test_run_missing_model_raises(self, tmp_path):
        output_dir = tmp_path / "outputs"
        results_dir = tmp_path / "results"
        output_dir.mkdir()

        data = _make_results(5, "base")
        with open(output_dir / "base.jsonl", "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

        with pytest.raises(FileNotFoundError):
            run(pair=("base", "grpo"), output_dir=output_dir, results_dir=results_dir)
