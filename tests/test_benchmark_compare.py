"""Tests for evaluation/benchmark_compare.py"""

import json

import pytest

from evaluation.benchmark_compare import (
    TaskResult,
    _fmt_pct,
    _fmt_pp,
    _pick_primary_metric,
    build_comparison_summary,
    build_comparison_table,
    build_full_markdown,
    compare_results,
    find_latest_result_json,
)


# ============================================================
# TaskResult
# ============================================================

class TestTaskResult:
    def test_level_overall(self):
        r = TaskResult(task="mmlu", alias="mmlu", metric_name="acc,none")
        assert r.level == "overall"

    def test_level_group(self):
        r = TaskResult(task="mmlu_stem", alias=" - stem", metric_name="acc,none")
        assert r.level == "group"

    def test_level_subtask(self):
        r = TaskResult(task="mmlu_physics", alias="  - physics", metric_name="acc,none")
        assert r.level == "subtask"

    def test_level_no_prefix(self):
        r = TaskResult(task="arc", alias="arc_challenge", metric_name="acc,none")
        assert r.level == "overall"


# ============================================================
# _pick_primary_metric
# ============================================================

class TestPickPrimaryMetric:
    def test_prefers_acc(self):
        task_result = {
            "acc,none": 0.75,
            "acc_stderr,none": 0.02,
            "f1,none": 0.80,
            "alias": "test",
        }
        name, val = _pick_primary_metric(task_result)
        assert name == "acc,none"
        assert val == pytest.approx(0.75)

    def test_falls_back_to_exact_match(self):
        task_result = {
            "exact_match,none": 0.65,
            "exact_match_stderr,none": 0.01,
            "alias": "test",
        }
        name, val = _pick_primary_metric(task_result)
        assert name == "exact_match,none"
        assert val == pytest.approx(0.65)

    def test_falls_back_to_first_numeric(self):
        task_result = {
            "some_custom_metric": 0.42,
            "some_custom_metric_stderr": 0.03,
            "alias": "test",
        }
        name, val = _pick_primary_metric(task_result)
        assert name == "some_custom_metric"
        assert val == pytest.approx(0.42)

    def test_raises_on_no_numeric(self):
        task_result = {"alias": "test", "note": "no numbers here"}
        with pytest.raises(ValueError, match="No numeric metric"):
            _pick_primary_metric(task_result)

    def test_ignores_stderr(self):
        task_result = {
            "acc_stderr,none": 0.02,
            "bleu,none": 0.55,
        }
        name, val = _pick_primary_metric(task_result)
        assert "stderr" not in name
        assert val == pytest.approx(0.55)


# ============================================================
# find_latest_result_json
# ============================================================

class TestFindLatestResultJson:
    def test_finds_latest(self, tmp_path):
        d = tmp_path / "model_x"
        d.mkdir()
        p1 = d / "results_2024-01-01.json"
        p1.write_text("{}")
        p2 = d / "results_2024-06-01.json"
        p2.write_text("{}")
        result = find_latest_result_json(tmp_path)
        assert result == p2

    def test_returns_none_when_empty(self, tmp_path):
        assert find_latest_result_json(tmp_path) is None

    def test_finds_in_nested_dir(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        nested.mkdir(parents=True)
        p = nested / "results_2024-03-15.json"
        p.write_text("{}")
        assert find_latest_result_json(tmp_path) == p


# ============================================================
# compare_results
# ============================================================

def _make_lm_eval_json(tasks: dict[str, dict]) -> dict:
    """Build a minimal lm-eval results JSON structure."""
    return {"results": tasks}


class TestCompareResults:
    def _write_result(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def test_basic_comparison(self, tmp_path):
        base_data = _make_lm_eval_json({
            "mmlu": {"acc,none": 0.75, "alias": "mmlu"},
        })
        sft_data = _make_lm_eval_json({
            "mmlu": {"acc,none": 0.72, "alias": "mmlu"},
        })
        base_path = tmp_path / "base.json"
        sft_path = tmp_path / "sft.json"
        self._write_result(base_path, base_data)
        self._write_result(sft_path, sft_data)

        comparisons = compare_results({"base": base_path, "sft": sft_path})
        assert len(comparisons) == 1
        assert comparisons[0].scores["base"] == pytest.approx(0.75)
        assert comparisons[0].scores["sft"] == pytest.approx(0.72)

    def test_skips_none_paths(self, tmp_path):
        base_data = _make_lm_eval_json({"mmlu": {"acc,none": 0.75, "alias": "mmlu"}})
        base_path = tmp_path / "base.json"
        self._write_result(base_path, base_data)

        comparisons = compare_results({"base": base_path, "sft": None})
        assert comparisons == []

    def test_only_common_tasks(self, tmp_path):
        base_data = _make_lm_eval_json({
            "mmlu": {"acc,none": 0.75, "alias": "mmlu"},
            "arc": {"acc,none": 0.60, "alias": "arc"},
        })
        sft_data = _make_lm_eval_json({
            "mmlu": {"acc,none": 0.72, "alias": "mmlu"},
        })
        base_path = tmp_path / "base.json"
        sft_path = tmp_path / "sft.json"
        self._write_result(base_path, base_data)
        self._write_result(sft_path, sft_data)

        comparisons = compare_results({"base": base_path, "sft": sft_path})
        task_names = [c.task for c in comparisons]
        assert "mmlu" in task_names
        assert "arc" not in task_names

    def test_three_models(self, tmp_path):
        tasks = {"mmlu": {"acc,none": 0.75, "alias": "mmlu"}}
        for name, score in [("base", 0.75), ("sft", 0.72), ("grpo", 0.73)]:
            data = _make_lm_eval_json({"mmlu": {"acc,none": score, "alias": "mmlu"}})
            self._write_result(tmp_path / f"{name}.json", data)

        model_paths = {
            name: tmp_path / f"{name}.json"
            for name in ["base", "sft", "grpo"]
        }
        comparisons = compare_results(model_paths)
        assert len(comparisons) == 1
        assert "grpo" in comparisons[0].scores


# ============================================================
# Formatting Functions
# ============================================================

class TestFormatting:
    def test_fmt_pct(self):
        assert _fmt_pct(0.7512) == "75.12%"
        assert _fmt_pct(1.0) == "100.00%"

    def test_fmt_pp_positive(self):
        assert _fmt_pp(0.03) == "+3.00pp"

    def test_fmt_pp_negative(self):
        assert _fmt_pp(-0.02) == "-2.00pp"

    def test_fmt_pp_zero(self):
        assert _fmt_pp(0.0) == "0.00pp"


class TestBuildComparisonTable:
    def test_basic_table(self):
        rows = [
            TaskResult("mmlu", "mmlu", "acc,none", {"base": 0.75, "sft": 0.72}),
        ]
        table = build_comparison_table(rows, ["base", "sft"])
        assert "mmlu" in table
        assert "75.00%" in table
        assert "72.00%" in table

    def test_empty_rows(self):
        table = build_comparison_table([], ["base", "sft"])
        assert table == ""

    def test_with_reference_model(self):
        rows = [
            TaskResult("mmlu", "mmlu", "acc,none", {"base": 0.75, "sft": 0.72}),
        ]
        table = build_comparison_table(rows, ["base", "sft"], reference_model="base")
        assert "Delta" in table
        assert "-3.00pp" in table


class TestBuildComparisonSummary:
    def test_summary_content(self):
        comparisons = [
            TaskResult("t1", "task1", "acc,none", {"base": 0.75, "sft": 0.80}),
            TaskResult("t2", "task2", "acc,none", {"base": 0.60, "sft": 0.55}),
        ]
        summary = build_comparison_summary(comparisons, ["base", "sft"])
        assert "Reference model: **base**" in summary
        assert "1 improved" in summary
        assert "1 degraded" in summary

    def test_empty_comparisons(self):
        assert build_comparison_summary([], ["base", "sft"]) == ""

    def test_single_model(self):
        comparisons = [TaskResult("t1", "t1", "acc,none", {"base": 0.75})]
        assert build_comparison_summary(comparisons, ["base"]) == ""


class TestBuildFullMarkdown:
    def test_contains_sections(self):
        comparisons = [
            TaskResult("mmlu", "mmlu", "acc,none", {"base": 0.75, "sft": 0.72}),
            TaskResult("mmlu_stem", " - stem", "acc,none", {"base": 0.70, "sft": 0.69}),
            TaskResult("mmlu_phys", "  - physics", "acc,none", {"base": 0.65, "sft": 0.60}),
        ]
        md = build_full_markdown(comparisons, ["base", "sft"])
        assert "# Benchmark Comparison Report" in md
        assert "Overall Tasks" in md
        assert "Task Groups" in md

    def test_top_k_sections(self):
        subtasks = [
            TaskResult(f"t{i}", f"  - subtask{i}", "acc,none",
                       {"base": 0.5 + i * 0.01, "sft": 0.5 - i * 0.01})
            for i in range(20)
        ]
        md = build_full_markdown(subtasks, ["base", "sft"], top_k=5)
        assert "Top 5" in md
