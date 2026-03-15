"""Tests for evaluation/pipeline.py"""

import pytest

from evaluation.pipeline import ALL_STEPS, _resolve_steps


# ============================================================
# _resolve_steps
# ============================================================

class TestResolveSteps:
    def test_all(self):
        result = _resolve_steps("all", None)
        assert result == ALL_STEPS

    def test_all_case_insensitive(self):
        assert _resolve_steps("ALL", None) == ALL_STEPS
        assert _resolve_steps("All", None) == ALL_STEPS

    def test_single_step(self):
        assert _resolve_steps("benchmark", None) == ["benchmark"]

    def test_multiple_steps(self):
        result = _resolve_steps("benchmark,generate,report", None)
        assert result == ["benchmark", "generate", "report"]

    def test_steps_with_whitespace(self):
        result = _resolve_steps("  benchmark , generate , report  ", None)
        assert result == ["benchmark", "generate", "report"]

    def test_skip_single(self):
        result = _resolve_steps("all", "benchmark")
        assert "benchmark" not in result
        assert "generate" in result
        assert "report" in result

    def test_skip_multiple(self):
        result = _resolve_steps("all", "benchmark,llm_judge")
        assert "benchmark" not in result
        assert "llm_judge" not in result
        assert "generate" in result
        assert "report" in result

    def test_skip_with_whitespace(self):
        result = _resolve_steps("all", " benchmark , llm_judge ")
        assert "benchmark" not in result
        assert "llm_judge" not in result

    def test_invalid_step_raises(self):
        with pytest.raises(ValueError, match="Unknown step"):
            _resolve_steps("nonexistent_step", None)

    def test_invalid_among_valid_raises(self):
        with pytest.raises(ValueError, match="Unknown step"):
            _resolve_steps("benchmark,invalid_step,report", None)

    def test_skip_no_effect_on_absent_step(self):
        result = _resolve_steps("benchmark,report", "generate")
        assert result == ["benchmark", "report"]

    def test_skip_all_yields_empty(self):
        result = _resolve_steps("benchmark", "benchmark")
        assert result == []

    def test_order_preserved(self):
        result = _resolve_steps("report,benchmark,generate", None)
        assert result == ["report", "benchmark", "generate"]

    def test_empty_steps_string(self):
        result = _resolve_steps("", None)
        assert result == []

    def test_all_with_skip_none(self):
        result = _resolve_steps("all", None)
        assert len(result) == len(ALL_STEPS)


# ============================================================
# ALL_STEPS constant
# ============================================================

class TestAllSteps:
    def test_contains_expected_steps(self):
        expected = {"benchmark", "generate", "auto_metrics", "llm_judge", "human_eval", "report"}
        assert set(ALL_STEPS) == expected

    def test_report_is_last(self):
        assert ALL_STEPS[-1] == "report"

    def test_benchmark_is_first(self):
        assert ALL_STEPS[0] == "benchmark"
