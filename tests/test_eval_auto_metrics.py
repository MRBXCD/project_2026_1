"""Tests for evaluation/eval_auto_metrics.py"""

import json

import pytest

from evaluation.eval_auto_metrics import (
    _distinct_n,
    _is_degenerate,
    _is_format_compliant,
    _keyword_satisfied,
    compute_metrics,
    compute_per_lang_metrics,
    run,
)


# ============================================================
# _is_format_compliant
# ============================================================

class TestIsFormatCompliant:
    def test_empty_string(self):
        assert _is_format_compliant("") is False

    def test_too_short(self):
        assert _is_format_compliant("short") is False

    def test_too_long(self):
        text = "a " * 200
        assert _is_format_compliant(text) is False

    def test_normal_text(self):
        assert _is_format_compliant("This is a perfectly normal joke response.") is True

    def test_degenerate_text(self):
        assert _is_format_compliant("ha ha ha ha ha ha ha ha ha ha ha ha") is False

    def test_boundary_min_length(self):
        assert _is_format_compliant("a" * 9) is False
        assert _is_format_compliant("a" * 10) is True

    def test_boundary_max_length(self):
        assert _is_format_compliant("a" * 280) is True
        assert _is_format_compliant("a" * 281) is False

    def test_short_word_count_skips_trigram_check(self):
        assert _is_format_compliant("ab ab ab ab") is True


# ============================================================
# _is_degenerate
# ============================================================

class TestIsDegenerate:
    def test_normal_text(self):
        assert _is_degenerate("This is a perfectly normal joke.") is False

    def test_repetitive_text(self):
        assert _is_degenerate("ha ha ha ha ha ha ha ha ha ha") is True

    def test_short_text_not_degenerate(self):
        assert _is_degenerate("ha ha ha") is False


# ============================================================
# _keyword_satisfied
# ============================================================

class TestKeywordSatisfied:
    def test_all_present(self):
        assert _keyword_satisfied("The penguin filed for bankruptcy.", ["penguin", "bankruptcy"]) is True

    def test_partial_present(self):
        assert _keyword_satisfied("The penguin walked away.", ["penguin", "bankruptcy"]) is False

    def test_none_present(self):
        assert _keyword_satisfied("A random sentence here.", ["penguin", "bankruptcy"]) is False

    def test_case_insensitive(self):
        assert _keyword_satisfied("PENGUIN and BANKRUPTCY.", ["penguin", "bankruptcy"]) is True

    def test_empty_keywords(self):
        assert _keyword_satisfied("any text", []) is True


# ============================================================
# _distinct_n
# ============================================================

class TestDistinctN:
    def test_all_unique(self):
        responses = ["alpha beta gamma", "delta epsilon zeta"]
        assert _distinct_n(responses, 1) == pytest.approx(1.0)

    def test_all_same(self):
        responses = ["ha ha ha", "ha ha ha"]
        result = _distinct_n(responses, 1)
        assert result == pytest.approx(1 / 6)

    def test_distinct_2_unique_bigrams(self):
        responses = ["a b c d e"]
        result = _distinct_n(responses, 2)
        assert result == pytest.approx(1.0)

    def test_empty_responses(self):
        assert _distinct_n([], 1) == 0.0

    def test_single_word_response(self):
        assert _distinct_n(["hello"], 2) == 0.0


# ============================================================
# compute_metrics
# ============================================================

def _make_result(response: str, keywords: list[str] | None = None, lang: str = "en") -> dict:
    return {
        "best_response": response,
        "keywords": keywords or [],
        "lang": lang,
    }


class TestComputeMetrics:
    def test_empty_results(self):
        assert compute_metrics([]) == {}

    def test_basic_metrics(self):
        results = [
            _make_result("This is a perfectly normal joke about things."),
            _make_result("Another good joke that is different enough here."),
        ]
        metrics = compute_metrics(results)
        assert metrics["n_samples"] == 2
        assert 0.0 <= metrics["format_compliance"] <= 1.0
        assert 0.0 <= metrics["degeneracy_rate"] <= 1.0
        assert metrics["distinct_1"] > 0
        assert metrics["distinct_2"] > 0
        assert metrics["avg_length"] > 0
        assert metrics["keyword_satisfaction"] is None
        assert metrics["keyword_total"] == 0

    def test_keyword_metrics(self):
        results = [
            _make_result("The penguin filed for bankruptcy.", ["penguin", "bankruptcy"]),
            _make_result("A random joke without keywords.", ["penguin", "bankruptcy"]),
        ]
        metrics = compute_metrics(results)
        assert metrics["keyword_satisfaction"] == pytest.approx(0.5)
        assert metrics["keyword_total"] == 2

    def test_mixed_keyword_and_no_keyword(self):
        results = [
            _make_result("The penguin filed for bankruptcy.", ["penguin", "bankruptcy"]),
            _make_result("Just a regular joke without constraints."),
        ]
        metrics = compute_metrics(results)
        assert metrics["keyword_total"] == 1
        assert metrics["keyword_satisfaction"] == pytest.approx(1.0)


# ============================================================
# compute_per_lang_metrics
# ============================================================

class TestComputePerLangMetrics:
    def test_separates_by_language(self):
        results = [
            _make_result("An english joke about something funny.", lang="en"),
            _make_result("Un chiste gracioso sobre algo divertido.", lang="es"),
            _make_result("An english joke about something funny.", lang="en"),
        ]
        per_lang = compute_per_lang_metrics(results)
        assert "en" in per_lang
        assert "es" in per_lang
        assert per_lang["en"]["n_samples"] == 2
        assert per_lang["es"]["n_samples"] == 1


# ============================================================
# run() programmatic entry point
# ============================================================

class TestRun:
    def test_run_with_output_files(self, tmp_path):
        output_dir = tmp_path / "outputs"
        results_dir = tmp_path / "results"
        output_dir.mkdir()

        data = [
            {"best_response": "A funny joke about penguin and bankruptcy.", "keywords": ["penguin"], "lang": "en"},
            {"best_response": "Another joke that is different and unique.", "keywords": [], "lang": "en"},
        ]
        with open(output_dir / "base.jsonl", "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

        result = run(models=["base"], output_dir=output_dir, results_dir=results_dir)
        assert result is not None
        assert "overall" in result
        assert "base" in result["overall"]
        assert (results_dir / "auto_metrics.json").exists()

    def test_run_missing_model_returns_partial(self, tmp_path):
        output_dir = tmp_path / "outputs"
        results_dir = tmp_path / "results"
        output_dir.mkdir()

        data = [{"best_response": "A funny joke here.", "keywords": [], "lang": "en"}]
        with open(output_dir / "base.jsonl", "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

        result = run(models=["base", "sft"], output_dir=output_dir, results_dir=results_dir)
        assert result is not None
        assert "base" in result["overall"]
        assert "sft" not in result["overall"]

    def test_run_no_outputs_returns_none(self, tmp_path):
        output_dir = tmp_path / "outputs"
        results_dir = tmp_path / "results"
        output_dir.mkdir()

        result = run(models=["base"], output_dir=output_dir, results_dir=results_dir)
        assert result is None
