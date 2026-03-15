"""Tests for evaluation/generate_report.py"""

import csv
import json

import pytest

from evaluation.generate_report import (
    _build_auto_metrics_section,
    _build_benchmark_section,
    _build_human_eval_section,
    _build_llm_judge_section,
    build_report,
    run,
)


# ============================================================
# _build_benchmark_section
# ============================================================

class TestBuildBenchmarkSection:
    def test_no_results(self, tmp_path):
        section = _build_benchmark_section(tmp_path)
        assert "not available" in section

    def test_from_json(self, tmp_path):
        data = {
            "models": ["base", "sft"],
            "tasks": [
                {
                    "task": "mmlu",
                    "alias": "mmlu",
                    "metric_name": "acc,none",
                    "level": "overall",
                    "scores": {"base": 0.75, "sft": 0.72},
                },
            ],
        }
        with open(tmp_path / "benchmark_comparison.json", "w") as f:
            json.dump(data, f)

        section = _build_benchmark_section(tmp_path)
        assert "75.00%" in section
        assert "72.00%" in section
        assert "mmlu" in section

    def test_from_markdown(self, tmp_path):
        md_content = "# Benchmark Report\n\nSome content here.\n"
        (tmp_path / "benchmark_comparison.md").write_text(md_content)

        section = _build_benchmark_section(tmp_path)
        assert "Some content here." in section
        assert "# Benchmark Report" not in section

    def test_markdown_preferred_over_json(self, tmp_path):
        (tmp_path / "benchmark_comparison.md").write_text("# Title\nMD content.")
        data = {"models": ["base"], "tasks": []}
        with open(tmp_path / "benchmark_comparison.json", "w") as f:
            json.dump(data, f)

        section = _build_benchmark_section(tmp_path)
        assert "MD content" in section

    def test_empty_json_data(self, tmp_path):
        data = {"models": [], "tasks": []}
        with open(tmp_path / "benchmark_comparison.json", "w") as f:
            json.dump(data, f)

        section = _build_benchmark_section(tmp_path)
        assert "empty" in section.lower()


# ============================================================
# _build_auto_metrics_section
# ============================================================

class TestBuildAutoMetricsSection:
    def test_no_results(self, tmp_path):
        section = _build_auto_metrics_section(tmp_path)
        assert "not available" in section

    def test_with_results(self, tmp_path):
        data = {
            "overall": {
                "base": {
                    "n_samples": 50,
                    "format_compliance": 0.95,
                    "degeneracy_rate": 0.02,
                    "distinct_1": 0.8,
                    "distinct_2": 0.9,
                    "keyword_satisfaction": 0.7,
                    "avg_length": 45.0,
                    "median_length": 42.0,
                },
            },
            "per_language": {},
        }
        with open(tmp_path / "auto_metrics.json", "w") as f:
            json.dump(data, f)

        section = _build_auto_metrics_section(tmp_path)
        assert "Format Compliance" in section
        assert "base" in section
        assert "95.0%" in section

    def test_with_per_language(self, tmp_path):
        data = {
            "overall": {
                "base": {"n_samples": 30, "format_compliance": 0.9,
                         "degeneracy_rate": 0.0, "distinct_1": 0.5,
                         "distinct_2": 0.6, "keyword_satisfaction": None,
                         "avg_length": 40.0, "median_length": 38.0},
            },
            "per_language": {
                "base": {
                    "en": {"n_samples": 20, "format_compliance": 0.95,
                           "degeneracy_rate": 0.0, "distinct_1": 0.55,
                           "distinct_2": 0.65, "keyword_satisfaction": None,
                           "avg_length": 42.0, "median_length": 40.0},
                },
            },
        }
        with open(tmp_path / "auto_metrics.json", "w") as f:
            json.dump(data, f)

        section = _build_auto_metrics_section(tmp_path)
        assert "Language: en" in section

    def test_empty_overall(self, tmp_path):
        data = {"overall": {}, "per_language": {}}
        with open(tmp_path / "auto_metrics.json", "w") as f:
            json.dump(data, f)

        section = _build_auto_metrics_section(tmp_path)
        assert "empty" in section.lower()


# ============================================================
# _build_llm_judge_section
# ============================================================

class TestBuildLlmJudgeSection:
    def test_no_results(self, tmp_path):
        section = _build_llm_judge_section(tmp_path)
        assert "not available" in section

    def test_with_results(self, tmp_path):
        data = [
            {
                "model_a": "base",
                "model_b": "grpo",
                "n_comparisons": 50,
                "win_base": 15,
                "win_grpo": 25,
                "tie": 10,
                "win_rate_base": 0.3,
                "win_rate_grpo": 0.5,
                "tie_rate": 0.2,
                "consistency_rate": 0.85,
            },
        ]
        with open(tmp_path / "llm_judge.json", "w") as f:
            json.dump(data, f)

        section = _build_llm_judge_section(tmp_path)
        assert "base vs grpo" in section
        assert "50 pairs" in section

    def test_empty_list(self, tmp_path):
        with open(tmp_path / "llm_judge.json", "w") as f:
            json.dump([], f)

        section = _build_llm_judge_section(tmp_path)
        assert "empty" in section.lower()


# ============================================================
# _build_human_eval_section
# ============================================================

class TestBuildHumanEvalSection:
    def test_no_csv(self, tmp_path):
        section = _build_human_eval_section(tmp_path)
        assert "not exported" in section

    def test_csv_unfilled(self, tmp_path):
        with open(tmp_path / "human_eval_samples.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "headline", "lang",
                                                     "response_a", "response_b", "your_verdict"])
            writer.writeheader()
            writer.writerow({"id": 1, "headline": "h", "lang": "en",
                             "response_a": "a", "response_b": "b", "your_verdict": ""})
            writer.writerow({"id": 2, "headline": "h", "lang": "en",
                             "response_a": "a", "response_b": "b", "your_verdict": ""})

        section = _build_human_eval_section(tmp_path)
        assert "Pending" in section
        assert "2 pairs" in section

    def test_csv_filled_with_key(self, tmp_path):
        with open(tmp_path / "human_eval_samples.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "headline", "lang",
                                                     "response_a", "response_b", "your_verdict"])
            writer.writeheader()
            writer.writerow({"id": 1, "headline": "h", "lang": "en",
                             "response_a": "a", "response_b": "b", "your_verdict": "A"})
            writer.writerow({"id": 2, "headline": "h", "lang": "en",
                             "response_a": "a", "response_b": "b", "your_verdict": "B"})
            writer.writerow({"id": 3, "headline": "h", "lang": "en",
                             "response_a": "a", "response_b": "b", "your_verdict": "TIE"})

        answer_key = [
            {"id": 1, "response_a_source": "base", "response_b_source": "grpo"},
            {"id": 2, "response_a_source": "grpo", "response_b_source": "base"},
            {"id": 3, "response_a_source": "base", "response_b_source": "grpo"},
        ]
        with open(tmp_path / "human_eval_answer_key.json", "w") as f:
            json.dump(answer_key, f)

        section = _build_human_eval_section(tmp_path)
        assert "3/3" in section
        assert "Completed" in section

    def test_csv_partially_filled(self, tmp_path):
        with open(tmp_path / "human_eval_samples.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "headline", "lang",
                                                     "response_a", "response_b", "your_verdict"])
            writer.writeheader()
            writer.writerow({"id": 1, "headline": "h", "lang": "en",
                             "response_a": "a", "response_b": "b", "your_verdict": "A"})
            writer.writerow({"id": 2, "headline": "h", "lang": "en",
                             "response_a": "a", "response_b": "b", "your_verdict": ""})

        answer_key = [
            {"id": 1, "response_a_source": "base", "response_b_source": "grpo"},
            {"id": 2, "response_a_source": "base", "response_b_source": "grpo"},
        ]
        with open(tmp_path / "human_eval_answer_key.json", "w") as f:
            json.dump(answer_key, f)

        section = _build_human_eval_section(tmp_path)
        assert "1/2" in section


# ============================================================
# build_report (full integration)
# ============================================================

class TestBuildReport:
    def test_empty_results_dir(self, tmp_path):
        report = build_report(tmp_path)
        assert "# Humor Generation" in report
        assert "not available" in report

    def test_full_report_with_all_sections(self, tmp_path):
        bench_data = {
            "models": ["base", "sft"],
            "tasks": [{"task": "mmlu", "alias": "mmlu", "metric_name": "acc,none",
                        "level": "overall", "scores": {"base": 0.75, "sft": 0.72}}],
        }
        with open(tmp_path / "benchmark_comparison.json", "w") as f:
            json.dump(bench_data, f)

        auto_data = {
            "overall": {"base": {"n_samples": 50, "format_compliance": 0.95,
                                  "degeneracy_rate": 0.02, "distinct_1": 0.8,
                                  "distinct_2": 0.9, "keyword_satisfaction": None,
                                  "avg_length": 45.0, "median_length": 42.0}},
            "per_language": {},
        }
        with open(tmp_path / "auto_metrics.json", "w") as f:
            json.dump(auto_data, f)

        judge_data = [{
            "model_a": "base", "model_b": "sft",
            "n_comparisons": 30, "win_base": 10, "win_sft": 15, "tie": 5,
            "win_rate_base": 0.33, "win_rate_sft": 0.5, "tie_rate": 0.17,
            "consistency_rate": 0.8,
        }]
        with open(tmp_path / "llm_judge.json", "w") as f:
            json.dump(judge_data, f)

        report = build_report(tmp_path)
        assert "## 1." in report
        assert "## 2." in report
        assert "## 3." in report
        assert "## 4." in report
        assert "Appendix" in report


# ============================================================
# run() programmatic entry point
# ============================================================

class TestRun:
    def test_run_creates_report_file(self, tmp_path):
        report = run(results_dir=tmp_path)
        assert isinstance(report, str)
        assert (tmp_path / "evaluation_report.md").exists()
        content = (tmp_path / "evaluation_report.md").read_text()
        assert "# Humor Generation" in content
