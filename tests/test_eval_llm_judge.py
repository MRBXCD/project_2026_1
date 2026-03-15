"""Tests for evaluation/eval_llm_judge.py"""

import pytest

from evaluation.eval_llm_judge import _parse_verdict, judge_pair


# ============================================================
# _parse_verdict
# ============================================================

class TestParseVerdict:
    def test_exact_a(self):
        assert _parse_verdict("A") == "A"

    def test_exact_b(self):
        assert _parse_verdict("B") == "B"

    def test_exact_tie(self):
        assert _parse_verdict("TIE") == "TIE"

    def test_lowercase_a(self):
        assert _parse_verdict("a") == "A"

    def test_lowercase_b(self):
        assert _parse_verdict("b") == "B"

    def test_lowercase_tie(self):
        assert _parse_verdict("tie") == "TIE"

    def test_tie_in_sentence(self):
        assert _parse_verdict("I think it's a TIE between them") == "TIE"

    def test_a_in_sentence(self):
        assert _parse_verdict("Response A is funnier") == "A"

    def test_b_in_sentence(self):
        assert _parse_verdict("I prefer B") == "B"

    def test_none_input(self):
        assert _parse_verdict(None) == "TIE"

    def test_empty_string(self):
        assert _parse_verdict("") == "TIE"

    def test_whitespace_a(self):
        assert _parse_verdict("  A  ") == "A"

    def test_garbage_defaults_to_tie(self):
        assert _parse_verdict("something completely random 123") == "TIE"


# ============================================================
# judge_pair (with mocked API)
# ============================================================

class TestJudgePair:
    def test_consistent_a_wins(self, monkeypatch):
        call_count = [0]
        def mock_call(client, prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return "A"
            return "B"

        monkeypatch.setattr("evaluation.eval_llm_judge._call_gemini", mock_call)
        monkeypatch.setattr("evaluation.eval_llm_judge.API_CALL_DELAY", 0)

        result = judge_pair(None, "headline", "resp_a", "resp_b")
        assert result["verdict"] == "model_a"
        assert result["consistent"] is True

    def test_consistent_b_wins(self, monkeypatch):
        call_count = [0]
        def mock_call(client, prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return "B"
            return "A"

        monkeypatch.setattr("evaluation.eval_llm_judge._call_gemini", mock_call)
        monkeypatch.setattr("evaluation.eval_llm_judge.API_CALL_DELAY", 0)

        result = judge_pair(None, "headline", "resp_a", "resp_b")
        assert result["verdict"] == "model_b"
        assert result["consistent"] is True

    def test_consistent_tie(self, monkeypatch):
        monkeypatch.setattr(
            "evaluation.eval_llm_judge._call_gemini",
            lambda c, p: "TIE",
        )
        monkeypatch.setattr("evaluation.eval_llm_judge.API_CALL_DELAY", 0)

        result = judge_pair(None, "headline", "resp_a", "resp_b")
        assert result["verdict"] == "tie"
        assert result["consistent"] is True

    def test_inconsistent_defaults_to_tie(self, monkeypatch):
        call_count = [0]
        def mock_call(client, prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return "A"
            return "A"

        monkeypatch.setattr("evaluation.eval_llm_judge._call_gemini", mock_call)
        monkeypatch.setattr("evaluation.eval_llm_judge.API_CALL_DELAY", 0)

        result = judge_pair(None, "headline", "resp_a", "resp_b")
        assert result["verdict"] == "tie"
        assert result["consistent"] is False

    def test_api_failure_defaults_to_tie(self, monkeypatch):
        monkeypatch.setattr(
            "evaluation.eval_llm_judge._call_gemini",
            lambda c, p: None,
        )
        monkeypatch.setattr("evaluation.eval_llm_judge.API_CALL_DELAY", 0)

        result = judge_pair(None, "headline", "resp_a", "resp_b")
        assert result["verdict"] == "tie"

    def test_result_contains_raw_verdicts(self, monkeypatch):
        call_count = [0]
        def mock_call(client, prompt):
            call_count[0] += 1
            return "A" if call_count[0] == 1 else "B"

        monkeypatch.setattr("evaluation.eval_llm_judge._call_gemini", mock_call)
        monkeypatch.setattr("evaluation.eval_llm_judge.API_CALL_DELAY", 0)

        result = judge_pair(None, "headline", "resp_a", "resp_b")
        assert "pass1_raw" in result
        assert "pass2_raw" in result
        assert result["pass1_raw"] in ("A", "B", "TIE")
        assert result["pass2_raw"] in ("A", "B", "TIE")
