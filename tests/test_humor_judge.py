"""Tests for rl/humor_judge.py"""

from unittest.mock import MagicMock, patch

import pytest

from rl.humor_judge import (
    _parse_batch_scores,
    _parse_single_score,
)


# ============================================================
# 1. _parse_single_score
# ============================================================

class TestParseSingleScore:
    def test_score_3(self):
        assert _parse_single_score("3") == pytest.approx(0.0)

    def test_score_5(self):
        assert _parse_single_score("5") == pytest.approx(1.0)

    def test_score_1(self):
        assert _parse_single_score("1") == pytest.approx(-1.0)

    def test_score_from_noisy_text(self):
        # "4" is the first valid 1-5 integer found
        assert _parse_single_score("Score: 4/5") == pytest.approx(0.5)

    def test_no_valid_number(self):
        assert _parse_single_score("no number here") is None

    def test_number_out_of_range(self):
        # "10" is > 5, "0" is < 1; neither should match
        assert _parse_single_score("The score is 10 out of 0") is None

    def test_score_2(self):
        assert _parse_single_score("2") == pytest.approx(-0.5)


# ============================================================
# 2. _parse_batch_scores
# ============================================================

class TestParseBatchScores:
    def test_standard_format(self):
        text = "1: 3\n2: 4\n3: 2"
        scores = _parse_batch_scores(text, 3)
        assert len(scores) == 3
        assert scores[0] == pytest.approx(0.0)   # 3 -> 0.0
        assert scores[1] == pytest.approx(0.5)   # 4 -> 0.5
        assert scores[2] == pytest.approx(-0.5)  # 2 -> -0.5

    def test_out_of_order(self):
        text = "2: 4\n1: 3"
        scores = _parse_batch_scores(text, 2)
        assert scores[0] == pytest.approx(0.0)   # index 1 -> score 3
        assert scores[1] == pytest.approx(0.5)   # index 2 -> score 4

    def test_fallback_unstructured(self):
        text = "3 4 2"
        scores = _parse_batch_scores(text, 3)
        assert scores[0] == pytest.approx(0.0)   # 3
        assert scores[1] == pytest.approx(0.5)   # 4
        assert scores[2] == pytest.approx(-0.5)  # 2

    def test_fewer_scores_than_expected(self):
        text = "1: 3\n2: 4"
        scores = _parse_batch_scores(text, 5)
        assert len(scores) == 5
        assert scores[0] == pytest.approx(0.0)
        assert scores[1] == pytest.approx(0.5)
        assert scores[2] is None
        assert scores[3] is None
        assert scores[4] is None

    def test_empty_text(self):
        scores = _parse_batch_scores("", 3)
        assert len(scores) == 3
        assert all(s is None for s in scores)


# ============================================================
# 3. build_humor_scorer (mock API)
# ============================================================

class TestBuildHumorScorer:
    @patch("rl.humor_judge._call_gemini", return_value="4")
    @patch("rl.humor_judge._init_gemini_client", return_value=MagicMock())
    def test_returns_float(self, mock_client, mock_call):
        from rl.humor_judge import build_humor_scorer
        scorer = build_humor_scorer(call_delay=0)
        result = scorer("Write a joke", "A funny joke")
        assert isinstance(result, float)
        assert result == pytest.approx(0.5)  # score 4 -> 0.5

    @patch("rl.humor_judge._call_gemini", return_value=None)
    @patch("rl.humor_judge._init_gemini_client", return_value=MagicMock())
    def test_api_failure_returns_zero(self, mock_client, mock_call):
        from rl.humor_judge import build_humor_scorer
        scorer = build_humor_scorer(call_delay=0)
        result = scorer("Write a joke", "A funny joke")
        assert result == 0.0

    @patch("rl.humor_judge._call_gemini", return_value="garbage text no numbers")
    @patch("rl.humor_judge._init_gemini_client", return_value=MagicMock())
    def test_unparseable_response_returns_zero(self, mock_client, mock_call):
        from rl.humor_judge import build_humor_scorer
        scorer = build_humor_scorer(call_delay=0)
        result = scorer("Write a joke", "A funny joke")
        assert result == 0.0


# ============================================================
# 4. build_batch_humor_scorer (mock API)
# ============================================================

class TestBuildBatchHumorScorer:
    @patch("rl.humor_judge._call_gemini", return_value="1: 3\n2: 5\n3: 1")
    @patch("rl.humor_judge._init_gemini_client", return_value=MagicMock())
    def test_returns_correct_length(self, mock_client, mock_call):
        from rl.humor_judge import build_batch_humor_scorer
        scorer = build_batch_humor_scorer(batch_size=8, call_delay=0)
        scores = scorer(
            ["p1", "p2", "p3"],
            ["r1", "r2", "r3"],
        )
        assert len(scores) == 3

    @patch("rl.humor_judge._call_gemini", return_value="1: 4\n2: 2")
    @patch("rl.humor_judge._init_gemini_client", return_value=MagicMock())
    def test_scores_mapped_correctly(self, mock_client, mock_call):
        from rl.humor_judge import build_batch_humor_scorer
        scorer = build_batch_humor_scorer(batch_size=8, call_delay=0)
        scores = scorer(["p1", "p2"], ["r1", "r2"])
        assert scores[0] == pytest.approx(0.5)   # 4 -> 0.5
        assert scores[1] == pytest.approx(-0.5)  # 2 -> -0.5

    @patch("rl.humor_judge._call_gemini", return_value=None)
    @patch("rl.humor_judge._init_gemini_client", return_value=MagicMock())
    def test_api_failure_returns_zeros(self, mock_client, mock_call):
        from rl.humor_judge import build_batch_humor_scorer
        scorer = build_batch_humor_scorer(batch_size=8, call_delay=0)
        scores = scorer(["p1", "p2"], ["r1", "r2"])
        assert scores == [0.0, 0.0]

    @patch("rl.humor_judge._init_gemini_client", return_value=MagicMock())
    def test_multi_batch_calls(self, mock_client):
        # batch_size=2, 5 items -> should make 3 API calls (2+2+1)
        call_results = ["1: 4\n2: 3", "1: 5\n2: 1", "1: 2"]

        with patch("rl.humor_judge._call_gemini", side_effect=call_results):
            from rl.humor_judge import build_batch_humor_scorer
            scorer = build_batch_humor_scorer(batch_size=2, call_delay=0)
            scores = scorer(
                ["p1", "p2", "p3", "p4", "p5"],
                ["r1", "r2", "r3", "r4", "r5"],
            )

        assert len(scores) == 5
        assert scores[0] == pytest.approx(0.5)   # batch 1: 4
        assert scores[1] == pytest.approx(0.0)   # batch 1: 3
        assert scores[2] == pytest.approx(1.0)   # batch 2: 5
        assert scores[3] == pytest.approx(-1.0)  # batch 2: 1
        assert scores[4] == pytest.approx(-0.5)  # batch 3: 2
