"""Tests for data_preprocessing/synthesize_task_data.py."""

from unittest.mock import MagicMock, patch

from data_preprocessing.synthesize_task_data import (
    _generate_multi_responses,
    synthesize_for_language,
)


class TestRealtimeMultiBackend:
    @patch("data_preprocessing.synthesize_task_data._call_gemini")
    @patch("data_preprocessing.synthesize_task_data._call_gemini_multi")
    def test_generate_multi_responses_with_fallback(
        self,
        mock_multi,
        mock_single,
    ):
        records = [
            {"prompt": "p0"},
            {"prompt": "p1"},
            {"prompt": "p2"},
        ]
        mock_multi.return_value = [
            {"index": 0, "output": "o0"},
            {"index": 2, "output": "o2"},
        ]
        mock_single.return_value = "o1"

        client = MagicMock()
        outputs = _generate_multi_responses(client, records, lang="en", group_size=10)
        assert outputs == ["o0", "o1", "o2"]


class TestSynthesizeForLanguage:
    @patch("data_preprocessing.synthesize_task_data.time.sleep")
    @patch("data_preprocessing.synthesize_task_data._init_gemini_client")
    @patch("data_preprocessing.synthesize_task_data._load_headlines")
    @patch("data_preprocessing.synthesize_task_data._generate_keyword_pairs")
    @patch("data_preprocessing.synthesize_task_data._call_gemini_multi")
    @patch("data_preprocessing.synthesize_task_data._call_gemini")
    def test_synthesize_uses_realtime_multi_with_fallback(
        self,
        mock_single,
        mock_multi,
        mock_pairs,
        mock_headlines,
        mock_client,
        _mock_sleep,
    ):
        mock_client.return_value = MagicMock()
        mock_headlines.return_value = ["h1", "h2"]
        mock_pairs.return_value = [("cat", "moon"), ("piano", "cloud")]
        # headline chunk -> one miss; keyword chunk -> all present
        mock_multi.side_effect = [
            [{"index": 0, "output": "headline joke 1"}],
            [{"index": 0, "output": "cat moon joke"}, {"index": 1, "output": "piano cloud joke"}],
        ]
        mock_single.return_value = "headline joke 2"

        samples = synthesize_for_language(lang="en", n_headline=2, n_keyword=2, seed=42)
        assert len(samples) == 4
