"""Tests for data_preprocessing/synthesize_task_data.py."""

from unittest.mock import MagicMock, patch

from data_preprocessing.synthesize_task_data import (
    _download_content_to_text,
    _parse_batch_download_to_texts,
    query_all_remote_batch_jobs,
    synthesize_for_language,
)


class TestParseBatchDownloadToTexts:
    def test_parse_valid_lines(self):
        content = "\n".join(
            [
                '{"response":{"text":"first text"}}',
                '{"response":{"candidates":[{"content":{"parts":[{"text":"second text"}]}}]}}',
                '{"error":{"code":500}}',
            ]
        )
        result = _parse_batch_download_to_texts(content)
        assert result == ["first text", "second text"]

    def test_download_content_to_text(self):
        assert _download_content_to_text(b"abc") == "abc"
        assert _download_content_to_text("xyz") == "xyz"


class TestSynthesizeForLanguageBackend:
    @patch("data_preprocessing.synthesize_task_data.time.sleep")
    @patch("data_preprocessing.synthesize_task_data._init_gemini_client")
    @patch("data_preprocessing.synthesize_task_data._load_headlines")
    @patch("data_preprocessing.synthesize_task_data._generate_keyword_pairs")
    @patch("data_preprocessing.synthesize_task_data._call_gemini_batch_requests")
    def test_batch_backend_generates_samples(
        self,
        mock_batch,
        mock_pairs,
        mock_headlines,
        mock_client,
        mock_sleep,
    ):
        mock_client.return_value = MagicMock()
        mock_headlines.return_value = ["news headline one", "news headline two"]
        mock_pairs.return_value = [("cat", "moon"), ("piano", "cloud")]
        mock_batch.side_effect = [
            ["headline joke one", "headline joke two"],
            ["cat meets moon in a joke", "piano with cloud as joke"],
        ]

        samples = synthesize_for_language(
            lang="en",
            n_headline=2,
            n_keyword=2,
            backend="batch_api",
            seed=42,
        )

        assert len(samples) == 4
        assert mock_batch.call_count == 2

    @patch("data_preprocessing.synthesize_task_data.time.sleep")
    @patch("data_preprocessing.synthesize_task_data._init_gemini_client")
    @patch("data_preprocessing.synthesize_task_data._load_headlines")
    @patch("data_preprocessing.synthesize_task_data._generate_keyword_pairs")
    @patch("data_preprocessing.synthesize_task_data._call_gemini_batch_requests")
    @patch("data_preprocessing.synthesize_task_data._call_gemini")
    def test_batch_backend_fallbacks_to_realtime(
        self,
        mock_realtime,
        mock_batch,
        mock_pairs,
        mock_headlines,
        mock_client,
        mock_sleep,
    ):
        mock_client.return_value = MagicMock()
        mock_headlines.return_value = ["headline one", "headline two"]
        mock_pairs.return_value = [("cat", "moon"), ("piano", "cloud")]
        mock_batch.side_effect = [[], []]
        mock_realtime.side_effect = [
            "headline realtime one",
            "headline realtime two",
            "cat moon realtime joke",
            "piano cloud realtime joke",
        ]

        samples = synthesize_for_language(
            lang="en",
            n_headline=2,
            n_keyword=2,
            backend="batch_api",
            seed=42,
        )

        assert len(samples) == 4
        assert mock_realtime.call_count == 4


class TestQueryAllRemoteBatchJobs:
    @patch("data_preprocessing.synthesize_task_data._init_gemini_client")
    def test_query_all_remote_batch_jobs(self, mock_client):
        class DummyTypes:
            class ListBatchJobsConfig:
                def __init__(self, page_size):
                    self.page_size = page_size

        import sys

        client = MagicMock()
        job1 = MagicMock()
        job1.name = "batches/1"
        job1.state = "JOB_STATE_PENDING"
        job2 = MagicMock()
        job2.name = "batches/2"
        job2.state = "JOB_STATE_SUCCEEDED"
        client.batches.list.return_value = [job1, job2]
        mock_client.return_value = client

        with patch.dict(sys.modules, {"google.genai.types": DummyTypes, "google.genai": MagicMock(types=DummyTypes)}):
            rows = query_all_remote_batch_jobs(max_jobs=10)

        assert len(rows) == 2
        assert rows[0]["job_id"] == "batches/1"
        assert rows[1]["state"] == "JOB_STATE_SUCCEEDED"
