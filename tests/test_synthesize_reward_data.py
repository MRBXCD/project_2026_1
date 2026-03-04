"""Tests for data_preprocessing/synthesize_reward_data.py — pure logic functions."""

import csv
import json
import re
from unittest.mock import patch, MagicMock

import pytest

from data_preprocessing.synthesize_reward_data import (
    load_cfun_jokes,
    load_high_score_jokes,
    assemble_preference_pairs,
    _filter_boring_response,
    _check_language_match,
    _call_gemini_batch,
    generate_boring_texts,
    BATCH_SIZE,
)

LANG_CHECK_PATCH = "data_preprocessing.synthesize_reward_data._check_language_match"


# ============================================================
# Helpers
# ============================================================

def _write_cfun_csv(path, jokes: list[str]) -> None:
    """Write a mock pure_jokes.csv with BOM header matching real data."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["joke"])
        for joke in jokes:
            writer.writerow([joke])


def _make_mock_unified_jsonl(path, records: list[dict]) -> None:
    """Write mock unified_all.jsonl file."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# TestLoadCfunJokes
# ============================================================

class TestLoadCfunJokes:
    def test_deduplication(self, tmp_path):
        """Duplicate rows in CSV are deduplicated."""
        csv_path = tmp_path / "pure_jokes.csv"
        _write_cfun_csv(csv_path, [
            "这是一个重复的笑话，长度足够通过过滤",
            "这是一个重复的笑话，长度足够通过过滤",
            "这是一个重复的笑话，长度足够通过过滤",
            "这是另一个不同的笑话，同样足够长度通过",
        ])

        with patch(
            "data_preprocessing.synthesize_reward_data.CFUN_JOKES_CSV",
            csv_path,
        ):
            jokes = load_cfun_jokes(n_samples=100)

        assert len(jokes) == 2

    def test_length_filter(self, tmp_path):
        """Texts shorter than 10 or longer than 500 chars are excluded."""
        csv_path = tmp_path / "pure_jokes.csv"
        _write_cfun_csv(csv_path, [
            "短",
            "…",
            "这是一个合格的笑话，长度刚好在范围内",
            "很长" * 300,
        ])

        with patch(
            "data_preprocessing.synthesize_reward_data.CFUN_JOKES_CSV",
            csv_path,
        ):
            jokes = load_cfun_jokes(n_samples=100)

        assert len(jokes) == 1
        assert "合格" in jokes[0]

    def test_samples_correct_count(self, tmp_path):
        """When pool is large enough, returns exactly n_samples."""
        csv_path = tmp_path / "pure_jokes.csv"
        _write_cfun_csv(csv_path, [
            f"这是第{i}个笑话，有足够的长度来通过过滤" for i in range(50)
        ])

        with patch(
            "data_preprocessing.synthesize_reward_data.CFUN_JOKES_CSV",
            csv_path,
        ):
            jokes = load_cfun_jokes(n_samples=10)

        assert len(jokes) == 10

    def test_file_not_found(self, tmp_path):
        """Missing CSV raises FileNotFoundError."""
        with patch(
            "data_preprocessing.synthesize_reward_data.CFUN_JOKES_CSV",
            tmp_path / "nonexistent.csv",
        ):
            with pytest.raises(FileNotFoundError):
                load_cfun_jokes(n_samples=10)

    def test_empty_after_filter(self, tmp_path):
        """All rows filtered out raises ValueError."""
        csv_path = tmp_path / "pure_jokes.csv"
        _write_cfun_csv(csv_path, [
            "短",
            "…",
            "",
            "！",
        ])

        with patch(
            "data_preprocessing.synthesize_reward_data.CFUN_JOKES_CSV",
            csv_path,
        ):
            with pytest.raises(ValueError, match="No CFun jokes passed filtering"):
                load_cfun_jokes(n_samples=10)


# ============================================================
# TestLoadHighScoreJokes
# ============================================================

class TestLoadHighScoreJokes:
    def test_score_threshold(self, tmp_path):
        """Only jokes with score >= threshold are returned."""
        records = [
            {"text": "A pretty good joke about something", "lang": "en", "score": 0.8, "source": "rjokes"},
            {"text": "A great joke here for testing", "lang": "en", "score": 0.9, "source": "rjokes"},
            {"text": "A bad joke text", "lang": "en", "score": 0.3, "source": "rjokes"},
            {"text": "A medium joke text", "lang": "en", "score": 0.5, "source": "rjokes"},
        ]
        unified_path = tmp_path / "unified_all.jsonl"
        _make_mock_unified_jsonl(unified_path, records)

        with patch(
            "data_preprocessing.synthesize_reward_data.UNIFIED_ALL_FILE",
            unified_path,
        ):
            jokes = load_high_score_jokes("en", n_samples=100, score_threshold=0.7)

        assert len(jokes) == 2
        assert any("pretty good" in j for j in jokes)
        assert any("great joke" in j for j in jokes)

    def test_correct_source_filter(self, tmp_path):
        """rjokes -> en, haha -> es, other sources ignored."""
        records = [
            {"text": "An English joke for testing purposes", "lang": "en", "score": 0.9, "source": "rjokes"},
            {"text": "A Spanish joke for testing purposes", "lang": "es", "score": 0.9, "source": "haha"},
            {"text": "A Chinese joke for testing purposes", "lang": "zh", "score": 0.9, "source": "chinese_humor"},
        ]
        unified_path = tmp_path / "unified_all.jsonl"
        _make_mock_unified_jsonl(unified_path, records)

        with patch(
            "data_preprocessing.synthesize_reward_data.UNIFIED_ALL_FILE",
            unified_path,
        ):
            en_jokes = load_high_score_jokes("en", n_samples=100, score_threshold=0.7)
            es_jokes = load_high_score_jokes("es", n_samples=100, score_threshold=0.7)

        assert len(en_jokes) == 1
        assert "English" in en_jokes[0]
        assert len(es_jokes) == 1
        assert "Spanish" in es_jokes[0]

    def test_unsupported_lang_raises(self, tmp_path):
        """Requesting zh (not supported by load_high_score_jokes) raises ValueError."""
        unified_path = tmp_path / "unified_all.jsonl"
        _make_mock_unified_jsonl(unified_path, [])

        with patch(
            "data_preprocessing.synthesize_reward_data.UNIFIED_ALL_FILE",
            unified_path,
        ):
            with pytest.raises(ValueError, match="only supports en/es"):
                load_high_score_jokes("zh", n_samples=10)

    def test_file_not_found_raises(self, tmp_path):
        """Missing unified_all.jsonl raises FileNotFoundError."""
        with patch(
            "data_preprocessing.synthesize_reward_data.UNIFIED_ALL_FILE",
            tmp_path / "nonexistent.jsonl",
        ):
            with pytest.raises(FileNotFoundError):
                load_high_score_jokes("en", n_samples=10)


# ============================================================
# TestAssemblePreferencePairs
# ============================================================

class TestAssemblePreferencePairs:
    def test_output_format_matches_trl(self):
        """Output has prompt/chosen/rejected keys with correct structure."""
        chosen = ["Joke A", "Joke B", "Joke C"]
        rejected = ["Boring 1", "Boring 2", "Boring 3"]

        pairs = assemble_preference_pairs(chosen, rejected, lang="en")

        assert len(pairs) == 3
        for pair in pairs:
            assert set(pair.keys()) == {"prompt", "chosen", "rejected"}
            assert isinstance(pair["prompt"], list)
            assert len(pair["prompt"]) == 1
            assert pair["prompt"][0]["role"] == "user"
            assert isinstance(pair["chosen"], list)
            assert len(pair["chosen"]) == 1
            assert pair["chosen"][0]["role"] == "assistant"
            assert isinstance(pair["rejected"], list)
            assert len(pair["rejected"]) == 1
            assert pair["rejected"][0]["role"] == "assistant"

    def test_prompt_uses_correct_language(self):
        """Prompt language matches the specified lang parameter."""
        for lang, pattern in [
            ("zh", r"[\u4e00-\u9fff]"),
            ("en", r"[a-zA-Z]"),
            ("es", r"[a-záéíóúñ¿¡]"),
        ]:
            pairs = assemble_preference_pairs(
                ["chosen text"], ["rejected text"], lang=lang,
            )
            prompt_text = pairs[0]["prompt"][0]["content"]
            assert re.search(pattern, prompt_text), (
                f"Expected {lang} characters in prompt, got: {prompt_text}"
            )

    def test_min_length_pairing(self):
        """Number of pairs = min(len(chosen), len(rejected))."""
        pairs = assemble_preference_pairs(
            ["A", "B", "C"], ["X", "Y"], lang="en",
        )
        assert len(pairs) == 2

    def test_chosen_rejected_content_preserved(self):
        """Chosen and rejected texts appear in the output."""
        pairs = assemble_preference_pairs(
            ["My great joke"], ["A boring statement"], lang="en",
        )
        assert pairs[0]["chosen"][0]["content"] == "My great joke"
        assert pairs[0]["rejected"][0]["content"] == "A boring statement"


# ============================================================
# TestFilterBoringResponse
# ============================================================

class TestFilterBoringResponse:
    def test_empty_rejected(self):
        assert _filter_boring_response("") is False
        assert _filter_boring_response(None) is False

    def test_too_short(self):
        assert _filter_boring_response("Hi") is False

    def test_too_long(self):
        assert _filter_boring_response("x" * 501) is False

    def test_refusal_pattern(self):
        assert _filter_boring_response("I'm sorry, I cannot help.") is False
        assert _filter_boring_response("抱歉，我无法生成这样的内容") is False

    def test_valid_boring_text(self):
        assert _filter_boring_response("The book is on the table.") is True
        assert _filter_boring_response("今天天气不错，气温二十度。") is True

    def test_lang_none_skips_check(self):
        assert _filter_boring_response("English text is fine.", lang=None) is True
        assert _filter_boring_response("中文文本也可以通过。", lang=None) is True

    def test_lang_match_accepted(self):
        with patch(LANG_CHECK_PATCH, return_value=True):
            assert _filter_boring_response("今天天气不错，气温二十度。", lang="zh") is True

    def test_lang_mismatch_rejected(self):
        with patch(LANG_CHECK_PATCH, return_value=False):
            assert _filter_boring_response("The book is on the table.", lang="zh") is False

    def test_lang_check_called_when_lang_set(self):
        with patch(LANG_CHECK_PATCH, return_value=False) as mock_check:
            result = _filter_boring_response("今天天气不错，气温二十度。", lang="zh")
        assert result is False
        mock_check.assert_called_once_with("今天天气不错，气温二十度。", "zh")


# ============================================================
# TestCheckLanguageMatch
# ============================================================

class TestCheckLanguageMatch:
    """Tests for _check_language_match with langid mocked."""

    @pytest.fixture(autouse=True)
    def _mock_langid(self):
        """Inject a fake langid into sys.modules so the lazy import picks it up."""
        import sys
        mock_langid = MagicMock()
        with patch.dict(sys.modules, {"langid": mock_langid}):
            self._langid_mock = mock_langid
            yield

    def _set_classify(self, detected_lang: str, confidence: float):
        self._langid_mock.classify.return_value = (detected_lang, confidence)

    def test_matching_lang_returns_true(self):
        self._set_classify("zh", -10.0)
        assert _check_language_match("今天下雨了", "zh") is True

    def test_mismatched_lang_returns_false(self):
        self._set_classify("en", -10.0)
        assert _check_language_match("It is raining today", "zh") is False

    def test_set_languages_called_with_all_three(self):
        self._set_classify("en", -10.0)
        _check_language_match("hello", "en")
        self._langid_mock.set_languages.assert_called_once_with(["en", "zh", "es"])


# ============================================================
# TestCallGeminiBatch
# ============================================================

class TestCallGeminiBatch:
    """Tests for _call_gemini_batch with google.genai mocked out."""

    @pytest.fixture(autouse=True)
    def _mock_genai(self):
        """Inject a fake google.genai.types module so the lazy import succeeds."""
        import sys
        types_mod = MagicMock()
        genai_mod = MagicMock()
        genai_mod.types = types_mod
        google_mod = MagicMock()
        google_mod.genai = genai_mod

        with patch.dict(sys.modules, {
            "google": google_mod,
            "google.genai": genai_mod,
            "google.genai.types": types_mod,
        }):
            yield

    def _make_mock_client(self, response_text: str | None):
        """Build a mock Gemini client returning a fixed response."""
        client = MagicMock()
        response = MagicMock()
        response.text = response_text
        client.models.generate_content.return_value = response
        return client

    def test_parses_json_array(self):
        payload = json.dumps(["Statement one.", "Statement two.", "Statement three."])
        client = self._make_mock_client(payload)
        result = _call_gemini_batch(client, "prompt")
        assert result == ["Statement one.", "Statement two.", "Statement three."]

    def test_strips_whitespace(self):
        payload = json.dumps(["  padded text  ", "\nnewlines\n"])
        client = self._make_mock_client(payload)
        result = _call_gemini_batch(client, "prompt")
        assert result == ["padded text", "newlines"]

    def test_filters_empty_items(self):
        payload = json.dumps(["valid text", "", None, "another"])
        client = self._make_mock_client(payload)
        result = _call_gemini_batch(client, "prompt")
        assert "valid text" in result
        assert "another" in result
        assert "" not in result

    def test_empty_response_returns_empty_list(self):
        client = self._make_mock_client(None)
        result = _call_gemini_batch(client, "prompt")
        assert result == []

    def test_non_array_json_returns_empty(self):
        payload = json.dumps({"key": "value"})
        client = self._make_mock_client(payload)
        result = _call_gemini_batch(client, "prompt")
        assert result == []

    def test_retries_on_rate_limit(self):
        client = MagicMock()
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 Too Many Requests")
            response = MagicMock()
            response.text = json.dumps(["recovered text"])
            return response

        client.models.generate_content.side_effect = side_effect

        with patch("data_preprocessing.synthesize_reward_data.time.sleep"):
            result = _call_gemini_batch(client, "prompt")

        assert result == ["recovered text"]
        assert call_count == 2

    def test_all_retries_exhausted_returns_empty(self):
        client = MagicMock()
        client.models.generate_content.side_effect = Exception("server error")

        with patch("data_preprocessing.synthesize_reward_data.time.sleep"):
            result = _call_gemini_batch(client, "prompt", max_retries=2)

        assert result == []
        assert client.models.generate_content.call_count == 2


# ============================================================
# TestGenerateBoringTexts
# ============================================================

class TestGenerateBoringTexts:
    """Tests for generate_boring_texts batch orchestration logic.

    _check_language_match is patched to always return True so these tests
    focus on batching and filtering behavior, not language detection.
    """

    @pytest.fixture(autouse=True)
    def _bypass_lang_check(self):
        with patch(LANG_CHECK_PATCH, return_value=True):
            yield

    @patch("data_preprocessing.synthesize_reward_data.time.sleep")
    @patch("data_preprocessing.synthesize_reward_data._call_gemini_batch")
    @patch("data_preprocessing.synthesize_reward_data._init_gemini_client")
    def test_single_batch_sufficient(self, mock_client, mock_batch, mock_sleep):
        mock_client.return_value = MagicMock()
        valid_texts = [f"Boring statement number {i}." for i in range(120)]
        mock_batch.return_value = valid_texts

        result = generate_boring_texts("en", n_samples=50)

        assert len(result) == 50
        assert mock_batch.call_count >= 1

    @patch("data_preprocessing.synthesize_reward_data.time.sleep")
    @patch("data_preprocessing.synthesize_reward_data._call_gemini_batch")
    @patch("data_preprocessing.synthesize_reward_data._init_gemini_client")
    def test_multiple_batches_needed(self, mock_client, mock_batch, mock_sleep):
        mock_client.return_value = MagicMock()
        batch_texts = [f"这是第{i}条测试用的无聊陈述。" for i in range(80)]
        mock_batch.return_value = batch_texts

        result = generate_boring_texts("zh", n_samples=200)

        assert len(result) == 200
        assert mock_batch.call_count >= 3

    @patch("data_preprocessing.synthesize_reward_data.time.sleep")
    @patch("data_preprocessing.synthesize_reward_data._call_gemini_batch")
    @patch("data_preprocessing.synthesize_reward_data._init_gemini_client")
    def test_filters_bad_items(self, mock_client, mock_batch, mock_sleep):
        mock_client.return_value = MagicMock()
        mixed = [
            "Normal boring text here.",
            "I'm sorry, I cannot help.",
            "抱歉，我无法完成",
            "Another normal statement.",
            "x" * 501,
            "",
        ]
        mock_batch.return_value = mixed

        result = generate_boring_texts("en", n_samples=2)

        assert len(result) == 2
        for text in result:
            assert "sorry" not in text.lower()
            assert "抱歉" not in text

    @patch("data_preprocessing.synthesize_reward_data.time.sleep")
    @patch("data_preprocessing.synthesize_reward_data._call_gemini_batch")
    @patch("data_preprocessing.synthesize_reward_data._init_gemini_client")
    def test_early_stop_when_target_reached(self, mock_client, mock_batch, mock_sleep):
        mock_client.return_value = MagicMock()
        mock_batch.return_value = [f"Text {i}" for i in range(BATCH_SIZE)]

        result = generate_boring_texts("es", n_samples=50)

        assert len(result) == 50
        assert mock_batch.call_count == 1

    def test_unsupported_lang_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            generate_boring_texts("fr", n_samples=10)
