"""Tests for data_preprocessing/synthesize_task_data.py."""

import json
from unittest.mock import MagicMock, patch

import pytest

from data_preprocessing import synthesize_task_data as synth
from data_preprocessing.synthesize_task_data import (
    _generate_multi_responses,
    synthesize_for_language,
)


class FakeToken:
    def __init__(
        self,
        text: str,
        pos: str,
        *,
        is_alpha: bool = True,
        is_punct: bool = False,
        like_num: bool = False,
    ):
        self.text = text
        self.pos_ = pos
        self.is_alpha = is_alpha
        self.is_punct = is_punct
        self.like_num = like_num


class FakeNlp:
    def __init__(self, token_specs: dict[str, tuple[str, bool]]):
        self.token_specs = token_specs

    def pipe(self, texts, batch_size: int = 128):
        for text in texts:
            pos, is_alpha = self.token_specs[text]
            yield [FakeToken(text, pos, is_alpha=is_alpha)]


class TestKeywordPoolHelpers:
    def test_normalize_keyword_uses_language_specific_rules(self):
        assert synth._normalize_keyword("  Café\tDel  Mar ", "es") == "café del mar"
        assert synth._normalize_keyword("  笔记 本\t电脑 ", "zh") == "笔记本电脑"
        assert synth._normalize_keyword("  Space   Ship  ", "en") == "space ship"

    def test_load_semeval_keyword_constraints_ignores_headlines(self, tmp_path):
        semeval_path = tmp_path / "task-a-en.tsv"
        semeval_path.write_text(
            "id\tword1\tword2\theadline\n"
            "en_1\t-\t-\tOnly a headline\n"
            "en_2\tBanana\tChair\t-\n"
            "en_3\tCamel\tTable\t-\n",
            encoding="utf-8",
        )

        with patch.dict(synth.SEMEVAL_TASK_FILES, {"en": semeval_path}, clear=False):
            blocked_words, blocked_pairs = synth._load_semeval_keyword_constraints("en")

        assert "banana" in blocked_words
        assert "chair" in blocked_words
        assert "only a headline" not in blocked_words
        assert ("banana", "chair") in blocked_pairs
        assert ("camel", "table") in blocked_pairs

    def test_keyword_pool_cache_roundtrip(self, tmp_path):
        output_path = synth._write_keyword_pool_cache(
            lang="en",
            keywords=["apple", "river", "planet"],
            target_size=3,
            blocked_words_count=5,
            cache_dir=tmp_path,
        )

        assert output_path == tmp_path / "en.json"
        loaded = synth._load_keyword_pool("en", cache_dir=tmp_path, min_size=2)
        assert loaded == ["apple", "river", "planet"]

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["lang"] == "en"
        assert payload["target_size"] == 3
        assert payload["blocked_words_count"] == 5

    def test_load_keyword_pool_rejects_duplicates(self, tmp_path):
        (tmp_path / "en.json").write_text(
            json.dumps(
                {
                    "lang": "en",
                    "target_size": 3,
                    "blocked_words_count": 0,
                    "keywords": ["apple", "apple", "river"],
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="duplicate"):
            synth._load_keyword_pool("en", cache_dir=tmp_path, min_size=2)

    @patch("data_preprocessing.synthesize_task_data._load_spacy_pipeline")
    @patch("data_preprocessing.synthesize_task_data._collect_wordfreq_candidates")
    @patch("data_preprocessing.synthesize_task_data._load_semeval_keyword_constraints")
    def test_build_keyword_pool_filters_and_saves_cache(
        self,
        mock_constraints,
        mock_candidates,
        mock_load_spacy,
        tmp_path,
    ):
        mock_constraints.return_value = (
            {"banana", "chair"},
            {("apple", "river")},
        )
        mock_candidates.return_value = [
            "banana",
            "apple",
            "river",
            "run",
            "planet",
            "city",
            "river",
        ]
        mock_load_spacy.return_value = FakeNlp(
            {
                "banana": ("NOUN", True),
                "apple": ("NOUN", True),
                "river": ("NOUN", True),
                "run": ("VERB", True),
                "planet": ("NOUN", True),
                "city": ("PROPN", True),
            }
        )

        output_path = synth.build_keyword_pool(
            lang="en",
            target_size=3,
            cache_dir=tmp_path,
            candidate_limit=20,
        )

        assert output_path == tmp_path / "en.json"
        assert synth._load_keyword_pool("en", cache_dir=tmp_path, min_size=3) == [
            "apple",
            "river",
            "planet",
        ]

    @patch("data_preprocessing.synthesize_task_data._load_semeval_keyword_constraints")
    @patch("data_preprocessing.synthesize_task_data._load_keyword_pool")
    def test_generate_keyword_pairs_uses_file_pool_and_skips_blocked_pairs(
        self,
        mock_load_pool,
        mock_constraints,
    ):
        mock_load_pool.return_value = ["apple", "river", "planet"]
        mock_constraints.return_value = (
            set(),
            {("apple", "river")},
        )

        pairs = synth._generate_keyword_pairs("en", 10, seed=42)
        pair_set = {tuple(sorted(pair)) for pair in pairs}

        assert ("apple", "river") not in pair_set
        assert pair_set == {("apple", "planet"), ("planet", "river")}


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
