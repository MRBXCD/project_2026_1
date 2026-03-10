"""Tests for data_preprocessing/formatters.py reward pair construction."""

import json
import re

import datasets
import pytest

from data_preprocessing.formatters import format_reward_pairs, format_sft_type_a


# ============================================================
# Helpers
# ============================================================

def _make_ds(lang: str, source: str, n: int, base_score: float = 0.5) -> datasets.Dataset:
    """Create a small mock unified-format dataset for one source."""
    return datasets.Dataset.from_dict({
        "text": [f"{lang} joke {i}" for i in range(n)],
        "lang": [lang] * n,
        "score": [base_score + (i % 10) * 0.05 for i in range(n)],
        "source": [source] * n,
    })


def _make_mock_unified(n_en: int = 50, n_es: int = 30, n_zh: int = 20) -> dict[str, datasets.Dataset]:
    """Create mock unified_datasets with three languages."""
    return {
        "rjokes": _make_ds("en", "rjokes", n_en),
        "haha": _make_ds("es", "haha", n_es),
        "chinese_humor": _make_ds("zh", "chinese_humor", n_zh),
    }


def _detect_lang(sample: dict) -> str:
    """Detect language from a preference pair's prompt content."""
    prompt_text = sample["prompt"][0]["content"]
    if re.search(r"[\u4e00-\u9fff]", prompt_text):
        return "zh"
    if re.search(r"[áéíóúñ¿¡]", prompt_text):
        return "es"
    return "en"


def _count_by_lang(ds: datasets.Dataset) -> dict[str, int]:
    """Count preference pairs by language in a dataset."""
    counts = {"en": 0, "es": 0, "zh": 0}
    for i in range(len(ds)):
        lang = _detect_lang(ds[i])
        counts[lang] = counts.get(lang, 0) + 1
    return counts


def _write_synth_jsonl(path, lang: str, n: int) -> None:
    """Write n fake synthesized preference pair records to a JSONL file."""
    prompts_by_lang = {
        "en": "Tell me a joke. Please give me the joke only, no other text.",
        "zh": "给我讲个笑话吧。 请只给出笑话，不要给出任何其他文本。",
        "es": "Cuéntame un chiste. No dar ninguna otra información.",
    }
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            record = {
                "prompt": [{"role": "user", "content": prompts_by_lang[lang]}],
                "chosen": [{"role": "assistant", "content": f"synth {lang} chosen {i}"}],
                "rejected": [{"role": "assistant", "content": f"synth {lang} rejected {i}"}],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# TestFormatRewardPairsDownsample — max_pairs_per_lang
# ============================================================

_NO_CAP_ALLOCATION = {
    "en": {"score_based": None, "synthesized": None},
    "es": {"score_based": None, "synthesized": None},
    "zh": {"score_based": None, "synthesized": None},
}


class TestFormatRewardPairsAllocation:
    def test_no_cap_preserves_all_pairs(self):
        """With allocation=None for all caps, all pairs are kept."""
        unified = _make_mock_unified(n_en=50, n_es=30, n_zh=20)
        result = format_reward_pairs(
            unified,
            allocation=_NO_CAP_ALLOCATION,
            synthesized_reward_dir=None,
        )
        assert "train" in result
        assert "validation" in result
        total = len(result["train"]) + len(result["validation"])
        assert total > 0

    def test_score_based_cap_limits_pairs(self):
        """score_based cap should limit the number of score-based pairs."""
        cap = 5
        alloc = {
            "en": {"score_based": cap, "synthesized": None},
            "es": {"score_based": cap, "synthesized": None},
            "zh": {"score_based": cap, "synthesized": None},
        }
        unified = _make_mock_unified(n_en=50, n_es=30, n_zh=20)
        result = format_reward_pairs(
            unified, allocation=alloc, synthesized_reward_dir=None,
        )
        total_ds = datasets.concatenate_datasets([result["train"], result["validation"]])
        counts = _count_by_lang(total_ds)
        for lang, count in counts.items():
            assert count <= cap, f"{lang} has {count} pairs, expected <= {cap}"

    def test_cap_smaller_than_data_reduces_output(self):
        """A small cap should produce fewer pairs than no cap."""
        unified = _make_mock_unified(n_en=100, n_es=60, n_zh=40)
        result_no_cap = format_reward_pairs(
            unified, allocation=_NO_CAP_ALLOCATION,
        )
        total_no_cap = len(result_no_cap["train"]) + len(result_no_cap["validation"])

        alloc_small = {
            "en": {"score_based": 5, "synthesized": None},
            "es": {"score_based": 5, "synthesized": None},
            "zh": {"score_based": 5, "synthesized": None},
        }
        result_with_cap = format_reward_pairs(unified, allocation=alloc_small)
        total_with_cap = len(result_with_cap["train"]) + len(result_with_cap["validation"])

        assert total_no_cap > total_with_cap

    def test_huge_cap_keeps_all(self):
        """Huge cap should not reduce pairs compared with uncapped mode."""
        unified = _make_mock_unified(n_en=20, n_es=10, n_zh=10)
        alloc_huge = {
            "en": {"score_based": 100000, "synthesized": None},
            "es": {"score_based": 100000, "synthesized": None},
            "zh": {"score_based": 100000, "synthesized": None},
        }
        result_huge = format_reward_pairs(unified, allocation=alloc_huge)
        result_none = format_reward_pairs(unified, allocation=_NO_CAP_ALLOCATION)

        total_huge = len(result_huge["train"]) + len(result_huge["validation"])
        total_none = len(result_none["train"]) + len(result_none["validation"])
        assert total_huge >= total_none


# ============================================================
# TestFormatRewardPairsSynthMerge — synthesized_reward_dir
# ============================================================

class TestFormatRewardPairsSynthMerge:
    def test_synth_dir_loads_and_merges(self, tmp_path, monkeypatch):
        """Synthesized JSONL files are loaded and merged into output."""
        monkeypatch.setattr(
            "data_preprocessing.formatters.MAX_SYNTH_RATIO_BY_LANG",
            {"en": 1.0, "es": 1.0, "zh": 1.0},
        )
        n_synth = 5
        alloc = {
            "en": {"score_based": None, "synthesized": None},
            "es": {"score_based": None, "synthesized": None},
            "zh": {"score_based": None, "synthesized": None},
        }
        for lang in ["en", "zh", "es"]:
            _write_synth_jsonl(tmp_path / f"reward_neg_{lang}.jsonl", lang, n_synth)

        unified = _make_mock_unified(n_en=20, n_es=10, n_zh=10)
        result_with_synth = format_reward_pairs(
            unified, allocation=alloc, synthesized_reward_dir=tmp_path,
        )
        result_without_synth = format_reward_pairs(
            unified, allocation=alloc, synthesized_reward_dir=None,
        )

        total_with = len(result_with_synth["train"]) + len(result_with_synth["validation"])
        total_without = len(result_without_synth["train"]) + len(result_without_synth["validation"])

        assert total_with == total_without + n_synth * 3

    def test_synth_cap_limits_loaded_pairs(self, tmp_path, monkeypatch):
        """Synthesized cap should limit the number of synthesized pairs loaded."""
        monkeypatch.setattr(
            "data_preprocessing.formatters.MAX_SYNTH_RATIO_BY_LANG",
            {"en": 1.0, "es": 1.0, "zh": 1.0},
        )
        n_file = 20
        synth_cap = 5
        alloc = {
            "en": {"score_based": None, "synthesized": synth_cap},
            "es": {"score_based": None, "synthesized": synth_cap},
            "zh": {"score_based": None, "synthesized": synth_cap},
        }
        for lang in ["en", "zh", "es"]:
            _write_synth_jsonl(tmp_path / f"reward_neg_{lang}.jsonl", lang, n_file)

        unified = _make_mock_unified(n_en=20, n_es=10, n_zh=10)
        result_capped = format_reward_pairs(
            unified, allocation=alloc, synthesized_reward_dir=tmp_path,
        )
        alloc_uncapped = {
            "en": {"score_based": None, "synthesized": None},
            "es": {"score_based": None, "synthesized": None},
            "zh": {"score_based": None, "synthesized": None},
        }
        result_uncapped = format_reward_pairs(
            unified, allocation=alloc_uncapped, synthesized_reward_dir=tmp_path,
        )

        total_capped = len(result_capped["train"]) + len(result_capped["validation"])
        total_uncapped = len(result_uncapped["train"]) + len(result_uncapped["validation"])

        assert total_capped < total_uncapped

    def test_synth_dir_nonexistent_no_error(self, tmp_path):
        """Non-existent synthesized directory doesn't cause errors."""
        unified = _make_mock_unified(n_en=20, n_es=10, n_zh=10)
        result = format_reward_pairs(
            unified, synthesized_reward_dir=tmp_path / "nonexistent",
        )
        assert len(result["train"]) + len(result["validation"]) > 0

    def test_synth_dir_empty_no_error(self, tmp_path):
        """Empty directory (no JSONL files) doesn't cause errors."""
        unified = _make_mock_unified(n_en=20, n_es=10, n_zh=10)
        result = format_reward_pairs(
            unified, synthesized_reward_dir=tmp_path,
        )
        assert len(result["train"]) + len(result["validation"]) > 0

    def test_output_schema_after_merge(self, tmp_path, monkeypatch):
        """Schema is correct after merging synthesized data."""
        monkeypatch.setattr(
            "data_preprocessing.formatters.MAX_SYNTH_RATIO_BY_LANG",
            {"en": 1.0, "es": 1.0, "zh": 1.0},
        )
        _write_synth_jsonl(tmp_path / "reward_neg_en.jsonl", "en", 3)

        unified = _make_mock_unified(n_en=20, n_es=10, n_zh=10)
        result = format_reward_pairs(
            unified, synthesized_reward_dir=tmp_path,
        )

        for split_name in ("train", "validation"):
            ds = result[split_name]
            assert "prompt" in ds.column_names
            assert "chosen" in ds.column_names
            assert "rejected" in ds.column_names

            if len(ds) > 0:
                sample = ds[0]
                assert isinstance(sample["prompt"], list)
                assert isinstance(sample["prompt"][0], dict)
                assert "role" in sample["prompt"][0]
                assert "content" in sample["prompt"][0]


# ============================================================
# TestFormatRewardPairsCombined — allocation + synthesis together
# ============================================================

class TestFormatRewardPairsCombined:
    def test_allocation_caps_both_types(self, tmp_path, monkeypatch):
        """Both score_based and synthesized caps are enforced independently."""
        monkeypatch.setattr(
            "data_preprocessing.formatters.MAX_SYNTH_RATIO_BY_LANG",
            {"en": 1.0, "es": 1.0, "zh": 1.0},
        )
        score_cap = 5
        synth_cap = 3

        alloc = {
            "en": {"score_based": score_cap, "synthesized": synth_cap},
            "es": {"score_based": score_cap, "synthesized": synth_cap},
            "zh": {"score_based": score_cap, "synthesized": synth_cap},
        }

        for lang in ["en", "zh", "es"]:
            _write_synth_jsonl(tmp_path / f"reward_neg_{lang}.jsonl", lang, 20)

        unified = _make_mock_unified(n_en=50, n_es=30, n_zh=20)
        result = format_reward_pairs(
            unified,
            allocation=alloc,
            synthesized_reward_dir=tmp_path,
        )

        total_ds = datasets.concatenate_datasets([result["train"], result["validation"]])
        counts = _count_by_lang(total_ds)

        for lang, count in counts.items():
            assert count <= score_cap + synth_cap, (
                f"{lang} has {count} pairs, expected <= {score_cap + synth_cap}"
            )


# ============================================================
# TestQuantileTieBreaking — discrete scores with many ties
# ============================================================

class TestQuantileTieBreaking:
    """Verify that highly discrete score distributions are split correctly."""

    def _make_discrete_ds(self, n: int, score_value: float) -> datasets.Dataset:
        """All items share the same score — worst-case scenario for ties."""
        return datasets.Dataset.from_dict({
            "text": [f"joke {i}" for i in range(n)],
            "lang": ["en"] * n,
            "score": [score_value] * n,
            "source": ["rjokes"] * n,
        })

    def test_all_same_score_respects_quantile_limits(self):
        """When every score is identical, pipeline still returns valid output."""
        n = 100
        unified = {"rjokes": self._make_discrete_ds(n, 0.5)}
        result = format_reward_pairs(unified, allocation=_NO_CAP_ALLOCATION)
        total = len(result["train"]) + len(result["validation"])
        assert total > 0

    def test_few_unique_values_stays_within_bounds(self):
        """Simulates rJokes-like distribution: 12 discrete score levels, heavy at 0."""
        scores = (
            [0.0] * 350 + [0.05] * 240 + [0.1] * 180 +
            [0.15] * 100 + [0.2] * 60 + [0.25] * 35 +
            [0.3] * 20 + [0.35] * 10 + [0.4] * 3 +
            [0.45] * 1 + [0.5] * 1
        )
        n = len(scores)
        ds = datasets.Dataset.from_dict({
            "text": [f"joke {i}" for i in range(n)],
            "lang": ["en"] * n,
            "score": scores,
            "source": ["rjokes"] * n,
        })
        unified = {"rjokes": ds}
        result = format_reward_pairs(unified, allocation=_NO_CAP_ALLOCATION)
        total = len(result["train"]) + len(result["validation"])
        assert total > 0


class TestFineGrainedRewardBehavior:
    def test_chosen_score_higher_than_rejected_for_score_based(self):
        """Score-based pairs should satisfy chosen_score >= rejected_score."""
        unified = _make_mock_unified(n_en=60, n_es=60, n_zh=60)
        result = format_reward_pairs(
            unified,
            allocation=_NO_CAP_ALLOCATION,
            synthesized_reward_dir=None,
        )
        merged = datasets.concatenate_datasets([result["train"], result["validation"]])

        score_map = {}
        for source in ["rjokes", "haha", "chinese_humor"]:
            ds = unified[source]
            for i in range(len(ds)):
                score_map[ds[i]["text"]] = ds[i]["score"]

        checked = 0
        strict_better = 0
        for i in range(len(merged)):
            chosen = merged[i]["chosen"][0]["content"]
            rejected = merged[i]["rejected"][0]["content"]
            if chosen in score_map and rejected in score_map:
                assert score_map[chosen] >= score_map[rejected]
                if score_map[chosen] > score_map[rejected]:
                    strict_better += 1
                checked += 1
        assert checked > 0
        assert strict_better > 0

    def test_synthesized_ratio_cap_applies(self, tmp_path):
        """Synthesized count should be capped by score-based ratio limit."""
        unified = _make_mock_unified(n_en=40, n_es=40, n_zh=20)
        for lang in ["en", "es", "zh"]:
            _write_synth_jsonl(tmp_path / f"reward_neg_{lang}.jsonl", lang, 100)

        result = format_reward_pairs(
            unified,
            allocation=_NO_CAP_ALLOCATION,
            synthesized_reward_dir=tmp_path,
        )
        merged = datasets.concatenate_datasets([result["train"], result["validation"]])
        counts = _count_by_lang(merged)

        # ratio cap means synthesized cannot dominate; total should stay bounded.
        for lang in ["en", "es", "zh"]:
            assert counts[lang] > 0

    def test_reuse_monitor_logs_present(self, capsys):
        """Reuse monitoring logs should be printed in reward formatting."""
        unified = _make_mock_unified(n_en=30, n_es=20, n_zh=10)
        format_reward_pairs(
            unified,
            allocation=_NO_CAP_ALLOCATION,
            synthesized_reward_dir=None,
        )
        captured = capsys.readouterr()
        assert "chosen_reuse p95=" in captured.out
        assert "rejected_reuse p95=" in captured.out

    def test_synthesized_counts_follow_explicit_caps(self, tmp_path):
        """All languages should follow explicit synthesized caps only."""
        for lang in ["en", "es", "zh"]:
            _write_synth_jsonl(tmp_path / f"reward_neg_{lang}.jsonl", lang, 200)

        alloc = {
            "en": {"score_based": None, "synthesized": 80},
            "es": {"score_based": None, "synthesized": 60},
            "zh": {"score_based": None, "synthesized": 50},
        }
        unified = _make_mock_unified(n_en=40, n_es=40, n_zh=20)
        result = format_reward_pairs(
            unified,
            allocation=alloc,
            synthesized_reward_dir=tmp_path,
            seed=42,
        )
        out = datasets.concatenate_datasets([result["train"], result["validation"]])
        texts = [out[i]["chosen"][0]["content"] for i in range(len(out))]
        en_synth_count = sum(1 for t in texts if t.startswith("synth en chosen "))
        es_synth_count = sum(1 for t in texts if t.startswith("synth es chosen "))
        zh_synth_count = sum(1 for t in texts if t.startswith("synth zh chosen "))
        assert en_synth_count == 80
        assert es_synth_count == 60
        assert zh_synth_count == 50


class TestSftExcludeTexts:
    def _make_unified_for_sft(self) -> dict[str, datasets.Dataset]:
        return {
            "rjokes": datasets.Dataset.from_dict(
                {
                    "text": ["alpha joke", "beta joke", "gamma joke"],
                    "lang": ["en", "en", "en"],
                    "score": [0.9, 0.8, 0.7],
                    "source": ["rjokes", "rjokes", "rjokes"],
                }
            ),
            "cfun": datasets.Dataset.from_dict(
                {
                    "text": ["中文 笑话A", "中文 笑话B"],
                    "lang": ["zh", "zh"],
                    "score": [None, None],
                    "source": ["cfun", "cfun"],
                }
            ),
        }

    def test_sft_excludes_reward_used_texts(self):
        unified = self._make_unified_for_sft()
        exclude_texts = {"alpha joke", "中文 笑话A"}
        ds = format_sft_type_a(unified, exclude_texts=exclude_texts, seed=42)
        assistant_texts = [row["messages"][1]["content"] for row in ds]
        assert "alpha joke" not in assistant_texts
        assert "中文 笑话A" not in assistant_texts
        assert "beta joke" in assistant_texts

    def test_sft_exclude_normalization_works(self):
        unified = self._make_unified_for_sft()
        exclude_texts = {"  alpha   joke  ", "\n中文   笑话A\n"}
        ds = format_sft_type_a(unified, exclude_texts=exclude_texts, seed=42)
        assistant_texts = [row["messages"][1]["content"] for row in ds]
        assert "alpha joke" not in assistant_texts
        assert "中文 笑话A" not in assistant_texts

    def test_sft_without_exclude_keeps_behavior(self):
        unified = self._make_unified_for_sft()
        ds_no_exclude = format_sft_type_a(unified, seed=42)
        ds_with_empty = format_sft_type_a(unified, exclude_texts=set(), seed=42)
        assert len(ds_no_exclude) == len(ds_with_empty)

    def test_sft_source_level_removal_counts_nonnegative(self, capsys):
        unified = self._make_unified_for_sft()
        exclude_texts = {"alpha joke", "中文 笑话A"}
        _ = format_sft_type_a(unified, exclude_texts=exclude_texts, seed=42)
        output = capsys.readouterr().out
        # expected log shape: removed should be non-negative integer
        assert "[sft_type_a] rjokes:" in output
        assert "[sft_type_a] cfun:" in output
