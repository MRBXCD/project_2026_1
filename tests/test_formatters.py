"""Tests for data_preprocessing/formatters.py — format_reward_pairs() new parameters."""

import json
import re

import datasets
import pytest

from data_preprocessing.formatters import format_reward_pairs


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
        """When cap is larger than available data, all pairs are kept."""
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
        assert total_huge == total_none


# ============================================================
# TestFormatRewardPairsSynthMerge — synthesized_reward_dir
# ============================================================

class TestFormatRewardPairsSynthMerge:
    def test_synth_dir_loads_and_merges(self, tmp_path):
        """Synthesized JSONL files are loaded and merged into output."""
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

    def test_synth_cap_limits_loaded_pairs(self, tmp_path):
        """Synthesized cap should limit the number of synthesized pairs loaded."""
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

    def test_output_schema_after_merge(self, tmp_path):
        """Schema is correct after merging synthesized data."""
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
    def test_allocation_caps_both_types(self, tmp_path):
        """Both score_based and synthesized caps are enforced independently."""
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
        """When every score is identical, high+low should still be ~60%, not 100%."""
        n = 100
        unified = {"rjokes": self._make_discrete_ds(n, 0.5)}
        result = format_reward_pairs(unified, allocation=_NO_CAP_ALLOCATION)
        total = len(result["train"]) + len(result["validation"])
        # bottom 30% + top 30% = 60 items; pairs = min(chosen*3, rejected)
        # with 30 high (×3=90 chosen pool) and 30 low (=30 rejected) → 30 pairs
        assert total == 30

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
        n_low = int(n * 0.3)   # 300
        n_high = n - int(n * 0.7)  # 300
        max_expected = min(n_high * 3, n_low)
        assert total <= max_expected
        assert total > 0
