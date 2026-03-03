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

class TestFormatRewardPairsDownsample:
    def test_no_new_params_preserves_behavior(self):
        """With new params as None, behavior matches original."""
        unified = _make_mock_unified(n_en=50, n_es=30, n_zh=20)
        result = format_reward_pairs(
            unified,
            max_pairs_per_lang=None,
            synthesized_reward_dir=None,
        )
        assert "train" in result
        assert "validation" in result
        total = len(result["train"]) + len(result["validation"])
        assert total > 0

    def test_max_pairs_per_lang_caps_each_language(self):
        """Each language should have at most max_pairs_per_lang pairs."""
        cap = 5
        unified = _make_mock_unified(n_en=50, n_es=30, n_zh=20)
        result = format_reward_pairs(unified, max_pairs_per_lang=cap)
        total_ds = datasets.concatenate_datasets([result["train"], result["validation"]])
        counts = _count_by_lang(total_ds)
        for lang, count in counts.items():
            assert count <= cap, f"{lang} has {count} pairs, expected <= {cap}"

    def test_max_pairs_per_lang_none_means_no_cap(self):
        """None means no truncation."""
        unified = _make_mock_unified(n_en=100, n_es=60, n_zh=40)
        result_no_cap = format_reward_pairs(unified, max_pairs_per_lang=None)
        total_no_cap = len(result_no_cap["train"]) + len(result_no_cap["validation"])

        result_with_cap = format_reward_pairs(unified, max_pairs_per_lang=5)
        total_with_cap = len(result_with_cap["train"]) + len(result_with_cap["validation"])

        assert total_no_cap > total_with_cap

    def test_max_pairs_larger_than_data_keeps_all(self):
        """When cap is larger than available data, all pairs are kept."""
        unified = _make_mock_unified(n_en=20, n_es=10, n_zh=10)
        result_huge_cap = format_reward_pairs(unified, max_pairs_per_lang=100000)
        result_no_cap = format_reward_pairs(unified, max_pairs_per_lang=None)

        total_huge = len(result_huge_cap["train"]) + len(result_huge_cap["validation"])
        total_none = len(result_no_cap["train"]) + len(result_no_cap["validation"])
        assert total_huge == total_none


# ============================================================
# TestFormatRewardPairsSynthMerge — synthesized_reward_dir
# ============================================================

class TestFormatRewardPairsSynthMerge:
    def test_synth_dir_loads_and_merges(self, tmp_path):
        """Synthesized JSONL files are loaded and merged into output."""
        n_synth = 5
        for lang in ["en", "zh", "es"]:
            _write_synth_jsonl(tmp_path / f"reward_neg_{lang}.jsonl", lang, n_synth)

        unified = _make_mock_unified(n_en=20, n_es=10, n_zh=10)
        result_with_synth = format_reward_pairs(
            unified, synthesized_reward_dir=tmp_path,
        )
        result_without_synth = format_reward_pairs(
            unified, synthesized_reward_dir=None,
        )

        total_with = len(result_with_synth["train"]) + len(result_with_synth["validation"])
        total_without = len(result_without_synth["train"]) + len(result_without_synth["validation"])

        assert total_with == total_without + n_synth * 3

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
# TestFormatRewardPairsCombined — both params together
# ============================================================

class TestFormatRewardPairsCombined:
    def test_cap_then_merge(self, tmp_path):
        """Downsampling + synthesis merge produces expected count."""
        cap = 5
        n_synth = 3

        for lang in ["en", "zh", "es"]:
            _write_synth_jsonl(tmp_path / f"reward_neg_{lang}.jsonl", lang, n_synth)

        unified = _make_mock_unified(n_en=50, n_es=30, n_zh=20)
        result = format_reward_pairs(
            unified,
            max_pairs_per_lang=cap,
            synthesized_reward_dir=tmp_path,
        )

        total_ds = datasets.concatenate_datasets([result["train"], result["validation"]])
        counts = _count_by_lang(total_ds)

        for lang, count in counts.items():
            assert count <= cap + n_synth, (
                f"{lang} has {count} pairs, expected <= {cap + n_synth}"
            )
