"""Tests for rl/rewards.py"""

from unittest.mock import MagicMock

import pytest

from rl.rewards import (
    REWARD_EMPTY,
    REWARD_FORMAT_PASS,
    REWARD_TOO_LONG,
    REWARD_TOO_SHORT,
    REWARD_REPETITIVE,
    RELEVANCE_REWARD_MAX,
    RELEVANCE_REWARD_MIN,
    WEIGHT_FORMAT,
    WEIGHT_KEYWORD,
    WEIGHT_RELEVANCE,
    WEIGHT_HUMOR,
    _extract_headline_tokens,
    build_reward_fn,
    compute_reward,
    reward_format,
    reward_humor,
    reward_keyword,
    reward_relevance,
)


# ============================================================
# Helper: build GRPOTrainer-style conversational messages
# ============================================================

def _make_prompt(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def _make_completion(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


# ============================================================
# 1. reward_format
# ============================================================

class TestRewardFormat:
    def test_empty_string(self):
        assert reward_format("") == REWARD_EMPTY

    def test_whitespace_only(self):
        assert reward_format("   \n\t  ") == REWARD_EMPTY

    def test_too_short(self):
        result = reward_format("short")
        assert result == pytest.approx(REWARD_FORMAT_PASS + REWARD_TOO_SHORT)

    def test_too_long(self):
        # Use a long but non-repetitive text to avoid triggering repetition penalty
        text = " ".join(f"word{i}" for i in range(100))  # ~600 chars, all unique
        result = reward_format(text)
        assert result == pytest.approx(REWARD_FORMAT_PASS + REWARD_TOO_LONG)

    def test_repetitive(self):
        result = reward_format("ha ha ha ha ha ha ha ha ha ha ha ha")
        assert result == pytest.approx(REWARD_FORMAT_PASS + REWARD_REPETITIVE)

    def test_too_long_and_repetitive(self):
        text = "ha " * 200  # > 280 chars and highly repetitive
        result = reward_format(text)
        assert result == pytest.approx(
            REWARD_FORMAT_PASS + REWARD_TOO_LONG + REWARD_REPETITIVE
        )

    def test_normal_text(self):
        result = reward_format(
            "Why did the chicken cross the road? To get away from the AI."
        )
        assert result == pytest.approx(REWARD_FORMAT_PASS)


# ============================================================
# 2. reward_keyword
# ============================================================

class TestRewardKeyword:
    def test_empty_keywords(self):
        assert reward_keyword("any text here", []) == 0.0

    def test_all_keywords_present(self):
        result = reward_keyword(
            "The penguin filed for bankruptcy.",
            ["penguin", "bankruptcy"],
        )
        assert result == pytest.approx(2 * 1.0 + 0.5)

    def test_partial_keywords(self):
        result = reward_keyword(
            "A funny joke about penguins.",
            ["penguin", "bankruptcy"],
        )
        assert result == pytest.approx(1 * 1.0 + (-0.5))

    def test_no_keywords_present(self):
        result = reward_keyword(
            "A random joke with no matching words.",
            ["penguin", "bankruptcy"],
        )
        assert result == pytest.approx(-1.0)

    def test_case_insensitive(self):
        result = reward_keyword(
            "The PENGUIN filed for BANKRUPTCY.",
            ["penguin", "bankruptcy"],
        )
        assert result == pytest.approx(2 * 1.0 + 0.5)


# ============================================================
# 3. reward_relevance
# ============================================================

class TestRewardRelevance:
    def test_empty_headline(self):
        assert reward_relevance("", "Any joke response") == 0.0

    def test_zero_overlap(self):
        result = reward_relevance(
            "Tech Giants Face AI Safety Regulations",
            "Something completely unrelated xyz abc",
        )
        assert result == pytest.approx(RELEVANCE_REWARD_MIN)

    def test_full_overlap(self):
        result = reward_relevance(
            "Tech Giants Face AI Safety Regulations",
            "tech giants face new ai safety regulations announced today",
        )
        assert result == pytest.approx(RELEVANCE_REWARD_MIN)

    def test_target_overlap_peak(self):
        # Construct a case near the 30% target
        # Headline tokens: {"tech", "giants", "face", "safety", "regulations"} = 5 tokens
        # ~30% = ~1.5 tokens -> 1 or 2 tokens hit
        result = reward_relevance(
            "Tech Giants Face Safety Regulations",
            "The tech industry is changing rapidly with new innovations",
        )
        # "tech" hits (1 out of 5 = 20%), close to target
        assert result > 0.0  # Should be positive (between 0% and target)

    def test_chinese_bigram(self):
        result = reward_relevance(
            "苹果公司面临新的安全法规",
            "苹果发布了新的产品",
        )
        # "苹果" bigram should match
        assert result != 0.0  # Should have non-zero relevance


# ============================================================
# 4. reward_humor
# ============================================================

class TestRewardHumor:
    def test_no_scorer(self):
        assert reward_humor("prompt", "response", scorer=None) == 0.0

    def test_scorer_returns_value(self):
        scorer = lambda p, r: 0.8
        result = reward_humor("prompt", "response", scorer=scorer)
        assert result == pytest.approx(0.8)

    def test_scorer_clamped(self):
        scorer = lambda p, r: 5.0
        result = reward_humor("prompt", "response", scorer=scorer)
        assert result == pytest.approx(1.0)

    def test_scorer_exception(self):
        scorer = MagicMock(side_effect=RuntimeError("API down"))
        result = reward_humor("prompt", "response", scorer=scorer)
        assert result == 0.0


# ============================================================
# 5. compute_reward
# ============================================================

class TestComputeReward:
    def test_phase1_no_scorer(self):
        result = compute_reward(
            prompt_text="Tell a joke",
            response_text="Why did the chicken cross the road? To avoid the AI.",
            keywords=[],
            headline="",
        )
        expected = WEIGHT_FORMAT * REWARD_FORMAT_PASS
        assert result == pytest.approx(expected)

    def test_short_circuit_on_empty(self):
        result = compute_reward(
            prompt_text="Tell a joke",
            response_text="",
            keywords=["penguin"],
            headline="Big News",
            humor_scorer=lambda p, r: 1.0,
        )
        assert result == REWARD_EMPTY

    def test_humor_score_override_takes_priority(self):
        single_scorer = MagicMock(return_value=0.9)
        result = compute_reward(
            prompt_text="p",
            response_text="A perfectly normal joke about things and stuff here.",
            humor_scorer=single_scorer,
            humor_score_override=0.3,
        )
        single_scorer.assert_not_called()
        assert WEIGHT_HUMOR * 0.3 == pytest.approx(
            result
            - WEIGHT_FORMAT * REWARD_FORMAT_PASS
            - WEIGHT_KEYWORD * 0.0
            - WEIGHT_RELEVANCE * reward_relevance("", "A perfectly normal joke about things and stuff here."),
        )

    def test_full_reward_with_headline_and_keywords(self):
        result = compute_reward(
            prompt_text="Write a joke about tech",
            response_text="Tech giants now require penguin bankruptcy insurance.",
            keywords=["penguin", "bankruptcy"],
            headline="Tech Giants News",
        )
        assert isinstance(result, float)
        # Should include positive keyword and some relevance
        assert result > WEIGHT_FORMAT * REWARD_FORMAT_PASS


# ============================================================
# 6. build_reward_fn — batch tests
# ============================================================

class TestBuildRewardFn:
    def _make_batch(self, texts: list[str]):
        """Helper: create prompts/completions/keywords/headline for a batch."""
        prompts = [_make_prompt("Write a joke") for _ in texts]
        completions = [_make_completion(t) for t in texts]
        keywords = [[] for _ in texts]
        headline = ["Some News Headline" for _ in texts]
        return prompts, completions, keywords, headline

    def test_phase1_basic(self):
        fn = build_reward_fn()
        prompts, completions, keywords, headline = self._make_batch(
            ["A valid joke about something interesting here.", "Another good joke for testing purposes."]
        )
        rewards = fn(prompts=prompts, completions=completions,
                     keywords=keywords, headline=headline)
        assert len(rewards) == 2
        assert all(isinstance(r, float) for r in rewards)

    def test_batch_scorer_called_once(self):
        mock_batch = MagicMock(return_value=[0.5, 0.5, 0.5])
        fn = build_reward_fn(batch_humor_scorer=mock_batch)
        prompts, completions, keywords, headline = self._make_batch(
            ["Joke one is pretty funny.", "Joke two is also good.", "Joke three here."]
        )
        fn(prompts=prompts, completions=completions,
           keywords=keywords, headline=headline)
        mock_batch.assert_called_once()

    def test_batch_scorer_receives_plain_text(self):
        mock_batch = MagicMock(return_value=[0.5])
        fn = build_reward_fn(batch_humor_scorer=mock_batch)
        prompts = [_make_prompt("Write a joke")]
        completions = [_make_completion("A perfectly fine joke about this and that.")]
        fn(prompts=prompts, completions=completions,
           keywords=[[]], headline=["News"])

        called_prompts, called_responses = mock_batch.call_args[0]
        assert isinstance(called_prompts[0], str)
        assert isinstance(called_responses[0], str)
        assert called_prompts[0] == "Write a joke"
        assert called_responses[0] == "A perfectly fine joke about this and that."

    def test_batch_scorer_skips_degenerate(self):
        received_responses = []

        def capture_batch(prompts, responses):
            received_responses.extend(responses)
            return [0.5] * len(responses)

        fn = build_reward_fn(batch_humor_scorer=capture_batch)
        prompts = [_make_prompt("p1"), _make_prompt("p2"), _make_prompt("p3")]
        completions = [
            _make_completion("A valid joke response number one here."),
            _make_completion(""),  # empty -> degenerate
            _make_completion("Another valid joke response for testing."),
        ]
        fn(prompts=prompts, completions=completions,
           keywords=[[], [], []], headline=["h1", "h2", "h3"])

        # Batch scorer should NOT receive the empty completion
        assert len(received_responses) == 2
        assert "" not in received_responses

    def test_degenerate_gets_short_circuit_reward(self):
        mock_batch = MagicMock(return_value=[0.5])
        fn = build_reward_fn(batch_humor_scorer=mock_batch)
        prompts = [_make_prompt("p1"), _make_prompt("p2")]
        completions = [
            _make_completion("A valid joke that is long enough to pass."),
            _make_completion(""),
        ]
        rewards = fn(prompts=prompts, completions=completions,
                     keywords=[[], []], headline=["h", "h"])

        assert rewards[1] == REWARD_EMPTY

    def test_batch_scorer_priority_over_single(self):
        single = MagicMock(return_value=0.9)
        batch = MagicMock(return_value=[0.1])
        fn = build_reward_fn(humor_scorer=single, batch_humor_scorer=batch)
        prompts = [_make_prompt("p")]
        completions = [_make_completion("A valid long joke for testing batch priority.")]
        rewards = fn(prompts=prompts, completions=completions,
                     keywords=[[]], headline=["h"])

        batch.assert_called_once()
        single.assert_not_called()
        # humor contribution = WEIGHT_HUMOR * 0.1, not 0.9
        r_format = reward_format("A valid long joke for testing batch priority.")
        r_relevance = reward_relevance("h", "A valid long joke for testing batch priority.")
        expected = (
            WEIGHT_FORMAT * r_format
            + WEIGHT_KEYWORD * 0.0
            + WEIGHT_RELEVANCE * r_relevance
            + WEIGHT_HUMOR * 0.1
        )
        assert rewards[0] == pytest.approx(expected)

    def test_none_keywords_and_headline(self):
        fn = build_reward_fn()
        prompts = [_make_prompt("p")]
        completions = [_make_completion("A valid joke response about something fun.")]
        rewards = fn(prompts=prompts, completions=completions,
                     keywords=None, headline=None)
        assert len(rewards) == 1
        assert isinstance(rewards[0], float)

    def test_return_length_matches_input(self):
        fn = build_reward_fn()
        n = 10
        prompts = [_make_prompt(f"prompt {i}") for i in range(n)]
        completions = [_make_completion(f"A valid joke number {i} for testing.") for i in range(n)]
        rewards = fn(prompts=prompts, completions=completions,
                     keywords=[[] for _ in range(n)],
                     headline=[f"headline {i}" for i in range(n)])
        assert len(rewards) == n


# ============================================================
# 7. _extract_headline_tokens
# ============================================================

class TestExtractHeadlineTokens:
    def test_english(self):
        tokens = _extract_headline_tokens("Tech Giants Face New AI Safety Regulations")
        # "new" (3 chars) is kept; "AI" (2 chars) is filtered out
        assert "tech" in tokens
        assert "giants" in tokens
        assert "regulations" in tokens
        assert "ai" not in tokens  # length < 3

    def test_chinese(self):
        tokens = _extract_headline_tokens("苹果公司面临安全法规")
        assert "苹果" in tokens
        assert "公司" in tokens
        assert "法规" in tokens

    def test_mixed(self):
        # Spaces between Chinese and English for proper whitespace splitting
        tokens = _extract_headline_tokens("苹果公司发布 iPhone 新品")
        has_cjk = any("\u4e00" <= c <= "\u9fff" for t in tokens for c in t)
        has_latin = any(t.isascii() for t in tokens)
        assert has_cjk   # CJK bigrams from "苹果公司发布"
        assert has_latin  # "iphone" from whitespace split (len >= 3)
