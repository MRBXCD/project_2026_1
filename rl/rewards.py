"""
Reward Functions for GRPO Training
===================================

This module defines the reward functions used in GRPO (Group Relative Policy
Optimization) training. The reward signal guides the model to generate outputs
that satisfy both hard constraints (format, keywords) and soft quality goals
(humor).

Design Overview:
    The composite reward is decomposed into independent sub-rewards, each
    measuring one aspect of generation quality:

    1. reward_format     — Format compliance (hard constraint, rule-based)
    2. reward_keyword    — Keyword inclusion (hard constraint, rule-based)
    3. reward_relevance  — Headline relevance (soft constraint, word overlap)
    4. reward_humor      — Humor quality (soft constraint, Phase 2 placeholder)

    These sub-rewards are combined by compute_reward() with configurable
    weights. The final build_reward_fn() wraps everything into the signature
    expected by TRL's GRPOTrainer.

Phased Training Strategy:
    Phase 1: Pure rule-based reward (format + keyword). humor_scorer=None.
    Phase 2: Add humor scoring (LLM-as-Judge or trained Reward Model).

GRPOTrainer Integration (TRL 0.27.1):
    GRPOTrainer calls custom reward functions with the following signature:

        reward_fn(
            prompts=prompts,                # list[list[dict]]: each element is a conversation
                                            #   e.g. [[{"role": "user", "content": "..."}], ...]
            completions=completions,        # list[list[dict]]: each element is assistant turn(s)
                                            #   e.g. [[{"role": "assistant", "content": "..."}], ...]
            completion_ids=completion_ids,  # list[list[int]]: token IDs of completions
            **reward_kwargs,                # Extra dataset columns + trainer_state
        ) -> list[float]

    Extra dataset columns (e.g. "headline", "keywords") from the training
    dataset are automatically passed via **reward_kwargs. The "prompt" and
    "completion" columns are excluded from kwargs since they are passed as
    positional-like keyword args.

Usage:
    from rl.rewards import build_reward_fn

    # Phase 1: rule-based only
    reward_fn = build_reward_fn(humor_scorer=None)

    # Phase 2: with humor scoring
    reward_fn = build_reward_fn(humor_scorer=my_humor_scorer)

Dependencies:
    - re (Standard library)
    - typing (Standard library)
"""

import re
from typing import Any, Callable


# ============================================================
# Constants — Reward Thresholds and Weights
# ============================================================

# --- Format Reward Constants ---
# Minimum acceptable response length (in characters).
# Responses shorter than this are considered degenerate or empty.
FORMAT_MIN_LENGTH = 10

# Maximum acceptable response length (in characters).
# Jokes should be concise; overly long responses are penalized.
FORMAT_MAX_LENGTH = 280

# Minimum ratio of unique trigrams to total trigrams.
# Below this threshold, the response is considered repetitive/degenerate.
# e.g., "ha ha ha ha ha" has very low unique trigram ratio.
TRIGRAM_UNIQUENESS_THRESHOLD = 0.5

# --- Format Reward Values ---
# Reward for an empty or whitespace-only response.
REWARD_EMPTY = -2.0

# Reward for a response that is too short (< FORMAT_MIN_LENGTH).
REWARD_TOO_SHORT = -1.0

# Reward for a response that is too long (> FORMAT_MAX_LENGTH).
REWARD_TOO_LONG = -0.5

# Reward for a response with excessive repetition (n-gram degeneracy).
REWARD_REPETITIVE = -1.5

# Base reward for a format-compliant response.
REWARD_FORMAT_PASS = 0.5

# --- Keyword Reward Values ---
# Reward when the prompt has keyword constraints but none are found in response.
REWARD_NO_KEYWORD_HIT = -1.0

# Reward per keyword hit.
REWARD_PER_KEYWORD = 1.0

# Bonus reward when ALL keywords are present.
REWARD_ALL_KEYWORDS_BONUS = 0.5

# Partial hit penalty (some keywords present but not all).
REWARD_PARTIAL_KEYWORD_PENALTY = -0.5

# --- Relevance Reward Constants ---
# Minimum word length to be considered a "content word" in EN/ES.
# Filters out stopwords like "a", "is", "the", "de", "en", "el".
RELEVANCE_MIN_WORD_LENGTH = 3

# Optimal overlap ratio between headline tokens and response.
# A joke "inspired by" a headline should reference some of its content
# (~30% overlap) but also introduce new elements (punchline, twist).
# - Too low overlap (→0%): response is unrelated to the headline.
# - Too high overlap (→100%): response is just paraphrasing the headline.
RELEVANCE_TARGET_OVERLAP = 0.3

# Reward range for the relevance score.
RELEVANCE_REWARD_MAX = 0.5   # At target overlap
RELEVANCE_REWARD_MIN = -0.5  # At 0% or 100% overlap

# --- Composite Reward Weights ---
# These weights control the relative importance of each sub-reward
# in the composite score. They are key hyperparameters for GRPO training.
WEIGHT_FORMAT = 1.0
WEIGHT_KEYWORD = 2.0    # Higher weight: keyword inclusion is a hard constraint
WEIGHT_RELEVANCE = 0.5  # Lower weight: word overlap is a noisy proxy for relevance
WEIGHT_HUMOR = 1.5


# ============================================================
# Sub-reward 1: Format Compliance
# ============================================================

def reward_format(text: str) -> float:
    """Evaluate format compliance of a generated response.

    Checks three aspects of format quality:
    1. Non-empty: the response must contain meaningful text.
    2. Length bounds: the response should be between FORMAT_MIN_LENGTH and
       FORMAT_MAX_LENGTH characters. Jokes are short-form text; excessively
       long or short outputs indicate failure.
    3. Repetition check: detects n-gram level degeneracy where the model
       produces repetitive tokens (a common failure mode in RL training).

    Scoring uses additive accumulation: starts from a base "pass" score
    (REWARD_FORMAT_PASS = 0.5), then each detected violation adds its
    penalty. This allows compound failures (e.g., too long AND repetitive)
    to produce a more negative score than either violation alone.

    Only truly empty responses get an early return, since there is nothing
    to evaluate for them.

    Args:
        text: The raw generated response text (before any post-processing).

    Returns:
        float: Reward score. Starts from REWARD_FORMAT_PASS (0.5) and
            accumulates penalties:
            - REWARD_EMPTY (-2.0): early return for empty/whitespace-only
            - REWARD_TOO_SHORT (-1.0): penalty added if below FORMAT_MIN_LENGTH
            - REWARD_TOO_LONG (-0.5): penalty added if above FORMAT_MAX_LENGTH
            - REWARD_REPETITIVE (-1.5): penalty added if trigram uniqueness
              is below TRIGRAM_UNIQUENESS_THRESHOLD
            Worst case (too long + repetitive): 0.5 + (-0.5) + (-1.5) = -1.5
            Perfect case: 0.5 (no penalties)

    Example:
        >>> reward_format("")
        -2.0
        >>> reward_format("ha " * 100)  # too long + repetitive
        -2.0  # -0.5 + (-1.5)
        >>> reward_format("Why did the chicken cross the road? To get away from the AI.")
        0.5
    """
    # Check 1: empty text — nothing to evaluate, early return.
    if not text or not text.strip():
        return REWARD_EMPTY

    stripped = text.strip()

    # Accumulate penalty from each independent check.
    # Start from the base "pass" reward; each violation subtracts from it.
    score = REWARD_FORMAT_PASS

    # Check 2a: too short
    if len(stripped) < FORMAT_MIN_LENGTH:
        score += REWARD_TOO_SHORT

    # Check 2b: too long
    if len(stripped) > FORMAT_MAX_LENGTH:
        score += REWARD_TOO_LONG

    # Check 3: n-gram repetition degeneracy
    words = stripped.split()
    if len(words) >= 4:
        trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
        unique_ratio = len(set(trigrams)) / len(trigrams)
        if unique_ratio < TRIGRAM_UNIQUENESS_THRESHOLD:
            score += REWARD_REPETITIVE

    return score


# ============================================================
# Sub-reward 2: Keyword Inclusion
# ============================================================

def reward_keyword(text: str, keywords: list[str]) -> float:
    """Evaluate whether required keywords are present in the response.

    For prompts with keyword constraints (the "keyword subtask" in SemEval),
    the generated joke must naturally include all specified keywords. This
    function checks for case-insensitive keyword presence.

    When no keywords are required (empty list), this function returns 0.0
    to avoid biasing the composite reward.

    Scoring logic:
        - No keywords required    -> 0.0 (neutral, no constraint)
        - All keywords present    -> N * REWARD_PER_KEYWORD + REWARD_ALL_KEYWORDS_BONUS
        - Some keywords present   -> hits * REWARD_PER_KEYWORD + REWARD_PARTIAL_KEYWORD_PENALTY
        - No keywords present     -> REWARD_NO_KEYWORD_HIT (-1.0)

    Args:
        text: The generated response text.
        keywords: List of required keywords. Can be empty (no constraint).
            Typically contains 2 words for the SemEval keyword subtask.

    Returns:
        float: Reward score.
            - 0.0 when keywords list is empty (no constraint)
            - Positive when keywords are found
            - Negative when required keywords are missing

    Example:
        >>> reward_keyword("The penguin filed for bankruptcy.", ["penguin", "bankruptcy"])
        2.5  # 2 * 1.0 + 0.5
        >>> reward_keyword("A funny joke about penguins.", ["penguin", "bankruptcy"])
        0.5  # 1 * 1.0 + (-0.5)
        >>> reward_keyword("A random joke.", ["penguin", "bankruptcy"])
        -1.0
        >>> reward_keyword("Any text here.", [])
        0.0
    """
    if not keywords:
        return 0.0

    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)

    if hits == 0:
        return REWARD_NO_KEYWORD_HIT
    elif hits == len(keywords):
        return hits * REWARD_PER_KEYWORD + REWARD_ALL_KEYWORDS_BONUS
    else:
        return hits * REWARD_PER_KEYWORD + REWARD_PARTIAL_KEYWORD_PENALTY


# ============================================================
# Sub-reward 3: Headline Relevance
# ============================================================

def _extract_headline_tokens(headline: str) -> set[str]:
    """Extract meaningful tokens from a headline for relevance checking.

    Uses a language-agnostic strategy that handles EN, ZH, and ES:

    - For space-separated languages (EN/ES): split by whitespace, strip
      punctuation, keep words with length >= RELEVANCE_MIN_WORD_LENGTH.
      This naturally filters out stopwords ("a", "the", "de", "en").

    - For CJK text (ZH): whitespace splitting doesn't segment Chinese.
      Instead, extract character bigrams (2-char windows), which are
      the minimal meaningful units in Chinese (e.g., "苹果", "公司").
      Single characters are too ambiguous for relevance checking.

    - For mixed text (e.g., Chinese headline with English words): both
      strategies apply simultaneously, capturing both types of tokens.

    Args:
        headline: The news headline text.

    Returns:
        set[str]: Lowercased content tokens extracted from the headline.
            Empty set if headline is empty or yields no tokens.
    """
    headline_lower = headline.lower().strip()
    if not headline_lower:
        return set()

    tokens = set()

    # Strategy 1: whitespace-split words for EN/ES
    for word in headline_lower.split():
        cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", word)
        if len(cleaned) >= RELEVANCE_MIN_WORD_LENGTH:
            tokens.add(cleaned)

    # Strategy 2: CJK character bigrams for ZH
    cjk_chars = [c for c in headline_lower if "\u4e00" <= c <= "\u9fff"]
    for i in range(len(cjk_chars) - 1):
        tokens.add(cjk_chars[i] + cjk_chars[i + 1])

    return tokens


def reward_relevance(headline: str, response: str) -> float:
    """Evaluate whether the response is relevant to the news headline.

    Computes token overlap between headline and response as a proxy for
    topical relevance. This is a lightweight rule-based approach (Phase 1).
    In Phase 2, this can be replaced or supplemented by LLM-as-Judge
    relevance scoring for higher accuracy.

    Limitations (documented for transparency):
        - Word overlap is a noisy proxy: a joke can be thematically related
          to a headline without sharing any words (e.g., headline about
          "Tech Giants" → joke about "Silicon Valley").
        - Conversely, high word overlap doesn't guarantee relevance (shared
          common words may be coincidental).
        - Because of these limitations, WEIGHT_RELEVANCE is set low (0.5)
          to avoid over-penalizing creative but relevant responses.

    Scoring (triangular curve, peak at RELEVANCE_TARGET_OVERLAP = 0.3):
        overlap_ratio = (headline tokens found in response) / (total headline tokens)

        The reward peaks at the target overlap and decreases linearly
        in both directions:

        - Left side [0, target]:  linearly increases from MIN to MAX.
          Going from "totally unrelated" to "appropriately related"
          is rewarded.
        - Right side [target, 1]: linearly decreases from MAX to MIN.
          Going from "appropriately related" to "just paraphrasing
          the headline" is penalized.

        This captures the insight that a good joke should reference
        some headline elements but not all — it needs room for the
        creative twist that makes it funny.

        Concrete values (with target=0.3):
        - 0% overlap  → -0.5 (completely unrelated)
        - 30% overlap → +0.5 (ideal: topically grounded + creative)
        - 100% overlap → -0.5 (just echoing the headline)

    Args:
        headline: The news headline text. Can be empty for the keyword
            subtask (no headline provided), in which case returns 0.0.
        response: The generated response text.

    Returns:
        float: Relevance score in range [RELEVANCE_REWARD_MIN, RELEVANCE_REWARD_MAX].
            Returns 0.0 if headline is empty (no headline constraint).

    Example:
        >>> reward_relevance("Tech Giants Face AI Safety Regulations",
        ...                  "Tech safety is no joke")
        0.167   # ~2/6 tokens hit, close to 30% target
        >>> reward_relevance("Tech Giants Face AI Safety Regulations",
        ...                  "Something completely unrelated")
        -0.5    # 0% overlap
        >>> reward_relevance("", "Any joke here")
        0.0     # no headline, neutral
    """
    if not headline or not headline.strip():
        return 0.0

    headline_tokens = _extract_headline_tokens(headline)
    if not headline_tokens:
        return 0.0

    response_lower = response.lower()
    hits = sum(1 for token in headline_tokens if token in response_lower)
    overlap_ratio = hits / len(headline_tokens)

    # Triangular reward curve peaking at target overlap.
    target = RELEVANCE_TARGET_OVERLAP
    reward_range = RELEVANCE_REWARD_MAX - RELEVANCE_REWARD_MIN

    if overlap_ratio <= target:
        # Left side: [0, target] → [MIN, MAX]
        reward = RELEVANCE_REWARD_MIN + reward_range * (overlap_ratio / target)
    else:
        # Right side: [target, 1] → [MAX, MIN]
        reward = RELEVANCE_REWARD_MIN + reward_range * ((1.0 - overlap_ratio) / (1.0 - target))

    return reward


# ============================================================
# Sub-reward 4: Humor Quality (Phase 2 Placeholder)
# ============================================================

def reward_humor(
    prompt_text: str,
    response_text: str,
    scorer: Callable[[str, str], float] | None = None,
) -> float:
    """Evaluate the humor quality of a generated response.

    This is a placeholder for Phase 2 of GRPO training. In Phase 1,
    scorer is None and this function always returns 0.0.

    In Phase 2, the scorer can be either:
    - An LLM-as-Judge function: calls an external API (e.g., Gemini) to
      rate humor on a 1-5 scale, then maps to [-1.0, 1.0].
    - A trained Reward Model: a small classifier (e.g., Qwen3-1.7B +
      classification head) trained on preference pair data.

    The scorer callable should have the signature:
        scorer(prompt: str, response: str) -> float
    and return a value in the range [-1.0, 1.0].

    Args:
        prompt_text: The user prompt (plain text, not message format).
        response_text: The generated response (plain text).
        scorer: Optional callable that scores humor quality.
            If None, returns 0.0 (Phase 1 behavior).

    Returns:
        float: Humor score in range [-1.0, 1.0], or 0.0 if no scorer.
    """
    if scorer is None:
        return 0.0

    try:
        score = scorer(prompt_text, response_text)
        # Clamp to expected range for safety
        return max(-1.0, min(1.0, float(score)))
    except Exception:
        # Return neutral score on any scorer failure to avoid
        # crashing the training loop
        return 0.0


# ============================================================
# Composite Reward
# ============================================================

def compute_reward(
    prompt_text: str,
    response_text: str,
    keywords: list[str] | None = None,
    headline: str | None = None,
    humor_scorer: Callable[[str, str], float] | None = None,
) -> float:
    """Compute the composite reward for a single (prompt, response) pair.

    Combines sub-rewards with configurable weights:

        total = WEIGHT_FORMAT    * reward_format(response_text)
              + WEIGHT_KEYWORD   * reward_keyword(response_text, keywords)
              + WEIGHT_RELEVANCE * reward_relevance(headline, response_text)
              + WEIGHT_HUMOR     * reward_humor(prompt_text, response_text, humor_scorer)

    Short-circuit: if format reward is severely negative (<= -1.0),
    returns the format reward directly without computing other sub-rewards.
    This saves computation on clearly degenerate outputs and provides a
    strong negative signal for the GRPO advantage estimation.

    Args:
        prompt_text: The user prompt in plain text.
        response_text: The generated response in plain text.
        keywords: List of required keywords. None or empty list means no
            keyword constraint.
        headline: The news headline text. None or empty string means no
            headline (e.g., keyword subtask), relevance reward is 0.0.
        humor_scorer: Optional humor scoring callable (see reward_humor docs).
            None in Phase 1.

    Returns:
        float: Weighted composite reward score.

    Example:
        >>> # Phase 1: format-compliant, no keywords, no humor scorer
        >>> compute_reward("Tell a joke", "Why did the chicken cross the road?", [])
        0.5  # 1.0 * 0.5 + 2.0 * 0.0 + 0.5 * 0.0 + 1.5 * 0.0
    """
    r_format = reward_format(response_text)

    # Short-circuit: severely non-compliant format -> skip remaining checks.
    # This gives a clear negative signal and avoids wasting compute on
    # obviously bad outputs (e.g., empty strings, degenerate repetitions).
    if r_format <= -1.0:
        return r_format

    r_keyword = reward_keyword(response_text, keywords or [])
    r_relevance = reward_relevance(headline or "", response_text)
    r_humor = reward_humor(prompt_text, response_text, humor_scorer)

    total = (
        WEIGHT_FORMAT * r_format
        + WEIGHT_KEYWORD * r_keyword
        + WEIGHT_RELEVANCE * r_relevance
        + WEIGHT_HUMOR * r_humor
    )

    return total


# ============================================================
# GRPOTrainer-Compatible Reward Function Wrapper
# ============================================================

def build_reward_fn(
    humor_scorer: Callable[[str, str], float] | None = None,
) -> Callable:
    """Build a reward function compatible with TRL GRPOTrainer (v0.27.1).

    Returns a closure that adapts our compute_reward() to the signature
    expected by GRPOTrainer._calculate_rewards(). The returned function
    handles the conversion between TRL's conversational message format
    and our plain-text reward functions.

    GRPOTrainer calls the reward function as:

        reward_fn(
            prompts=[[{"role": "user", "content": "..."}], ...],
            completions=[[{"role": "assistant", "content": "..."}], ...],
            completion_ids=[...],     # token ID lists (unused by us)
            headline=["...", ...],    # from dataset column
            keywords=[[], ...],       # from dataset column
            trainer_state=state,      # TrainerState object (unused by us)
        ) -> list[float]

    Note on data format:
        - prompts[i] is a list of message dicts (conversation history)
        - completions[i] is a list of message dicts (model's response)
        - The actual text content is in the "content" field of the last
          message dict in each list
        - keywords[i] comes directly from the dataset's "keywords" column
        - headline[i] comes directly from the dataset's "headline" column

    Args:
        humor_scorer: Optional humor scoring function for Phase 2.
            If None, humor reward is always 0.0 (Phase 1).

    Returns:
        Callable: A function with the signature expected by GRPOTrainer:
            (prompts, completions, **kwargs) -> list[float]

    Usage:
        # In train_grpo.py
        from rl.rewards import build_reward_fn

        reward_fn = build_reward_fn()  # Phase 1
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            ...
        )
    """
    def reward_fn(
        prompts: list[list[dict]],
        completions: list[list[dict]],
        keywords: list[list[str]] | None = None,
        headline: list[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        """Compute rewards for a batch of (prompt, completion) pairs.

        This inner function is called by GRPOTrainer during training.
        It extracts plain text from the conversational format and delegates
        to compute_reward() for each sample.

        Args:
            prompts: Batch of prompts in conversational format.
                Each element is a list of message dicts, e.g.
                [{"role": "user", "content": "Write a joke about..."}]
            completions: Batch of completions in conversational format.
                Each element is a list of message dicts, e.g.
                [{"role": "assistant", "content": "Why did the..."}]
            keywords: Batch of keyword lists from the dataset column.
                Each element is a list of strings (possibly empty).
                Can be None if the dataset has no "keywords" column.
            headline: Batch of headline strings from the dataset column.
                Can be None if the dataset has no "headline" column.
            **kwargs: Additional keyword args passed by GRPOTrainer
                (e.g., completion_ids, trainer_state).
                These are accepted but not used.

        Returns:
            list[float]: Reward scores, one per (prompt, completion) pair.
                Length equals len(prompts) == len(completions).
        """
        rewards = []

        for i in range(len(completions)):
            # Extract plain text from conversational message format.
            prompt_text = prompts[i][-1]["content"]
            response_text = completions[i][-1]["content"]

            # Get keywords and headline for this sample
            sample_keywords = keywords[i] if keywords is not None else []
            sample_headline = headline[i] if headline is not None else ""

            reward = compute_reward(
                prompt_text=prompt_text,
                response_text=response_text,
                keywords=sample_keywords,
                headline=sample_headline,
                humor_scorer=humor_scorer,
            )
            rewards.append(reward)

        return rewards

    return reward_fn
