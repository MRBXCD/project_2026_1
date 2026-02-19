"""
LLM-as-Judge Humor Scorer
=========================

This module implements a humor scoring system using an external LLM
(Google Gemini) as a judge. It evaluates the humor quality of generated
jokes on a 1-5 scale, which is then mapped to the [-1.0, 1.0] range
expected by the reward function.

Design for GRPO Integration:
    In GRPO training, the humor scorer is called for EVERY completion in
    EVERY training step. With batch_size=8 and num_generations=16, that
    is 128 API calls per step. To make this practical:

    1. Batch scoring: Multiple (prompt, response) pairs are packed into
       a single API call, asking the judge to rate them all at once.
       This reduces 128 individual calls to ~8-16 batch calls per step.

    2. Rate limiting: Built-in delay between API calls to stay within
       Gemini's quota limits and avoid 429 errors.

    3. Graceful degradation: If an API call fails, return neutral score
       (0.0) instead of crashing the training loop.

Scorer Interface:
    The module exposes build_humor_scorer() which returns a callable
    matching the signature expected by build_reward_fn():

        scorer(prompt: str, response: str) -> float

    This callable handles the API client internally (via closure),
    so the caller doesn't need to manage the Gemini client.

    For batch efficiency in GRPO training, a batch-aware alternative
    build_batch_humor_scorer() is also provided, returning:

        scorer(prompts: list[str], responses: list[str]) -> list[float]

Usage:
    # For inference (single scoring)
    from rl.humor_judge import build_humor_scorer
    scorer = build_humor_scorer()
    score = scorer("Write a joke about AI", "Why did the AI cross the road?")

    # For GRPO training (batch scoring integrated into reward)
    from rl.humor_judge import build_humor_scorer
    from rl.rewards import build_reward_fn
    scorer = build_humor_scorer()
    reward_fn = build_reward_fn(humor_scorer=scorer)

Dependencies:
    - google-genai (Google Gemini API)
    - GEMINI_API_KEY environment variable must be set

Environment Variables:
    - GEMINI_API_KEY: Google Gemini API Key (required)
"""

import os
import re
import time
from typing import Callable


# ============================================================
# Constants
# ============================================================

# Gemini model for humor judging.
# Using the flash model for speed and cost efficiency.
JUDGE_MODEL = "gemini-3-flash-preview"

# Number of (prompt, response) pairs to pack into a single API call.
# Larger batches = fewer API calls but longer prompts and higher risk
# of the judge losing track. 8-12 is a good balance.
JUDGE_BATCH_SIZE = 8

# Minimum delay between API calls (in seconds).
# Gemini free tier has strict rate limits (~15 RPM for flash).
# Adjust based on your API tier.
API_CALL_DELAY = 0.1

# Maximum retries per API call (handles transient errors and rate limits).
MAX_RETRIES = 3

# Score mapping: LLM judge returns 1-5, we map to [-1.0, 1.0].
# 1 → -1.0, 2 → -0.5, 3 → 0.0, 4 → 0.5, 5 → 1.0
SCORE_MIN = 1
SCORE_MAX = 5


# ============================================================
# Judge Prompt Template
# ============================================================

JUDGE_PROMPT_SINGLE = """You are an expert judge of humor quality.

Rate the following joke on a scale of 1 to 5.

Context: {prompt}
Joke: {response}

Scoring criteria:
1 = Not funny at all, makes no sense or is incoherent
2 = Slightly amusing but weak, forced, or cliché
3 = Moderately funny, has some wit
4 = Genuinely funny, clever wordplay or unexpected twist
5 = Hilarious, extremely witty and creative

Reply with ONLY a single number (1-5), nothing else."""


JUDGE_PROMPT_BATCH = """You are an expert judge of humor quality.

Rate each of the following jokes on a scale of 1 to 5.

Scoring criteria:
1 = Not funny at all, makes no sense or is incoherent
2 = Slightly amusing but weak, forced, or cliché
3 = Moderately funny, has some wit
4 = Genuinely funny, clever wordplay or unexpected twist
5 = Hilarious, extremely witty and creative

{items}

Reply with ONLY the scores, one per line, in the format:
1: <score>
2: <score>
...

Do not include any other text."""


# ============================================================
# Gemini Client Initialization
# ============================================================

def _init_gemini_client():
    """Initialize Google Gemini API Client.

    Reuses the same pattern as data_preprocessing/synthesize_task_data.py.

    Returns:
        google.genai.Client: Initialized Gemini client.

    Raises:
        ValueError: If GEMINI_API_KEY environment variable is not set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Please run: export GEMINI_API_KEY='your-api-key'"
        )

    from google import genai
    client = genai.Client(api_key=api_key)
    return client


# ============================================================
# API Call with Retry
# ============================================================

def _call_gemini(client, prompt: str, max_retries: int = MAX_RETRIES) -> str | None:
    """Call Gemini API with retry and rate limit handling.

    Sends a prompt to the Gemini model and returns the text response.
    Handles 429 rate limit errors with exponential backoff.

    Args:
        client: Initialized Gemini API client.
        prompt: The judge prompt text.
        max_retries: Maximum retry attempts.

    Returns:
        str: Model response text, or None if all retries failed.
    """
    from google.genai import types

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    temperature=0.1,  # Low temperature for consistent scoring
                    max_output_tokens=64,
                ),
            )
            text = response.text
            if text:
                return text.strip()
            return None

        except Exception as e:
            error_msg = str(e).lower()

            if "429" in error_msg or "resource" in error_msg or "quota" in error_msg:
                wait_time = 2 ** attempt * 5
                print(f"    Judge API rate limit, waiting {wait_time}s ({attempt}/{max_retries})")
                time.sleep(wait_time)
                continue

            print(f"    Judge API error ({attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2)
                continue

    return None


# ============================================================
# Score Parsing
# ============================================================

def _parse_single_score(text: str) -> float | None:
    """Parse a single 1-5 score from LLM judge response text.

    Extracts the first integer in range [1, 5] found in the text.
    Returns None if no valid score is found.

    Args:
        text: Raw response text from the LLM judge.

    Returns:
        float: Score mapped to [-1.0, 1.0], or None if parsing failed.
            Mapping: 1→-1.0, 2→-0.5, 3→0.0, 4→0.5, 5→1.0
    """
    # Extract all integers from the text
    numbers = re.findall(r"\d+", text)
    for num_str in numbers:
        num = int(num_str)
        if SCORE_MIN <= num <= SCORE_MAX:
            # Map [1, 5] → [-1.0, 1.0]
            return (num - SCORE_MIN) / (SCORE_MAX - SCORE_MIN) * 2.0 - 1.0
    return None


def _parse_batch_scores(text: str, expected_count: int) -> list[float | None]:
    """Parse multiple scores from a batch judge response.

    Expects the format:
        1: 3
        2: 4
        3: 2
        ...

    Falls back to extracting all integers in [1-5] if the format
    doesn't match exactly.

    Args:
        text: Raw response text from the batch judge call.
        expected_count: Number of scores expected.

    Returns:
        list[float | None]: Parsed scores mapped to [-1.0, 1.0].
            Length equals expected_count. None for unparseable entries.
    """
    scores: list[float | None] = [None] * expected_count

    # Try structured format first: "1: 3\n2: 4\n..."
    pattern = re.compile(r"(\d+)\s*:\s*(\d+)")
    matches = pattern.findall(text)

    if matches:
        for idx_str, score_str in matches:
            idx = int(idx_str) - 1  # Convert 1-based to 0-based
            score_val = int(score_str)
            if 0 <= idx < expected_count and SCORE_MIN <= score_val <= SCORE_MAX:
                scores[idx] = (score_val - SCORE_MIN) / (SCORE_MAX - SCORE_MIN) * 2.0 - 1.0
        return scores

    # Fallback: extract all integers in [1-5] in order of appearance
    all_numbers = re.findall(r"\d+", text)
    valid_scores = [int(n) for n in all_numbers if SCORE_MIN <= int(n) <= SCORE_MAX]

    for i, score_val in enumerate(valid_scores[:expected_count]):
        scores[i] = (score_val - SCORE_MIN) / (SCORE_MAX - SCORE_MIN) * 2.0 - 1.0

    return scores


# ============================================================
# Single Scorer (for inference / build_reward_fn)
# ============================================================

def build_humor_scorer(
    model: str = JUDGE_MODEL,
    call_delay: float = API_CALL_DELAY,
) -> Callable[[str, str], float]:
    """Build a single-pair humor scorer compatible with build_reward_fn.

    Returns a closure that scores one (prompt, response) pair at a time
    by calling the Gemini API. The Gemini client is initialized once and
    reused across all calls via the closure.

    This scorer matches the interface expected by build_reward_fn():
        scorer(prompt: str, response: str) -> float

    Note on GRPO training performance:
        For GRPO training with 128 completions per step, this single-pair
        scorer would make 128 sequential API calls per step, which is slow.
        Consider using build_batch_humor_scorer() instead for training,
        or accept the slower speed if API quota is not an issue.

    Args:
        model: Gemini model name for judging.
        call_delay: Minimum seconds between API calls.

    Returns:
        Callable: scorer(prompt, response) -> float in [-1.0, 1.0].
            Returns 0.0 (neutral) on any API failure.
    """
    client = _init_gemini_client()

    def scorer(prompt: str, response: str) -> float:
        prompt_text = JUDGE_PROMPT_SINGLE.format(prompt=prompt, response=response)

        time.sleep(call_delay)
        result = _call_gemini(client, prompt_text)

        if result is None:
            return 0.0

        score = _parse_single_score(result)
        return score if score is not None else 0.0

    return scorer


# ============================================================
# Batch Scorer (for efficient GRPO training)
# ============================================================

def build_batch_humor_scorer(
    model: str = JUDGE_MODEL,
    batch_size: int = JUDGE_BATCH_SIZE,
    call_delay: float = API_CALL_DELAY,
) -> Callable[[list[str], list[str]], list[float]]:
    """Build a batch humor scorer for efficient GRPO training.

    Returns a closure that scores multiple (prompt, response) pairs in
    batched API calls. Packs JUDGE_BATCH_SIZE pairs into each API call
    to minimize the total number of calls per training step.

    For 128 completions per step with batch_size=8:
        128 / 8 = 16 API calls per step (instead of 128).

    The returned callable has the signature:
        scorer(prompts: list[str], responses: list[str]) -> list[float]

    To integrate with build_reward_fn, the reward function's inner loop
    needs to be adapted to call this batch scorer once per step rather
    than calling the single scorer 128 times. See rl/rewards.py for
    the batch_humor_scorer integration pattern.

    Args:
        model: Gemini model name for judging.
        batch_size: Number of pairs per API call.
        call_delay: Minimum seconds between API calls.

    Returns:
        Callable: scorer(prompts, responses) -> list[float].
            Each score is in [-1.0, 1.0]. Returns 0.0 for failed items.
    """
    client = _init_gemini_client()

    def scorer(prompts: list[str], responses: list[str]) -> list[float]:
        all_scores: list[float] = []

        # Process in batches of JUDGE_BATCH_SIZE
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            batch_responses = responses[start:end]
            chunk_size = end - start

            # Build the batch items text
            items_lines = []
            for i, (p, r) in enumerate(zip(batch_prompts, batch_responses), 1):
                items_lines.append(f"Joke {i}:")
                items_lines.append(f"  Context: {p}")
                items_lines.append(f"  Joke: {r}")
                items_lines.append("")
            items_text = "\n".join(items_lines)

            judge_prompt = JUDGE_PROMPT_BATCH.format(items=items_text)

            # for debug
            # print("judge_prompt:", judge_prompt)
            # print("-"*40)
            
            time.sleep(call_delay)
            result = _call_gemini(client, judge_prompt)

            if result is None:
                all_scores.extend([0.0] * chunk_size)
                continue

            parsed = _parse_batch_scores(result, chunk_size)
            # Replace None with 0.0 (neutral on failure)
            all_scores.extend(s if s is not None else 0.0 for s in parsed)

        return all_scores

    return scorer


if __name__ == "__main__":
    # Quick test: score a single joke
    print("Testing humor scorer...")
    scorer = build_batch_humor_scorer()
    score = scorer(
        "Write a joke about AI safety regulations",
        "The new AI safety law requires all robots to wear seatbelts. "
        "Ironically, the only crash they're worried about is a system crash.",
    )
    print(f"Score: {score}")
