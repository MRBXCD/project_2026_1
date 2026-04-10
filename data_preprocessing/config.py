"""
Unified Configuration for Data Preprocessing Pipeline
======================================================

All tunable parameters for the data preprocessing pipeline are centralized here.
Each section corresponds to a pipeline stage. Code files import from this module
instead of defining their own constants.

What is NOT here (and why):
    - Path constants: structural, tied to project layout, stay in each file
    - Prompt templates: content (prompt_templates.py)
    - CFun instruction strings & regex patterns: cleaning logic (parsers.py)
    - Boring text generation prompts & API schema: prompt content (synthesize_reward_data.py)
"""


# ============================================================
# Section 1: Parsing
# ============================================================

# Cap value for rJokes score normalization.
# The actual maximum raw score in the dataset is 11 (Reddit upvotes).
# Using 11 maps the full data range onto [0, 1].
RJOKES_SCORE_CAP = 11

# Max score for Chinese Humor and HAHA (original is 1-5)
HUMOR_SCORE_MAX = 5.0

# CFun extraction length filters
CFUN_MIN_LEN = 10
CFUN_MAX_LEN = 500


# ============================================================
# Section 2: SFT Formatting
# ============================================================

# Fixed Type A source selection policy.
# Each entry: source_name -> {"strategy": "top_by_score"|"random", "count": int}
# Sources not listed here are excluded from SFT Type A.
SFT_TYPE_A_SOURCE_SELECTION = {
    "rjokes": {"strategy": "top_by_score", "count": 1000},
    "haha": {"strategy": "top_by_score", "count": 1000},
    "cfun": {"strategy": "random", "count": 1000},
}

# Target ratio of Type A data in the final SFT dataset (0.0 ~ 1.0).
# Default 0.7 means 70% Type A, 30% Type B.
SFT_TYPE_A_RATIO = 0.7

# Validation set ratio for SFT train/val split.
SFT_VAL_RATIO = 0.1


# ============================================================
# Section 3: Reward Pair Formatting
# ============================================================

# Per-language allocation for reward model preference pairs.
#   score_based: max number of score-based (joke-vs-joke) pairs to keep.
#                None means use all available.
#   synthesized: max number of synthesized hard-negative pairs to keep.
#                None means use all available.
REWARD_PAIR_ALLOCATION = {
    "en": {"score_based": 7_000, "synthesized": 7_000},
    "es": {"score_based": 7_000, "synthesized": 7_000},
    "zh": {"score_based": 7_000, "synthesized": 7_000},
}

# Fine-grained score-bucket configuration for reward pair construction
REWARD_SCORE_BUCKETS = 4
REWARD_PAIR_TEMPLATES = [(3, 2), (2, 1), (1, 0), (3, 1), (2, 0), (3, 0)]
REWARD_TEMPLATE_RATIOS = {
    (3, 2): 0.20,
    (2, 1): 0.18,
    (1, 0): 0.12,
    (3, 1): 0.16,
    (2, 0): 0.14,
    (3, 0): 0.20,
}

# Deprecated in current reward sampling policy.
# Synthesized data is now controlled only by explicit per-language
# allocation caps in REWARD_PAIR_ALLOCATION[*]["synthesized"].
MAX_SYNTH_RATIO_BY_LANG: dict[str, float] = {}

# Per-language bidirectional reuse limits for reward pair construction
MAX_REUSE_PER_CHOSEN_BY_LANG = {"en": 4, "es": 5, "zh": 6}
MAX_REUSE_PER_REJECTED_BY_LANG = {"en": 4, "es": 5, "zh": 6}

# Maximum rounds of pair reuse attempts
MAX_PAIR_REUSE_ROUNDS = 3

# Reuse monitoring
REUSE_MONITOR_ENABLED = True
REUSE_WARN_RATIO = 0.8
REUSE_STATS_EXPORT_PATH = None


# ============================================================
# Section 4: Synthesis - Task Data
# ============================================================

# Realtime multi-call group size for Gemini API.
# Keep this relatively small to avoid malformed/truncated JSON responses.
REALTIME_MULTI_GROUP_SIZE = 100

# Keyword pool defaults
DEFAULT_KEYWORD_POOL_SIZE = 2000
DEFAULT_KEYWORD_CANDIDATE_LIMIT = 20000

# spaCy model names per language (for keyword pool POS filtering)
SPACY_MODEL_NAMES = {
    "en": "en_core_web_sm",
    "zh": "zh_core_web_sm",
    "es": "es_core_news_sm",
}

# Allowed POS tags for keyword pool filtering
KEYWORD_ALLOWED_POS = {
    "en": {"NOUN", "PROPN"},
    "zh": {"NOUN", "PROPN"},
    "es": {"NOUN", "PROPN"},
}


# ============================================================
# Section 5: Synthesis - Reward Data
# ============================================================

# Number of boring statements to generate per Gemini API call
BATCH_SIZE = 100

# Per-language score thresholds for selecting high-quality chosen jokes.
# Only en and es need thresholds; zh uses CFun with no score filter.
_SCORE_THRESHOLDS = {
    # rJokes raw scores are Reddit upvotes, normalized as min(raw, 11) / 11.
    # raw >= 5 → normalized >= 5/11 ≈ 0.4545.  Selects the top ~8% of
    # jokes and yields a large enough pool for synthesis (~34K candidates).
    "en": round(5 / 11, 4),
    # HAHA funniness_average is normalized as avg / 5.0.
    # Using 0.3 (raw avg >= 1.5/5) keeps most genuinely humorous tweets
    # while excluding the very lowest quality, giving a large enough pool.
    "es": 0.3,
}


# ============================================================
# Section 6: Pipeline CLI Defaults
# ============================================================

DEFAULT_N_HEADLINE = 200
DEFAULT_N_KEYWORD = 100
DEFAULT_EVAL_RATIO = 0.2
DEFAULT_SEED = 42
