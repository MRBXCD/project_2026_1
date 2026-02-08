"""
Prompt Template Module
======================

This module defines three types of prompts, used by formatters.py when constructing SFT / GRPO data:

1. **Type A Prompt Pool** — General humor request prompts
   - Usage: User-side prompts for SFT Type A data and preference pair data
   - 15 diverse prompts defined for each language
   - A prompt is randomly selected from the corresponding language pool for each training sample
   - Purpose: Teach the model to respond to various expressions requesting jokes, not just "Tell me a joke"

2. **Type B Task Templates** — Task formatting prompts
   - Usage: SFT Type B data synthesis, GRPO prompt construction
   - 6 templates in total: Subtask type (headline / keyword) × Language (en / zh / es)
   - Contains placeholders {headline} or {word1}/{word2}, filled by the caller

3. **GRPO Templates** — Reusing Type B templates
   - GRPO stage uses exactly the same templates as Type B, just different data sources
   - Therefore not defined separately, directly call Type B construction functions

Exposed Interfaces:
    - get_random_type_a_prompt(lang, rng) → str
    - build_headline_prompt(headline, lang) → str
    - build_keyword_prompt(word1, word2, lang) → str

Dependencies:
    - random (Standard library)
"""

import random


# ============================================================
# Type A: General Humor Prompt Pool
# ============================================================
# 15 prompts per language, covering different styles (direct request, polite request, imperative, etc.)
# To ensure diversity of user inputs in SFT data.

TYPE_A_PROMPTS_EN = [
    "Tell me a joke.",
    "Tell me a short joke.",
    "Make me laugh with a quick joke.",
    "Can you share something funny?",
    "I need a good laugh. Hit me with a joke.",
    "Share a humorous one-liner.",
    "Give me your best short joke.",
    "I could use some humor right now. Got a joke?",
    "Tell me something that will make me smile.",
    "What's a good joke you know?",
    "Surprise me with a funny one-liner.",
    "Got any good jokes?",
    "Make me laugh.",
    "Hit me with your funniest joke.",
    "Tell me a witty joke.",
]

TYPE_A_PROMPTS_ZH = [
    "给我讲个笑话吧。",
    "说个段子听听。",
    "来点幽默的。",
    "讲个好笑的故事。",
    "我想听个笑话。",
    "能给我讲个段子吗？",
    "来个短笑话。",
    "说点有趣的东西。",
    "给我说个让人发笑的笑话。",
    "讲个让我开心一下的笑话。",
    "你知道什么好笑的笑话吗？",
    "逗我笑一个。",
    "有没有什么有趣的段子？",
    "给我讲个冷笑话。",
    "来个幽默的短段子吧。",
]

TYPE_A_PROMPTS_ES = [
    "Cuéntame un chiste.",
    "Hazme reír con un chiste corto.",
    "¿Puedes compartir algo gracioso?",
    "Necesito reírme. Dime un chiste.",
    "Comparte un chiste ingenioso.",
    "Dame tu mejor chiste corto.",
    "Me vendría bien algo de humor. ¿Tienes un chiste?",
    "Dime algo que me haga sonreír.",
    "¿Cuál es un buen chiste que conozcas?",
    "Sorpréndeme con un chiste divertido.",
    "¿Tienes algún chiste bueno?",
    "Hazme reír.",
    "Cuéntame tu chiste más gracioso.",
    "Dime un chiste ingenioso.",
    "¿Sabes algún chiste corto?",
]

# Prompt pool dictionary indexed by language code, for easy lookup in get_random_type_a_prompt
_TYPE_A_POOLS = {
    "en": TYPE_A_PROMPTS_EN,
    "zh": TYPE_A_PROMPTS_ZH,
    "es": TYPE_A_PROMPTS_ES,
}

EXTRA_PROMPT = {
    "en": "Please give me the joke only, no other text.",
    "zh": "请只给出笑话，不要给出任何其他文本。",
    "es": "No dar ninguna otra información.",
}


# ============================================================
# Type B: Task Formatting Prompt Templates
# ============================================================
# Placeholders in templates:
#   - {headline}  — News headline (headline subtask)
#   - {word1}, {word2} — Keywords (keyword subtask)
#
# Note: Templates use Python's str.format() syntax for filling,
#       so braces {headline} etc. in templates will be replaced by actual values.

# --- Headline Subtask Templates ---

_HEADLINE_TEMPLATE_EN = (
    "You are a witty comedian. Given the following news headline, "
    "write a short, funny one-liner joke inspired by it.\n\n"
    "Headline: \"{headline}\"\n\n"
    "Write a humorous one-liner inspired by the headline. Please give me the joke only, no other text."
)

_HEADLINE_TEMPLATE_ZH = (
    "你是一位机智的喜剧演员。根据以下新闻标题，写一个简短有趣的笑话。\n\n"
    "新闻标题：「{headline}」\n\n"
    "写一句幽默的段子。请只给出笑话，不要给出任何其他文本。"
)

_HEADLINE_TEMPLATE_ES = (
    "Eres un comediante ingenioso. Dado el siguiente titular de noticias, "
    "escribe un chiste corto y divertido inspirado en él.\n\n"
    "Titular: \"{headline}\"\n\n"
    "Escribe un chiste divertido de una línea inspirado en el titular. No dar ninguna otra información."
)

_HEADLINE_TEMPLATES = {
    "en": _HEADLINE_TEMPLATE_EN,
    "zh": _HEADLINE_TEMPLATE_ZH,
    "es": _HEADLINE_TEMPLATE_ES,
}

# --- Keyword Subtask Templates ---

_KEYWORD_TEMPLATE_EN = (
    "You are a witty comedian. Write a short, funny one-liner joke "
    "that naturally includes both of the following words: "
    "'{word1}' and '{word2}'.\n\n"
    "Write a humorous one-liner that contains both required words. Please give me the joke only, no other text."
)

_KEYWORD_TEMPLATE_ZH = (
    "你是一位机智的喜剧演员。写一个简短有趣的笑话，"
    "其中必须自然地包含以下两个词：「{word1}」和「{word2}」。\n\n"
    "写一句包含以上两个词语的幽默段子。请只给出笑话，不要给出任何其他文本。"
)

_KEYWORD_TEMPLATE_ES = (
    "Eres un comediante ingenioso. Escribe un chiste corto y divertido "
    "que incluya naturalmente las siguientes dos palabras: "
    "'{word1}' y '{word2}'.\n\n"
    "Escribe un chiste divertido de una línea que contenga ambas palabras. No dar ninguna otra información."
)

_KEYWORD_TEMPLATES = {
    "en": _KEYWORD_TEMPLATE_EN,
    "zh": _KEYWORD_TEMPLATE_ZH,
    "es": _KEYWORD_TEMPLATE_ES,
}


# ============================================================
# Exposed Interfaces
# ============================================================

def get_random_type_a_prompt(lang: str, rng: random.Random | None = None) -> str:
    """Select a random prompt from the Type A prompt pool for the corresponding language.

    Used for SFT Type A data construction and preference pair construction.

    Args:
        lang: Language code, "en" / "zh" / "es"
        rng: Optional random.Random instance, used to control random seed for reproducibility.
             If not provided, uses the global random module (non-reproducible).

    Returns:
        str: A randomly selected prompt text

    Raises:
        ValueError: Unsupported language code

    Example:
        >>> rng = random.Random(42)
        >>> get_random_type_a_prompt("en", rng)
        'Hit me with your funniest joke.'
    """
    if lang not in _TYPE_A_POOLS:
        raise ValueError(f"Unsupported language code: '{lang}', options: {list(_TYPE_A_POOLS.keys())}")

    pool = _TYPE_A_POOLS[lang]



    if rng is not None:
        return rng.choice(pool) + " " + EXTRA_PROMPT[lang]
    return random.choice(pool) + " " + EXTRA_PROMPT[lang]


def build_headline_prompt(headline: str, lang: str) -> str:
    """Construct complete prompt for headline subtask based on news headline and language.

    Used for SFT Type B data, GRPO prompt construction, and Type B data synthesis.

    Args:
        headline: Original news headline
        lang: Language code, "en" / "zh" / "es"

    Returns:
        str: Complete prompt filled with headline

    Raises:
        ValueError: Unsupported language code

    Example:
        >>> build_headline_prompt("NASA finds water on Mars", "en")
        'You are a witty comedian. Given the following news headline, ...'
    """
    if lang not in _HEADLINE_TEMPLATES:
        raise ValueError(f"Unsupported language code: '{lang}', options: {list(_HEADLINE_TEMPLATES.keys())}")

    return _HEADLINE_TEMPLATES[lang].format(headline=headline)


def build_keyword_prompt(word1: str, word2: str, lang: str) -> str:
    """Construct complete prompt for keyword subtask based on two keywords and language.

    Used for SFT Type B data, GRPO prompt construction, and Type B data synthesis.

    Args:
        word1: First required keyword
        word2: Second required keyword
        lang: Language code, "en" / "zh" / "es"

    Returns:
        str: Complete prompt filled with keywords

    Raises:
        ValueError: Unsupported language code

    Example:
        >>> build_keyword_prompt("hammer", "flower", "en")
        "You are a witty comedian. Write a short, funny one-liner joke ..."
    """
    if lang not in _KEYWORD_TEMPLATES:
        raise ValueError(f"Unsupported language code: '{lang}', options: {list(_KEYWORD_TEMPLATES.keys())}")

    return _KEYWORD_TEMPLATES[lang].format(word1=word1, word2=word2)
