"""
Type B Data Synthesis Script
============================

This script is independent of pipeline.py and is used to synthesize
SFT Type B data (task-formatted training samples).

Supported workflow:
    run:
      Build prompts -> call realtime Gemini (multi-task JSON) -> quality filter
      -> save type_b_{lang}.jsonl

Important Design Principles:
    - Do not use SemEval-provided headline/keyword pairs (prevent leakage)
    - Headlines come from external Babel Briefings dataset
    - Keyword pairs are generated from local language-specific keyword pools
    - Final SFT Type B data is written to data/synthesized/type_b_{lang}.jsonl
    - pipeline.py format_sft stage automatically loads type_b_*.jsonl

Usage:
    # Synthesize English data
    python -m data_preprocessing.synthesize_task_data --lang en

    # Synthesize Chinese data, specify quantity
    python -m data_preprocessing.synthesize_task_data --lang zh --n_headline 300 --n_keyword 150

    # Synthesize all three languages
    python -m data_preprocessing.synthesize_task_data --lang all

Dependencies:
    - google-genai (Google Gemini API)
    - prompt_templates (Local)
    - datasets (HuggingFace, for loading Babel Briefings)

Environment Variables:
    - GEMINI_API_KEY: Google Gemini API Key (Must be set)
"""

import argparse
import itertools
import json
import os
import random
import time
from pathlib import Path

from data_preprocessing.prompt_templates import (
    build_headline_prompt,
    build_keyword_prompt,
)

# ============================================================
# Path Constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTHESIZED_DIR = PROJECT_ROOT / "data" / "synthesized"


# ============================================================
# Keyword Vocabulary (For keyword subtask)
# ============================================================
# Principles for selecting these word pairs:
#   - Do not overlap with keywords provided by SemEval
#   - Cover various POS and semantic categories (animals, objects, actions, food, etc.)
#   - Create "incongruity" when paired, helping humor effect
#
# Maintain an independent vocabulary for each language.
# The synthesis flow for keyword subtask will randomly pick two words to pair from here.

KEYWORD_POOL_EN = [
    "astronaut", "cactus", "piano", "tornado", "kangaroo",
    "umbrella", "volcano", "noodle", "submarine", "giraffe",
    "avalanche", "toaster", "jellyfish", "telescope", "pretzel",
    "dinosaur", "trampoline", "mushroom", "lighthouse", "accordion",
    "parrot", "tuxedo", "glacier", "chopstick", "saxophone",
    "elevator", "flamingo", "suitcase", "comet", "waffle",
    "lantern", "backpack", "igloo", "fountain", "zeppelin",
    "crystal", "marshmallow", "octopus", "microscope", "violin",
    "pyramid", "carousel", "otter", "galaxy", "windmill",
    "cinnamon", "meadow", "satellite", "pancake", "tunnel",
    "pebble", "seahorse", "origami", "skateboard", "nectar",
    "volleyball", "harpoon", "beehive", "moonlight", "campfire",
    "hourglass", "bamboo", "cupcake", "canyon", "firework",
    "whistle", "compass", "raincoat", "snowflake", "reef",
    "coconut", "meerkat", "drumstick", "stardust", "museum",
    "kayak", "robot", "fossil", "carnation", "pinwheel",
    "donut", "meteor", "chessboard", "waistcoat", "ginger",
    "castle", "seashell", "pillow", "chimney", "trombone",
    "hedgehog", "pagoda", "puddle", "firefly", "bathtub",
    "thunder", "almond", "goblet", "orchid", "waterfall",
    "harbor", "sunbeam", "clocktower", "strawberry", "walrus",
    "teapot", "scooter", "anvil", "plankton", "hammock",
    "popsicle", "bison", "sandcastle", "dice", "quartz",
    "lizard", "rainbow", "trumpet", "cherry", "scooterbag",
    "snowman", "raccoon", "blueberry", "gingerbread", "hamster",
    "tortoise", "paperclip", "megaphone", "lilypad", "cliff",
    "sunflower", "mapcase", "starfish", "ukulele", "parachute",
    "millstone", "jigsaw", "helmet", "sapphire", "cabin",
    "peppermint", "locomotive", "tangerine", "monsoon", "riverbank",
    "binoculars", "moth", "catapult", "breadstick", "planetarium",
    "cobblestone", "windchime", "mooncake", "slingshot", "strawhat",
]

KEYWORD_POOL_ZH = [
    "宇航员", "仙人掌", "钢琴", "龙卷风", "袋鼠",
    "雨伞", "火山", "面条", "潜水艇", "长颈鹿",
    "雪崩", "烤面包机", "水母", "望远镜", "恐龙",
    "蹦床", "蘑菇", "灯塔", "手风琴", "鹦鹉",
    "冰川", "筷子", "萨克斯", "电梯", "火烈鸟",
    "行李箱", "彗星", "华夫饼", "拖拉机", "企鹅",
    "灯笼", "背包", "喷泉", "飞船", "水晶",
    "软糖", "章鱼", "显微镜", "小提琴", "金字塔",
    "旋转木马", "水獭", "银河", "风车", "肉桂",
    "草地", "卫星", "煎饼", "隧道", "鹅卵石",
    "海马", "折纸", "滑板", "树汁", "排球",
    "鱼叉", "蜂巢", "月光", "篝火", "沙漏",
    "竹子", "纸杯蛋糕", "峡谷", "焰火", "口哨",
    "指南针", "雨衣", "雪晶", "珊瑚礁", "椰子",
    "狐獴", "鼓槌", "星尘", "博物馆", "皮划艇",
    "机器人", "化石", "康乃馨", "风车玩具", "甜甜圈",
    "流星", "棋盘", "马甲", "姜饼", "城堡",
    "贝壳", "枕头", "烟囱", "长号", "刺猬",
    "宝塔", "水洼", "萤火虫", "浴缸", "雷声",
    "杏仁", "高脚杯", "铃兰", "瀑布", "港口",
    "钟楼", "草莓", "海象", "茶壶", "滑板车",
    "铁砧", "浮游生物", "吊床", "冰棍", "野牛",
    "沙堡", "骰子", "石英", "蜥蜴", "彩虹",
    "小号", "樱桃", "雪人", "浣熊", "蓝莓",
    "仓鼠", "乌龟", "回形针", "扩音器", "睡莲",
    "悬崖", "向日葵", "海星", "尤克里里", "降落伞",
    "磨盘", "拼图", "头盔", "蓝宝石", "木屋",
    "薄荷糖", "火车头", "橘子", "季风", "河岸",
    "双筒望远镜", "飞蛾", "投石机", "面包棒", "天文馆",
    "鹅卵路", "风铃", "月饼", "弹弓", "草帽",
]

KEYWORD_POOL_ES = [
    "astronauta", "cactus", "piano", "tornado", "canguro",
    "paraguas", "volcán", "fideos", "submarino", "jirafa",
    "avalancha", "tostadora", "medusa", "telescopio", "pretzel",
    "dinosaurio", "trampolín", "hongo", "faro", "acordeón",
    "loro", "esmoquin", "glaciar", "palillos", "saxofón",
    "ascensor", "flamenco", "maleta", "cometa", "gofre",
    "linterna", "mochila", "iglú", "fuente", "zepelín",
    "cristal", "caramelo", "pulpo", "microscopio", "violín",
    "pirámide", "carrusel", "nutria", "galaxia", "molino",
    "canela", "pradera", "satélite", "panqueque", "túnel",
    "guijarro", "caballito", "origami", "patineta", "néctar",
    "voleibol", "arpón", "colmena", "luzlunar", "fogata",
    "relojarena", "bambú", "magdalena", "cañón", "fuegos",
    "silbato", "brújula", "impermeable", "copo", "arrecife",
    "coco", "suricata", "baqueta", "polvoestelar", "museo",
    "kayak", "robot", "fósil", "clavel", "molinillo",
    "dónut", "meteoro", "tablero", "chaleco", "jengibre",
    "castillo", "concha", "almohada", "chimenea", "trombón",
    "erizo", "pagoda", "charco", "luciérnaga", "bañera",
    "trueno", "almendra", "cáliz", "orquídea", "cascada",
    "puerto", "torreloj", "fresa", "morsa", "tetera",
    "patinete", "yunque", "plancton", "hamaca", "paleta",
    "bisonte", "castilloarena", "dado", "cuarzo", "lagarto",
    "arcoíris", "trompeta", "cereza", "muñeco", "mapache",
    "arándano", "hámster", "tortuga", "clip", "megáfono",
    "nenúfar", "acantilado", "girasol", "estrellamar", "ukelele",
    "paracaídas", "piedramolino", "rompecabezas", "casco", "zafiro",
    "cabaña", "menta", "locomotora", "mandarina", "monzón",
    "ribera", "binoculares", "polilla", "catapulta", "grisín",
    "planetario", "adoquín", "campanaviento", "pastelluna", "tirachinas",
]

_KEYWORD_POOLS = {
    "en": KEYWORD_POOL_EN,
    "zh": KEYWORD_POOL_ZH,
    "es": KEYWORD_POOL_ES,
}

# Realtime multi-call group size.
# Keep this relatively small to avoid malformed/truncated JSON responses.
REALTIME_MULTI_GROUP_SIZE = 100


# ============================================================
# Gemini API Call
# ============================================================

def _init_gemini_client():
    """Initialize Google Gemini API Client.

    Requires environment variable GEMINI_API_KEY.

    Returns:
        google.genai.Client instance

    Raises:
        ValueError: GEMINI_API_KEY environment variable not set
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Please run: export GEMINI_API_KEY='your-api-key'"
        )

    from google import genai
    client = genai.Client(api_key=api_key)
    print("  Gemini API client initialized successfully")
    return client


def _call_gemini(client, prompt: str, max_retries: int = 3) -> str | None:
    """Call Gemini API to generate response, with retry and rate limit handling.

    Args:
        client: Gemini API client
        prompt: Prompt text sent to the model
        max_retries: Maximum number of retries (handling API rate limits or temporary errors)

    Returns:
        str: Text generated by the model, None if failed
    """
    from google.genai import types

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    # Disable thinking mode, just need direct response
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    # Higher temperature encourages creativity
                    temperature=0.9,
                    max_output_tokens=256,
                ),
            )
            text = response.text
            if text:
                return text.strip()
            return None

        except Exception as e:
            error_msg = str(e).lower()

            # API Rate Limit (429 Too Many Requests) or Service Temporarily Unavailable (503)
            if "429" in error_msg or "resource" in error_msg or "quota" in error_msg:
                wait_time = 2 ** attempt * 5  # Exponential backoff: 10s, 20s, 40s
                print(f"    API Rate Limit, waiting {wait_time}s before retry ({attempt}/{max_retries})")
                time.sleep(wait_time)
                continue

            # Other errors
            print(f"    Gemini call failed ({attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2)
                continue

    return None


def _build_multi_generation_prompt(task_records: list[dict], lang: str) -> str:
    """Build a single prompt asking model to solve multiple tasks with JSON output."""
    header = (
        f"You are generating humorous responses in language={lang}.\n"
        "Solve each task independently.\n"
        "Return ONLY a valid JSON array. No markdown, no extra text.\n"
        "Each item must be an object with keys: index, output.\n"
        "index: integer task index.\n"
        "output: one joke response for that task.\n"
    )
    lines = [header, "Tasks:"]
    for idx, item in enumerate(task_records):
        lines.append(f"{idx}. input={item.get('prompt','')}")
    return "\n".join(lines)


def _call_gemini_multi(
    client,
    task_records: list[dict],
    lang: str,
    max_retries: int = 3,
) -> list[dict]:
    """Call realtime Gemini once and get multiple structured outputs."""
    if not task_records:
        return []
    from google.genai import types

    response_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "index": {"type": "integer"},
                "output": {"type": "string"},
            },
            "required": ["index", "output"],
        },
    }
    prompt = _build_multi_generation_prompt(task_records, lang)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json",
                    response_json_schema=response_schema,
                    temperature=0.9,
                    max_output_tokens=min(8192, max(1024, len(task_records) * 220)),
                ),
            )
            if not response.text:
                return []
            parsed = json.loads(response.text)
            if not isinstance(parsed, list):
                return []
            results: list[dict] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                index = item.get("index")
                output = item.get("output")
                if isinstance(index, int) and isinstance(output, str) and output.strip():
                    results.append(
                        {
                            "index": index,
                            "output": output.strip(),
                        }
                    )
            return results
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "resource" in error_msg or "quota" in error_msg:
                wait_time = 2 ** attempt * 5
                print(f"    API Rate Limit, waiting {wait_time}s before retry ({attempt}/{max_retries})")
                time.sleep(wait_time)
                continue
            print(f"    Gemini multi call failed ({attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2)
                continue
    return []


def _generate_multi_responses(
    client,
    prompt_records: list[dict],
    lang: str,
    group_size: int = REALTIME_MULTI_GROUP_SIZE,
) -> list[str | None]:
    """Generate responses for many prompts via grouped realtime multi-calls."""
    outputs: list[str | None] = [None] * len(prompt_records)
    if not prompt_records:
        return outputs

    for start, chunk_outputs, _, _ in _iter_multi_response_chunks(
        client=client,
        prompt_records=prompt_records,
        lang=lang,
        group_size=group_size,
    ):
        for i, value in enumerate(chunk_outputs):
            outputs[start + i] = value
    return outputs


def _format_eta_by_passed(passed: int, target: int, elapsed_s: float) -> str:
    """Estimate ETA by current passed rate."""
    if passed <= 0 or elapsed_s <= 0:
        return "N/A"
    remaining = max(target - passed, 0)
    if remaining == 0:
        return "0s"
    rate = passed / elapsed_s
    if rate <= 0:
        return "N/A"
    eta_s = int(remaining / rate)
    minutes, seconds = divmod(eta_s, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes}m{seconds}s"
    if minutes > 0:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"


def _iter_multi_response_chunks(
    client,
    prompt_records: list[dict],
    lang: str,
    group_size: int = REALTIME_MULTI_GROUP_SIZE,
):
    """Yield grouped outputs and fallback stats for realtime_multi path."""
    total = len(prompt_records)
    for start in range(0, total, group_size):
        chunk = prompt_records[start:start + group_size]
        chunk_outputs: list[str | None] = [None] * len(chunk)
        fallback_count = 0
        multi_success_count = 0

        chunk_results = _call_gemini_multi(client, chunk, lang=lang)
        if not chunk_results:
            # fallback to single-call realtime for this chunk
            for i, rec in enumerate(chunk):
                single = _call_gemini(client, rec.get("prompt", ""))
                chunk_outputs[i] = single
                fallback_count += 1
            yield start, chunk_outputs, fallback_count, multi_success_count
            continue

        seen = set()
        for item in chunk_results:
            idx = item["index"]
            if 0 <= idx < len(chunk):
                chunk_outputs[idx] = item["output"]
                seen.add(idx)
        multi_success_count = len(seen)

        # fill missing indexes with single-call realtime
        for i, rec in enumerate(chunk):
            if i not in seen:
                single = _call_gemini(client, rec.get("prompt", ""))
                chunk_outputs[i] = single
                fallback_count += 1

        yield start, chunk_outputs, fallback_count, multi_success_count


# ============================================================
# News Headline Acquisition (Babel Briefings)
# ============================================================

def _load_headlines(lang: str, n_samples: int, seed: int = 42) -> list[str]:
    """Randomly sample specified number of news headlines from Babel Briefings dataset.

    Babel Briefings is a multilingual news headline dataset (HuggingFace: felixludos/babel-briefings),
    containing 4.7 million news headlines in 30 languages including English, Chinese, Spanish, etc.

    Args:
        lang: Language code "en" / "zh" / "es"
        n_samples: Number of headlines required
        seed: Random seed

    Returns:
        list[str]: List of news headlines
    """
    import datasets as ds_lib

    print(f"  Loading {lang} headlines from Babel Briefings (streaming mode)...")

    # Use streaming=True to avoid downloading full ~5GB dataset
    dataset = ds_lib.load_dataset(
        "felixludos/babel-briefings",
        split="train",
        streaming=True,
    )

    # Filter by language, collect enough candidate headlines
    # Collect more (3x) to have sufficient pool for random sampling later
    collect_target = n_samples * 3
    candidates = []

    for example in dataset:
        if example.get("language") == lang:
            title = example.get("title", "")
            # Basic filter: non-empty, reasonable length (not too short or too long)
            if title and 10 <= len(title) <= 300:
                candidates.append(title)
            if len(candidates) >= collect_target:
                break

    print(f"  Collected {len(candidates)} candidate headlines")

    if len(candidates) == 0:
        raise ValueError(f"Failed to fetch {lang} headlines from Babel Briefings")

    # Random sampling
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n_samples]

    print(f"  Selected {len(selected)} headlines")
    return selected


# ============================================================
# Keyword Pair Generation
# ============================================================

def _generate_keyword_pairs(lang: str, n_pairs: int, seed: int = 42) -> list[tuple[str, str]]:
    """Randomly pair keywords from pool to generate unique keyword pairs.

    Pairing rules:
    - Two words are not identical
    - Same combination not repeated (Order irrelevant: (a,b) and (b,a) are same)

    Args:
        lang: Language code
        n_pairs: Number of pairs to generate
        seed: Random seed

    Returns:
        list[tuple[str, str]]: List of keyword pairs
    """
    if lang not in _KEYWORD_POOLS:
        raise ValueError(f"Unsupported language code: '{lang}'")

    pool = _KEYWORD_POOLS[lang]

    # Generate all unique combinations of two
    all_combos = list(itertools.combinations(pool, 2))

    if n_pairs > len(all_combos):
        print(
            f"  Warning: Requested {n_pairs} keyword pairs, but vocabulary can only generate {len(all_combos)} combinations, "
            f"using all combinations"
        )
        n_pairs = len(all_combos)

    rng = random.Random(seed)
    selected = rng.sample(all_combos, n_pairs)

    print(f"  Generated {len(selected)} keyword pairs")
    return selected


# ============================================================
# Quality Filtering
# ============================================================

def _filter_headline_response(response: str) -> bool:
    """Check if the response generated for headline subtask is qualified.

    Filter conditions:
    - Non-empty
    - Reasonable length (10 ~ 500 characters)
    - Does not contain common refusal patterns ("I cannot", "I'm sorry", etc.)

    Args:
        response: Response text generated by model

    Returns:
        bool: True if qualified
    """
    if not response or not response.strip():
        return False

    text = response.strip()

    # Length check
    if len(text) < 10 or len(text) > 500:
        return False

    # Refusal pattern check (Model sometimes refuses to generate humor)
    refusal_patterns = [
        "i cannot", "i can't", "i'm sorry", "i apologize",
        "as an ai", "as a language model",
        "不能", "抱歉", "对不起", "作为一个ai", "作为语言模型",
        "no puedo", "lo siento", "disculpa",
    ]
    text_lower = text.lower()
    for pattern in refusal_patterns:
        if pattern in text_lower:
            return False

    return True


def _filter_keyword_response(response: str, word1: str, word2: str) -> bool:
    """Check if the response generated for keyword subtask is qualified.

    Filter conditions:
    - Meets all conditions of _filter_headline_response
    - Extra: Response must contain both word1 and word2 (case-insensitive)

    Args:
        response: Response text generated by model
        word1: First required keyword
        word2: Second required keyword

    Returns:
        bool: True if qualified
    """
    # Check general conditions first
    if not _filter_headline_response(response):
        return False

    # Extra check: both keywords must appear in response (case-insensitive)
    text_lower = response.lower()
    if word1.lower() not in text_lower:
        return False
    if word2.lower() not in text_lower:
        return False

    return True


# ============================================================
# Synthesis Main Logic
# ============================================================


def synthesize_for_language(
    lang: str,
    n_headline: int = 200,
    n_keyword: int = 100,
    seed: int = 42,
) -> list[dict]:
    """Synthesize Type B data for specified language using realtime_multi."""
    oversample_ratio = 1.5
    client = _init_gemini_client()
    all_samples = []

    # ---- Headline Part ----
    if n_headline > 0:
        print(f"\n  --- Headline Subtask (Target: {n_headline}) ---")
        stage_start = time.time()
        n_headline_fetch = int(n_headline * oversample_ratio)
        headlines = _load_headlines(lang, n_headline_fetch, seed=seed)

        passed = 0
        failed = 0
        fallback_total = 0
        multi_success_total = 0
        headline_records = [{"prompt": build_headline_prompt(headline, lang)} for headline in headlines]
        for start, chunk_outputs, fallback_count, multi_success_count in _iter_multi_response_chunks(
            client=client,
            prompt_records=headline_records,
            lang=lang,
        ):
            fallback_total += fallback_count
            multi_success_total += multi_success_count
            for offset, response in enumerate(chunk_outputs):
                if passed >= n_headline:
                    break
                user_prompt = headline_records[start + offset]["prompt"]
                if response and _filter_headline_response(response):
                    all_samples.append(
                        {
                            "messages": [
                                {"role": "user", "content": user_prompt},
                                {"role": "assistant", "content": response},
                            ]
                        }
                    )
                    passed += 1
                else:
                    failed += 1
            processed = min(len(headline_records), start + len(chunk_outputs))
            elapsed_s = time.time() - stage_start
            eta_text = _format_eta_by_passed(passed, n_headline, elapsed_s)
            print(
                f"    Progress: processed={processed}/{len(headline_records)}, "
                f"passed={passed}/{n_headline}, filtered={failed}, "
                f"multi_ok={multi_success_total}, fallback={fallback_total}, eta={eta_text}"
            )
            time.sleep(0.1)
            if passed >= n_headline:
                print(f"    Target {n_headline} reached, stopping early (processed={processed})")
                break

        print(f"  Headline Done: {passed} passed, {failed} filtered")
        if passed < n_headline:
            print(
                f"  Warning: Target not reached (Target {n_headline}, Actual {passed}), "
                "try increasing oversample_ratio or lowering group size."
            )

    # ---- Keyword Part ----
    if n_keyword > 0:
        print(f"\n  --- Keyword Subtask (Target: {n_keyword}) ---")
        stage_start = time.time()
        n_keyword_fetch = int(n_keyword * oversample_ratio)
        keyword_pairs = _generate_keyword_pairs(lang, n_keyword_fetch, seed=seed)

        passed = 0
        failed = 0
        fallback_total = 0
        multi_success_total = 0
        keyword_records = [
            {"word1": w1, "word2": w2, "prompt": build_keyword_prompt(w1, w2, lang)}
            for w1, w2 in keyword_pairs
        ]
        for start, chunk_outputs, fallback_count, multi_success_count in _iter_multi_response_chunks(
            client=client,
            prompt_records=keyword_records,
            lang=lang,
        ):
            fallback_total += fallback_count
            multi_success_total += multi_success_count
            for offset, response in enumerate(chunk_outputs):
                if passed >= n_keyword:
                    break
                record = keyword_records[start + offset]
                if response and _filter_keyword_response(response, record["word1"], record["word2"]):
                    all_samples.append(
                        {
                            "messages": [
                                {"role": "user", "content": record["prompt"]},
                                {"role": "assistant", "content": response},
                            ]
                        }
                    )
                    passed += 1
                else:
                    failed += 1
            processed = min(len(keyword_records), start + len(chunk_outputs))
            elapsed_s = time.time() - stage_start
            eta_text = _format_eta_by_passed(passed, n_keyword, elapsed_s)
            print(
                f"    Progress: processed={processed}/{len(keyword_records)}, "
                f"passed={passed}/{n_keyword}, filtered={failed}, "
                f"multi_ok={multi_success_total}, fallback={fallback_total}, eta={eta_text}"
            )
            time.sleep(0.1)
            if passed >= n_keyword:
                print(f"    Target {n_keyword} reached, stopping early (processed={processed})")
                break

        print(f"  Keyword Done: {passed} passed, {failed} filtered")
        if passed < n_keyword:
            print(
                f"  Warning: Target not reached (Target {n_keyword}, Actual {passed}), "
                "try increasing oversample_ratio or lowering group size."
            )

    print(f"\n  Total synthesized: {len(all_samples)} samples")
    return all_samples


def save_synthesized(samples: list[dict], lang: str) -> Path:
    """Save synthesized samples as JSONL file.

    Args:
        samples: Synthesized SFT sample list
        lang: Language code

    Returns:
        Path: Saved file path
    """
    SYNTHESIZED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = SYNTHESIZED_DIR / f"type_b_{lang}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return output_path


# ============================================================
# Command Line Entry
# ============================================================

def main():
    """Command line entry point, parse arguments and run synthesis."""
    parser = argparse.ArgumentParser(
        description="Synthesize Type B Task Formatted Data (Requires Gemini API)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=["en", "zh", "es", "all"],
        help="Target language, or 'all' to synthesize all three languages",
    )
    parser.add_argument(
        "--n_headline",
        type=int,
        default=70,
        help="Number of headline subtask samples per language (default 70)",
    )
    parser.add_argument(
        "--n_keyword",
        type=int,
        default=70,
        help="Number of keyword subtask samples per language (default 70)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default 42)",
    )

    args = parser.parse_args()

    languages = ["en", "zh", "es"] if args.lang == "all" else [args.lang]
    for lang in languages:
        print(f"\n{'=' * 60}")
        print(f"Synthesizing Type B data: lang={lang}")
        print(f"  headline: {args.n_headline} rows, keyword: {args.n_keyword} rows")
        print(f"{'=' * 60}")

        samples = synthesize_for_language(
            lang=lang,
            n_headline=args.n_headline,
            n_keyword=args.n_keyword,
            seed=args.seed,
        )
        output_path = save_synthesized(samples, lang)
        print(f"\nSaved: {output_path} ({len(samples)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
