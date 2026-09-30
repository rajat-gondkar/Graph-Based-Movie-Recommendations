"""Theme extraction (spec section 6): the prompt, output cleaning, and the cached extraction loop."""
import re
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

import config
from src import llm_client
from src.cache import save_json

# Spec 6.4, verbatim.
SYSTEM_PROMPT = """You are a film analyst. You read a movie's metadata and plot summary and return its core narrative THEMES.
Rules:
- Return 3 to 5 themes.
- Each theme is 1 to 3 lowercase words.
- Prefer themes from the provided seed list. Only invent a new theme if none of the seed themes fits.
- Themes describe what the story is ABOUT (ideas, conflicts, emotional arcs), not the genre and not character names.
- Respond with ONLY valid JSON, no markdown fences, no commentary, in exactly this form:
{"themes": ["theme one", "theme two", "theme three"]}"""

USER_TEMPLATE = """Seed themes: {seed_list_comma_separated}

Title: {title} ({year})
Genres: {genres}
Tagline: {tagline}
Popular user tags: {top_tags_or_none}
Plot: {overview}"""

MAX_THEMES = 5


def _key(theme: str) -> str:
    """Matching key that treats hyphens like spaces: 'coming-of-age' and 'coming of age' match."""
    return re.sub(r"[-\s]+", " ", theme).strip()


_SEED_BY_KEY = {_key(s): s for s in config.SEED_THEMES}


def _text(value, default: str) -> str:
    """A CSV cell as clean text, or the default when it's empty or missing."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    value = str(value).strip()
    return value or default


def clean_theme(raw) -> str | None:
    """Tidy one theme label (D15): lowercase, '&' -> 'and', letters/digits/spaces/hyphens only.

    Exact seed themes are always kept (one seed has 4 words); anything else over 3 words is discarded.
    """
    if not isinstance(raw, str):
        return None
    theme = raw.lower().replace("&", " and ")
    theme = re.sub(r"[.'\"`\u2018\u2019\u201c\u201d]", "", theme)  # "vs." -> "vs", drop quotes and apostrophes
    theme = re.sub(r"[^a-z0-9\- ]+", " ", theme)                      # other punctuation becomes a space
    theme = re.sub(r"\s+", " ", theme).strip(" -")
    if not theme:
        return None
    seed = _SEED_BY_KEY.get(_key(theme))
    if seed:
        return seed
    return theme if len(theme.split()) <= 3 else None


def parse_themes(reply: str) -> list[str] | None:
    """Validate a reply (spec 6.5): {"themes": [strings]} -> up to 5 cleaned, unique themes, or None."""
    obj = llm_client.parse_json_object(reply)
    themes = obj.get("themes") if obj else None
    if not isinstance(themes, list) or not all(isinstance(t, str) for t in themes):
        return None
    cleaned: list[str] = []
    for raw in themes:
        theme = clean_theme(raw)
        if theme and theme not in cleaned:
            cleaned.append(theme)
    return cleaned[:MAX_THEMES] or None


def build_messages(movie: dict, tags: list[str]) -> list[dict]:
    """System and user messages for one movie (spec 6.4 template)."""
    genres = ", ".join(g for g in _text(movie.get("genres"), "").split("|") if g) or "unknown"
    user = USER_TEMPLATE.format(
        seed_list_comma_separated=", ".join(config.SEED_THEMES),
        title=_text(movie.get("title"), "unknown"),
        year=_text(movie.get("year"), "unknown").removesuffix(".0"),
        genres=genres,
        tagline=_text(movie.get("tagline"), "none"),
        top_tags_or_none=", ".join(tags) if tags else "none",
        overview=_text(movie.get("overview"), "none"),
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def extract_one(movie: dict, tags: list[str]) -> dict:
    """Ask the LLM for one movie's themes. One retry on invalid output, then record an empty list (spec 6.5)."""
    messages = build_messages(movie, tags)
    reply = ""
    for attempt in (1, 2):
        reply = llm_client.chat(messages, max_tokens=200, temperature=0.2, purpose="theme_extraction")
        themes = parse_themes(reply)
        if themes:
            return {"themes": themes, "status": "ok", "attempts": attempt, "raw_reply": reply[:300]}
    return {"themes": [], "status": "invalid_output", "attempts": 2, "raw_reply": reply[:300]}


def run_extraction(todo: pd.DataFrame, tag_map: dict[int, list[str]], cache: dict) -> dict:
    """Extract themes for every movie in `todo`, writing the cache to disk after each movie.

    API failures (after retries) aren't cached, so a re-run tries those movies again.
    LLMFatalError (bad key, no credits, unknown model) is passed up so the caller can stop.
    """
    counts = {"ok": 0, "invalid_output": 0, "api_error": 0}
    model = llm_client.model_name()
    for movie in tqdm(todo.to_dict("records"), desc="Themes", unit="movie"):
        movie_id = int(movie["movieId"])
        try:
            result = extract_one(movie, tag_map.get(movie_id, []))
        except llm_client.LLMError as exc:
            counts["api_error"] += 1
            tqdm.write(f"  {movie['title']}: {exc} (not cached; a re-run will retry it)")
            continue
        if result["status"] != "ok":
            tqdm.write(f"  {movie['title']}: no valid themes after 2 tries, recorded as empty")
        counts[result["status"]] += 1
        cache[str(movie_id)] = {
            "title": movie["title"],
            **result,
            "model": model,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_json(cache, config.THEMES_CACHE_PATH)  # after every movie, so nothing paid for is lost
    return counts


def cache_to_pairs(cache: dict, movie_ids) -> pd.DataFrame:
    """(movieId, theme) rows for the given movies that have valid themes in the cache."""
    rows = [
        {"movieId": int(mid), "theme": theme}
        for mid in map(str, movie_ids)
        if cache.get(mid, {}).get("status") == "ok"
        for theme in cache[mid]["themes"]
    ]
    return pd.DataFrame(rows, columns=["movieId", "theme"])
