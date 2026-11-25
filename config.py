"""Central configuration: paths, constants, seed themes, and credentials loaded from .env.

Run `python config.py` for a setup check. It never prints secret values.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths (relative to this file, so scripts work from any folder)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MOVIELENS_DIR = RAW_DIR / "ml-32m"                 # D37: MovieLens 32M (collected 10/2023), subsampled
SMALL_MOVIELENS_DIR = RAW_DIR / "ml-latest-small"  # the dataset used for the first runs, kept for reference
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"

for _folder in (RAW_DIR, CACHE_DIR, PROCESSED_DIR, RESULTS_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

# Caches of paid or slow API results. Re-runs skip anything already cached.
TMDB_CACHE_PATH = CACHE_DIR / "tmdb_cache.json"
THEMES_CACHE_PATH = CACHE_DIR / "themes_cache.json"
EXPLANATIONS_CACHE_PATH = CACHE_DIR / "explanations_cache.json"
LLM_USAGE_PATH = CACHE_DIR / "llm_usage.csv"

# Processed data
MOVIES_BASE_PATH = PROCESSED_DIR / "movies_base.csv"
MOVIES_ENRICHED_PATH = PROCESSED_DIR / "movies_enriched.csv"
MOVIE_THEMES_RAW_PATH = PROCESSED_DIR / "movie_themes_raw.csv"
MOVIE_THEMES_PATH = PROCESSED_DIR / "movie_themes.csv"
THEME_MAPPING_PATH = PROCESSED_DIR / "theme_mapping.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings_out"

# ---------------------------------------------------------------------------
# Data and split settings
# ---------------------------------------------------------------------------
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"   # 239 MB
# Subsampling (D37): the MAX_MOVIES most-rated movies that have a tmdbId, then SAMPLE_USERS random users
# (seed 42) with enough ratings on them, then the usual rating-count filters inside that sample.
MAX_MOVIES = 2500               # keeps the LLM bill small (spec 5.2 warns above ~2,500 movies); None = no limit
SAMPLE_USERS = 3000             # random users kept; None = all eligible users (1000 before D44)
# D44 (dataset v3): add recent films and sample recently active users, so 2015-2023 films are well covered.
RECENT_SINCE_YEAR = 2015        # films released in or after this year ...
RECENT_MIN_RATINGS = 50         # ... with at least this many MovieLens ratings join the top MAX_MOVIES
ACTIVE_USERS_SINCE = "2019-01-01"  # sample only users with at least one rating on or after this date
MIN_RATINGS_PER_MOVIE = 10      # counted inside the user sample (small-dataset runs used 20, then 10)
MIN_RATINGS_PER_USER = 20
TEST_FRACTION = 0.2             # the latest 20% of each user's ratings go to the test set
RELEVANT_THRESHOLD = 4.0        # a test rating at or above this counts as relevant
DROP_GENRES = {"(no genres listed)", "IMAX"}
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Recommender settings
# ---------------------------------------------------------------------------
LIKE_THRESHOLD = 4.0            # ratings at or above this build the user's theme profile
ALPHA = 0.8                     # final = ALPHA * graph score + (1 - ALPHA) * popularity
GENRE_WEIGHT = 0.3              # weight of the genre bonus inside the graph score
MIN_CANDIDATE_RATINGS = 5       # candidate movies need at least this many ratings
POOL_SIZE = 100                 # theme-matched candidates fetched before re-ranking
EVIDENCE_MAX_THEMES = 5         # shared themes shown in an explanation (3 before D45; films have at most 5)
EVIDENCE_MAX_LIKED = 3          # liked films listed per shared theme
SIMILAR_NEIGHBOURS = 50         # D48: SIMILAR edges per movie (nearest neighbours by rating patterns)
CF_WEIGHT = 0.0                 # D48: share of the similarity score inside the graph score (0 = themes/genres only)
CF_POOL_SIZE = 200              # candidates fetched through SIMILAR edges when CF_WEIGHT > 0
ALPHA_SIMILAR = ALPHA            # alpha used with the similarity blend (set from the validation grid, D49)

# ---------------------------------------------------------------------------
# External services
# ---------------------------------------------------------------------------
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w342"
TMDB_SLEEP_SECONDS = 0.05
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
NANOGPT_BASE_URL = "https://nano-gpt.com/api/v1"   # default LLM provider from Phase 2 onwards (D35)
APP_TITLE = "Movie KG Recommender"   # sent to OpenRouter as the optional X-Title header (not sent to NanoGPT)

# ---------------------------------------------------------------------------
# Theme extraction and normalization
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
THEME_SIM_THRESHOLD = 0.75      # cosine similarity needed to map a new theme onto a seed theme
MIN_MOVIES_PER_THEME = 3        # rarer themes are dropped, unless a movie would be left with none

# Manual review of the normalized themes (task 2.7). Applied after the seed / embedding mapping.
# raw theme -> canonical theme; mapping a theme to itself keeps it separate from its embedding match.
THEME_OVERRIDES = {
    "vengeance": "revenge",
    "revenger": "revenge",            # LLM typo
    "revengue": "revenge",            # LLM typo
    "morality": "moral dilemma",
    "obsessive love": "obsession",
    "ambition": "ambition",           # general ambition (power, career) is not "artistic ambition"
    "madness": "mental illness",      # dataset v3 review (D46)
    "revolution": "rebellion",        # dataset v3 review (D46)
}
# Genre words that duplicate HAS_GENRE edges (D17); dropped as themes.
THEME_BLOCKLIST = {"romance", "adventure", "crime"}   # "crime" added in the dataset v3 review (D46)

# Seed vocabulary (spec section 6.3). The LLM is told to prefer these so that movies share themes.
# "fear of the unknown" is the only seed with 4 words; seed themes are always accepted as they are.
SEED_THEMES = [
    "redemption", "found family", "coming of age", "revenge", "betrayal", "sacrifice", "survival",
    "man vs nature", "man vs machine", "first contact", "artificial intelligence", "time travel",
    "identity", "loss and grief", "forbidden love", "friendship", "loyalty", "corruption",
    "power and politics", "class struggle", "justice", "war and trauma", "freedom", "obsession",
    "isolation", "fear of the unknown", "underdog triumph", "quest", "heist", "conspiracy",
    "family conflict", "mentor and student", "dystopia", "technology and society", "good vs evil",
    "moral dilemma", "small-town life", "road trip", "rise and fall", "second chances", "deception",
    "mental illness", "addiction", "immigration", "racism and prejudice", "childhood innocence",
    "nostalgia", "apocalypse", "supernatural", "artistic ambition",
]

# ---------------------------------------------------------------------------
# Credentials from .env
# ---------------------------------------------------------------------------
load_dotenv(ENV_PATH)

# Optional reasoning effort ("minimal", "low", "medium", "high"). Empty = don't send it.
# Some models (e.g. google/gemini-3.5-flash-lite) always reason; "minimal" keeps that short and cheap.
# LLM_REASONING_EFFORT is the new name; OPENROUTER_REASONING_EFFORT still works.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", os.getenv("OPENROUTER_REASONING_EFFORT", "minimal")).strip()

# Example values from .env.example that still need replacing.
_PLACEHOLDERS = {
    "LLM_API_KEY": "your_llm_key_here",
    "LLM_MODEL": "put_a_model_id_here",
    "OPENROUTER_API_KEY": "your_openrouter_key_here",
    "OPENROUTER_MODEL": "put_a_model_id_here",
    "TMDB_API_KEY": "your_tmdb_key_here",
}


def missing_env(*names: str) -> list[str]:
    """Return the names that are unset, empty, or still hold the .env.example placeholder."""
    missing = []
    for name in names:
        value = os.getenv(name, "").strip()
        if not value or value == _PLACEHOLDERS.get(name):
            missing.append(name)
    return missing


def require_env(*names: str) -> dict[str, str]:
    """Return the requested settings, or exit with a clear message if any are missing."""
    missing = missing_env(*names)
    if missing:
        raise SystemExit(
            f"Missing setting(s) in .env: {', '.join(missing)}.\n"
            f"Copy .env.example to .env in {BASE_DIR} and fill them in."
        )
    return {name: os.environ[name].strip() for name in names}


def get_neo4j_settings() -> dict[str, str]:
    """Neo4j connection settings: uri, user, password."""
    env = require_env("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
    return {"uri": env["NEO4J_URI"], "user": env["NEO4J_USER"], "password": env["NEO4J_PASSWORD"]}


def get_tmdb_key() -> str:
    """TMDB credential (v3 API key or v4 read access token)."""
    return require_env("TMDB_API_KEY")["TMDB_API_KEY"]


def get_llm_settings() -> dict[str, str]:
    """LLM provider settings: api_key, model, base_url, and provider ('nanogpt', 'openrouter' or 'custom').

    New style: LLM_API_KEY and LLM_MODEL, optionally LLM_BASE_URL (default NanoGPT).
    Old style (still works): OPENROUTER_API_KEY and OPENROUTER_MODEL, which use OpenRouter.
    """
    if not missing_env("LLM_API_KEY"):
        key, default_url = os.environ["LLM_API_KEY"].strip(), NANOGPT_BASE_URL
    elif not missing_env("OPENROUTER_API_KEY"):
        key, default_url = os.environ["OPENROUTER_API_KEY"].strip(), OPENROUTER_BASE_URL
    else:
        raise SystemExit(f"Missing setting in .env: LLM_API_KEY.\nAdd it to {ENV_PATH} (see .env.example).")
    model = next((os.environ[n].strip() for n in ("LLM_MODEL", "OPENROUTER_MODEL") if not missing_env(n)), None)
    if model is None:
        raise SystemExit(f"Missing setting in .env: LLM_MODEL (a model id, e.g. google/gemini-3.5-flash-lite).")
    base_url = os.getenv("LLM_BASE_URL", "").strip().rstrip("/") or default_url
    provider = "nanogpt" if "nano-gpt.com" in base_url else "openrouter" if "openrouter.ai" in base_url else "custom"
    return {"api_key": key, "model": model, "base_url": base_url, "provider": provider}


# ---------------------------------------------------------------------------
# Setup check: python config.py
# ---------------------------------------------------------------------------
def _check_packages() -> bool:
    """Compare installed package versions with the pins in requirements.txt."""
    from importlib.metadata import PackageNotFoundError, version

    all_installed = True
    for line in (BASE_DIR / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        name, pinned = line.split("==")
        try:
            installed = version(name)
        except PackageNotFoundError:
            print(f"  {name:<14} MISSING (run: pip install -r requirements.txt)")
            all_installed = False
            continue
        note = "ok" if installed == pinned else f"installed, but the pin is {pinned}"
        print(f"  {name:<14} {installed:<9} {note}")
    return all_installed


def _check_settings() -> None:
    """Show which settings are present, without printing secret values."""
    if not ENV_PATH.exists():
        print("  .env not found. Create it: cp .env.example .env  (Windows: copy .env.example .env)")
    not_secret = {"NEO4J_URI", "NEO4J_USER"}
    needed_by = {
        "NEO4J_URI": "Phase 0", "NEO4J_USER": "Phase 0", "NEO4J_PASSWORD": "Phase 0",
        "TMDB_API_KEY": "Phase 1",
    }
    for name, phase in needed_by.items():
        if missing_env(name):
            print(f"  {name:<19} missing (needed from {phase})")
        elif name in not_secret:
            print(f"  {name:<19} {os.environ[name].strip()}")
        else:
            print(f"  {name:<19} set")
    try:  # the LLM provider: key presence only, never the key itself
        llm = get_llm_settings()
        print(f"  {'LLM':<19} {llm['provider']}, model {llm['model']} (key set)")
    except SystemExit:
        print(f"  {'LLM_API_KEY/MODEL':<19} missing (needed from Phase 2)")
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    if password and len(password) < 8:
        print("  Warning: NEO4J_PASSWORD must be at least 8 characters for Neo4j.")
    if password == "change_me_min8chars":
        print("  Warning: NEO4J_PASSWORD still has the example value. Pick your own before the first start.")


def _check_neo4j() -> bool:
    """Try to connect to Neo4j and print the server version."""
    if missing_env("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        print("  skipped: Neo4j settings are missing in .env")
        return False
    from neo4j import GraphDatabase
    from neo4j.exceptions import AuthError, ServiceUnavailable

    settings = get_neo4j_settings()
    try:
        with GraphDatabase.driver(
            settings["uri"],
            auth=(settings["user"], settings["password"]),
            connection_timeout=5,
        ) as driver:
            driver.verify_connectivity()
            info = driver.get_server_info()
        print(f"  Neo4j connection OK ({info.agent} at {settings['uri']})")
        return True
    except AuthError:
        print("  Login failed: check NEO4J_USER and NEO4J_PASSWORD.")
        print("  (Docker sets the password only on the first start with an empty neo4j_data/ folder.)")
    except ServiceUnavailable:
        print(f"  Neo4j is not reachable at {settings['uri']}. Is it running? Try: docker compose up -d")
    except Exception as exc:  # anything else, shown plainly
        print(f"  Neo4j check failed: {type(exc).__name__}: {exc}")
    return False


def _self_check() -> int:
    """Print a setup report and return an exit code (0 means the Phase 0 checks passed)."""
    print("Setup check: movie-kg-recsys\n")
    py = sys.version_info
    python_ok = py >= (3, 10)
    verdict = "ok" if python_ok else "too old: need 3.10 or newer (3.12 recommended)"
    print(f"Python {py.major}.{py.minor}.{py.micro}: {verdict}")

    print("\nPackages (pins from requirements.txt):")
    packages_ok = _check_packages()

    print(f"\nSettings ({ENV_PATH}):")
    _check_settings()

    print("\nNeo4j:")
    neo4j_ok = _check_neo4j()

    passed = python_ok and packages_ok and neo4j_ok
    print("\nResult:", "Phase 0 checks passed." if passed else "not ready yet, see the messages above.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(_self_check())
