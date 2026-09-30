"""Phase 2, step 1: extract 3-5 narrative themes per movie with the LLM (cached and resumable).

Usage:
    python scripts/03_extract_themes.py --limit 10      # pilot: at most 10 new movies
    python scripts/03_extract_themes.py                 # every movie not cached yet (asks y/n first)
    python scripts/03_extract_themes.py --yes           # skip the y/n question
    python scripts/03_extract_themes.py --retry-failed  # also retry movies recorded with no valid themes
Outputs: data/cache/themes_cache.json, data/processed/movie_themes_raw.csv (movieId, theme),
         results/theme_extraction_stats.json. Every API call is logged in data/cache/llm_usage.csv.
"""
import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config and src importable

import config  # noqa: E402
from src import data_loading as dl  # noqa: E402
from src import llm_client  # noqa: E402
from src import theme_extraction as te  # noqa: E402
from src.cache import load_json  # noqa: E402

MODELS_URL = "https://openrouter.ai/api/v1/models"  # public model list with prices (no key needed)
EST_OUTPUT_TOKENS = 40                              # a JSON list of 3-5 short themes
STATS_PATH = config.RESULTS_DIR / "theme_extraction_stats.json"


def model_prices(model: str) -> tuple[float, float] | None:
    """(USD per prompt token, USD per completion token) from OpenRouter's public list, or None."""
    try:
        models = requests.get(MODELS_URL, timeout=15).json()["data"]
        pricing = next(m for m in models if m["id"] == model)["pricing"]
        return float(pricing["prompt"]), float(pricing["completion"])
    except Exception:  # offline, renamed model, or a format change: just skip the estimate
        return None


def print_estimate(todo: pd.DataFrame, tag_map: dict, model: str) -> None:
    """Planned calls plus a rough token and cost estimate (about 4 characters per token)."""
    chars = sum(
        len(msg["content"])
        for movie in todo.to_dict("records")
        for msg in te.build_messages(movie, tag_map.get(int(movie["movieId"]), []))
    )
    tokens_in, tokens_out = chars / 4, EST_OUTPUT_TOKENS * len(todo)
    print(f"  model: {model}")
    print(f"  planned LLM calls: {len(todo):,} (up to {2 * len(todo):,} if every reply needs its one retry)")
    print(f"  rough tokens: ~{tokens_in:,.0f} in, ~{tokens_out:,.0f} out, plus any hidden reasoning tokens")
    prices = model_prices(model)
    if prices:
        cost = tokens_in * prices[0] + tokens_out * prices[1]
        print(f"  rough cost at list prices: ~${cost:.3f} "
              f"(${prices[0] * 1e6:.2f} in / ${prices[1] * 1e6:.2f} out per 1M tokens)")
    else:
        print("  rough cost: unavailable (couldn't read OpenRouter's price list)")
    if model.endswith(":free"):
        print("  Note: free models allow 50 requests a day (1,000 after buying $10+ of credits).")


def confirm(skip: bool) -> bool:
    """Ask y/n before spending money (spec 6.5). --yes skips the question."""
    if skip:
        return True
    try:
        return input("Start the LLM calls? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def usage_rows() -> pd.DataFrame:
    """Every logged LLM call so far (an empty table if there are none)."""
    if not config.LLM_USAGE_PATH.exists():
        return pd.DataFrame(columns=llm_client.USAGE_FIELDS)
    return pd.read_csv(config.LLM_USAGE_PATH)


def usage_summary(rows: pd.DataFrame) -> dict:
    """Call, token, and cost totals for a set of logged calls."""
    total = lambda col: pd.to_numeric(rows[col], errors="coerce").fillna(0).sum()  # noqa: E731
    return {
        "calls": len(rows),
        "prompt_tokens": int(total("prompt_tokens")),
        "completion_tokens": int(total("completion_tokens")),
        "reasoning_tokens": int(total("reasoning_tokens")),
        "cost_usd": round(float(total("cost_usd")), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract narrative themes for each movie with the LLM.")
    parser.add_argument("--limit", type=int, default=None, help="process at most N new movies (pilot runs)")
    parser.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    parser.add_argument("--retry-failed", action="store_true", help="retry movies recorded with no valid themes")
    args = parser.parse_args()

    if not config.MOVIES_ENRICHED_PATH.exists():
        raise SystemExit("movies_enriched.csv not found. Run scripts/01_prepare_data.py and 02_fetch_tmdb.py first.")
    movies = pd.read_csv(config.MOVIES_ENRICHED_PATH)
    tag_map = dl.top_tags(dl.load_movielens()["tags"], movies["movieId"])
    cache = load_json(config.THEMES_CACHE_PATH)

    ids = movies["movieId"].astype(str)
    status = ids.map(lambda i: cache.get(i, {}).get("status"))
    todo = movies[status.isna() | ((status == "invalid_output") & args.retry_failed)]
    total_todo = len(todo)
    if args.limit is not None:
        todo = todo.head(args.limit)
    print(f"[1/3] Plan: {len(movies):,} movies with a plot; {(status == 'ok').sum():,} already have themes, "
          f"{(status == 'invalid_output').sum():,} recorded as failed; {total_todo:,} to do, {len(todo):,} in this run")

    run = None
    if not todo.empty:
        model = llm_client.model_name()
        print_estimate(todo, tag_map, model)
        if not confirm(args.yes):
            print("Cancelled. No LLM calls were made.")
            return
        before = len(usage_rows())
        started = time.perf_counter()
        try:
            counts = te.run_extraction(todo, tag_map, cache)
        except llm_client.LLMFatalError as exc:
            raise SystemExit(f"Stopped: {exc}\nThemes extracted so far are saved in the cache.")
        seconds = round(time.perf_counter() - started, 1)
        run = {"movies": len(todo), **counts, "seconds": seconds, "usage": usage_summary(usage_rows().iloc[before:])}
        u = run["usage"]
        print(f"  this run: {counts['ok']} ok, {counts['invalid_output']} invalid output, "
              f"{counts['api_error']} API errors, {seconds} s")
        print(f"  tokens: {u['prompt_tokens']:,} in, {u['completion_tokens']:,} out "
              f"({u['reasoning_tokens']:,} of those hidden reasoning), {u['calls']} calls, cost ${u['cost_usd']:.4f}")

    print("[2/3] Writing movie_themes_raw.csv")
    pairs = te.cache_to_pairs(cache, movies["movieId"])
    pairs.to_csv(config.MOVIE_THEMES_RAW_PATH, index=False)
    seed_share = float(pairs["theme"].isin(config.SEED_THEMES).mean()) if len(pairs) else 0.0
    print(f"  wrote {config.MOVIE_THEMES_RAW_PATH.relative_to(config.BASE_DIR)} ({len(pairs):,} rows, "
          f"{pairs['movieId'].nunique():,} movies, {pairs['theme'].nunique():,} distinct themes; "
          f"{seed_share:.0%} of theme labels are seed themes)")

    print("[3/3] Sample results")
    pool = [str(i) for i in (todo["movieId"] if not todo.empty else movies["movieId"])
            if cache.get(str(i), {}).get("status") == "ok"]
    random.Random(config.RANDOM_SEED).shuffle(pool)
    for mid in pool[:5]:
        print(f"  {cache[mid]['title']}: {', '.join(cache[mid]['themes'])}")

    status = ids.map(lambda i: cache.get(i, {}).get("status"))
    per_movie = pairs.groupby("movieId").size()
    extraction_calls = usage_rows()
    extraction_calls = extraction_calls[extraction_calls["purpose"] == "theme_extraction"]
    stats = {
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "models": sorted({e.get("model") for e in cache.values() if e.get("model")}),
        "movies_with_plot": len(movies),
        "movies_ok": int((status == "ok").sum()),
        "movies_invalid_output": int((status == "invalid_output").sum()),
        "movies_not_done": int(status.isna().sum()),
        "movies_needing_retry": sum(1 for e in cache.values() if e.get("attempts", 1) > 1),
        "raw_pairs": len(pairs),
        "distinct_raw_themes": int(pairs["theme"].nunique()),
        "share_of_labels_from_seed_list": round(seed_share, 3),
        "themes_per_movie": {
            "mean": round(float(per_movie.mean()), 2) if len(per_movie) else None,
            "min": int(per_movie.min()) if len(per_movie) else None,
            "max": int(per_movie.max()) if len(per_movie) else None,
        },
        "usage_all_runs": usage_summary(extraction_calls),
        "last_run": run,
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"  wrote {STATS_PATH.relative_to(config.BASE_DIR)}")


if __name__ == "__main__":
    main()
