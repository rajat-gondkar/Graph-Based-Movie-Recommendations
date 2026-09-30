"""Phase 1, step 2: fetch plot summaries and posters from TMDB (cached and resumable).

Usage:
    python scripts/02_fetch_tmdb.py --limit 5         # quick test: at most 5 new API calls
    python scripts/02_fetch_tmdb.py                   # everything that isn't cached yet
    python scripts/02_fetch_tmdb.py --retry-missing   # also retry movies TMDB reported as not found
Outputs: data/processed/movies_enriched.csv (usable movies), data/processed/movies_dropped.csv,
         results/tmdb_stats.json. Cache: data/cache/tmdb_cache.json
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config and src importable

import config  # noqa: E402
from src import tmdb_client as tmdb  # noqa: E402
from src.cache import load_json, save_json  # noqa: E402

SAVE_EVERY = 25  # write the cache to disk every N new responses (and always on exit)
DROPPED_PATH = config.PROCESSED_DIR / "movies_dropped.csv"
STATS_PATH = config.RESULTS_DIR / "tmdb_stats.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def fetch_missing(tmdb_ids: list[int], cache: dict, limit: int | None, retry_missing: bool) -> dict:
    """Fetch every tmdbId that isn't cached yet (plus cached 404s with --retry-missing)."""
    todo = [i for i in dict.fromkeys(tmdb_ids)
            if str(i) not in cache or (retry_missing and cache[str(i)].get("status") == 404)]
    total_todo = len(todo)
    if limit is not None:
        todo = todo[:limit]
    print(f"[1/3] Fetching from TMDB: {len(tmdb_ids) - total_todo:,} movies already cached, "
          f"{total_todo:,} still to fetch, fetching {len(todo):,} now")
    run = {"fetched": 0, "failed": 0, "seconds": 0.0}
    if not todo:
        return run

    session = tmdb.make_session(config.get_tmdb_key())
    started = time.perf_counter()
    try:
        for n, tmdb_id in enumerate(tqdm(todo, desc="TMDB", unit="movie"), start=1):
            try:
                entry = tmdb.fetch_movie(session, tmdb_id)
            except RuntimeError as exc:
                run["failed"] += 1
                tqdm.write(f"  tmdbId {tmdb_id}: {exc} (not cached, a re-run will retry it)")
                continue
            entry["fetched_utc"] = utc_now()
            cache[str(tmdb_id)] = entry
            run["fetched"] += 1
            if n % SAVE_EVERY == 0:
                save_json(cache, config.TMDB_CACHE_PATH)
            time.sleep(config.TMDB_SLEEP_SECONDS)
    finally:
        save_json(cache, config.TMDB_CACHE_PATH)  # also runs on Ctrl+C, so nothing fetched is lost
        run["seconds"] = round(time.perf_counter() - started, 1)
    return run


def build_tables(movies: pd.DataFrame, cache: dict) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Join TMDB fields onto movies_base. Returns (usable movies, dropped movies, number not fetched yet)."""
    fields = pd.DataFrame([tmdb.movie_fields(cache.get(str(i))) for i in movies["tmdbId"]])
    table = pd.concat([movies.reset_index(drop=True), fields], axis=1)
    status = pd.Series([cache.get(str(i), {}).get("status") for i in movies["tmdbId"]])
    found = status == 200
    has_overview = table["overview"] != ""
    reason = pd.Series("", index=table.index)
    reason[status == 404] = "not found on TMDB (404)"
    reason[found & ~has_overview] = "empty overview"
    dropped = table.loc[reason != "", ["movieId", "title", "year", "tmdbId"]].assign(reason=reason[reason != ""])
    usable = table[found & has_overview].reset_index(drop=True)
    return usable, dropped, int(status.isna().sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch TMDB plots and posters for movies_base.csv.")
    parser.add_argument("--limit", type=int, default=None, help="fetch at most N new movies (for testing)")
    parser.add_argument("--retry-missing", action="store_true", help="retry movies cached as not found (404)")
    args = parser.parse_args()

    if not config.MOVIES_BASE_PATH.exists():
        raise SystemExit("movies_base.csv not found. Run: python scripts/01_prepare_data.py")
    movies = pd.read_csv(config.MOVIES_BASE_PATH, dtype={"tmdbId": "Int64", "year": "Int64"})
    cache = load_json(config.TMDB_CACHE_PATH)
    run = fetch_missing(movies["tmdbId"].astype(int).tolist(), cache, args.limit, args.retry_missing)
    print(f"  this run: {run['fetched']:,} fetched, {run['failed']:,} failed, {run['seconds']} s")

    print("[2/3] Building movies_enriched.csv")
    usable, dropped, not_fetched = build_tables(movies, cache)
    usable.to_csv(config.MOVIES_ENRICHED_PATH, index=False)
    dropped.to_csv(DROPPED_PATH, index=False)
    print(f"  wrote {config.MOVIES_ENRICHED_PATH.relative_to(config.BASE_DIR)} ({len(usable):,} movies with a plot)")
    print(f"  wrote {DROPPED_PATH.relative_to(config.BASE_DIR)} ({len(dropped):,} movies)")
    for _, row in dropped.iterrows():
        print(f"  Warning: dropping '{row['title']}' ({row['year']}), tmdbId {row['tmdbId']}: {row['reason']}")
    if not_fetched:
        print(f"  {not_fetched:,} movies aren't fetched yet. Run again without --limit to finish.")

    print("[3/3] Stats")
    words = usable["overview"].str.split().str.len()
    stats = {
        "created_utc": utc_now(),
        "movies_in_base": len(movies),
        "cached_responses": sum(str(i) in cache for i in movies["tmdbId"]),
        "not_fetched_yet": not_fetched,
        "not_found_404": int((dropped["reason"] == "not found on TMDB (404)").sum()),
        "empty_overview": int((dropped["reason"] == "empty overview").sum()),
        "usable_with_overview": len(usable),
        "with_poster": int((usable["poster_path"] != "").sum()),
        "with_tagline": int((usable["tagline"] != "").sum()),
        "overview_words": {
            "mean": round(float(words.mean()), 1) if len(usable) else None,
            "median": float(words.median()) if len(usable) else None,
            "min": int(words.min()) if len(usable) else None,
            "max": int(words.max()) if len(usable) else None,
        },
        "last_run": {**run, "limit": args.limit, "retry_missing": args.retry_missing},
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"  usable: {stats['usable_with_overview']:,} | posters: {stats['with_poster']:,} | "
          f"taglines: {stats['with_tagline']:,} | overview length: {stats['overview_words']['mean']} words on average")
    print(f"  wrote {STATS_PATH.relative_to(config.BASE_DIR)}")


if __name__ == "__main__":
    main()
