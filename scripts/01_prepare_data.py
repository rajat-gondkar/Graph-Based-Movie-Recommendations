"""Phase 1, step 1: get MovieLens 32M, shrink it to demo size, clean titles, and split train/test.

Usage:
    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py --max-movies 2000 --users 800     # smaller graph, lower LLM cost
    python scripts/01_prepare_data.py --max-movies 0 --users 0          # no limits (very large!)
Outputs: data/processed/movies_base.csv, train.csv, test.csv, and results/dataset_stats.json
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config and src importable

import config  # noqa: E402
from src import data_loading as dl  # noqa: E402

COST_WARNING_MOVIES = 2500  # above this, theme extraction gets expensive (spec 5.2)


def to_date(ts: int) -> str:
    """Unix timestamp -> 'YYYY-MM-DD' (UTC)."""
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def split_stats(train, test) -> dict:
    """Numbers describing the train/test split, for the report."""
    per_user_test = test.groupby("userId").size()
    relevant = test[test["rating"] >= config.RELEVANT_THRESHOLD]
    return {
        "train_ratings": len(train),
        "test_ratings": len(test),
        "train_ratings_ge_4": int((train["rating"] >= 4.0).sum()),
        "test_ratings_ge_4": len(relevant),
        "users": int(train["userId"].nunique()),
        "users_with_relevant_test_item": int(relevant["userId"].nunique()),
        "test_items_per_user": {
            "min": int(per_user_test.min()),
            "mean": round(float(per_user_test.mean()), 1),
            "max": int(per_user_test.max()),
        },
        "train_period": [to_date(train["timestamp"].min()), to_date(train["timestamp"].max())],
        "test_period": [to_date(test["timestamp"].min()), to_date(test["timestamp"].max())],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the MovieLens subset and the train/test split.")
    parser.add_argument("--max-movies", type=int, default=config.MAX_MOVIES,
                        help="keep the N most-rated movies (0 = no limit; default %(default)s)")
    parser.add_argument("--users", type=int, default=config.SAMPLE_USERS,
                        help="random users to keep (0 = all eligible; default %(default)s)")
    parser.add_argument("--min-movie-ratings", type=int, default=config.MIN_RATINGS_PER_MOVIE,
                        help="keep movies with at least this many ratings in the user sample (default %(default)s)")
    parser.add_argument("--min-user-ratings", type=int, default=config.MIN_RATINGS_PER_USER,
                        help="keep users with at least this many remaining ratings (default %(default)s)")
    args = parser.parse_args()
    max_movies, n_users = args.max_movies or None, args.users or None
    started = time.perf_counter()

    print("[1/4] MovieLens files")
    raw = dl.load_movielens(dl.download_movielens())
    raw_stats = dl.rating_stats(raw["ratings"])
    print(f"  raw: {len(raw['movies']):,} movies listed ({raw_stats['movies']:,} rated), "
          f"{raw_stats['users']:,} users, {raw_stats['ratings']:,} ratings, {len(raw['tags']):,} tags")

    print(f"[2/4] Sampling: top {max_movies or 'all'} movies by rating count (with a tmdbId), "
          f"{n_users or 'all eligible'} random users (seed {config.RANDOM_SEED}), then movies >= "
          f"{args.min_movie_ratings} and users >= {args.min_user_ratings} ratings inside the sample")
    ratings, funnel = dl.sample_subset(raw["ratings"], raw["links"], max_movies, n_users,
                                       args.min_movie_ratings, args.min_user_ratings)
    movies = dl.build_movies_base(raw["movies"], raw["links"], ratings["movieId"].unique())
    kept = dl.rating_stats(ratings)
    print(f"  top movies: {funnel['top_movies_kept']:,} (the last one has {funnel['rating_count_of_last_top_movie']:,} "
          f"ratings overall); eligible users: {funnel['users_eligible']:,} of {funnel['users_total']:,}; "
          f"sampled: {funnel['users_sampled']:,}")
    print(f"  kept: {kept['movies']:,} movies, {kept['users']:,} users, {kept['ratings']:,} ratings "
          f"(density {kept['density']:.3f})")
    if kept["movies"] > COST_WARNING_MOVIES:
        print(f"  Warning: {kept['movies']:,} movies means about that many LLM calls in Phase 2.\n"
              f"  To cut cost: python scripts/01_prepare_data.py --max-movies 2000")

    print(f"[3/4] Train/test split: latest {config.TEST_FRACTION:.0%} of each user's ratings go to test")
    train, test = dl.split_by_time(ratings, config.TEST_FRACTION)
    split = split_stats(train, test)
    print(f"  train: {split['train_ratings']:,} ratings | test: {split['test_ratings']:,} ratings "
          f"({split['test_ratings_ge_4']:,} relevant, rating >= {config.RELEVANT_THRESHOLD})")
    print(f"  users with at least one relevant test item: {split['users_with_relevant_test_item']:,} "
          f"of {split['users']:,}")

    print("[4/4] Writing files")
    movies.to_csv(config.MOVIES_BASE_PATH, index=False)
    train.to_csv(config.TRAIN_PATH, index=False)
    test.to_csv(config.TEST_PATH, index=False)

    tags = dl.top_tags(raw["tags"], movies["movieId"])
    genre_counts = movies["genres"].str.split("|").explode().replace("", None).dropna().value_counts()
    stats = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": config.MOVIELENS_DIR.name,
        "thresholds": {
            "max_movies": max_movies,
            "sample_users": n_users,
            "random_seed": config.RANDOM_SEED,
            "min_movie_ratings": args.min_movie_ratings,
            "min_user_ratings": args.min_user_ratings,
            "test_fraction": config.TEST_FRACTION,
            "relevant_threshold": config.RELEVANT_THRESHOLD,
        },
        "raw": {
            "movies_listed": len(raw["movies"]),
            "tag_applications": len(raw["tags"]),
            "movies_without_tmdbId": int(raw["links"]["tmdbId"].isna().sum()),
            "latest_rating": to_date(raw["ratings"]["timestamp"].max()),
            **raw_stats,
        },
        "sampling": funnel,
        "filtered": {
            **kept,
            "movies_with_user_tags": len(tags),
            "movies_without_year": int(movies["year"].isna().sum()),
            "year_range": [int(movies["year"].min()), int(movies["year"].max())],
            "duplicate_tmdbIds": int(movies["tmdbId"].duplicated().sum()),
            "titles_changed_by_cleaning": int(
                (movies["ml_title"] != movies["title"] + " (" + movies["year"].astype(str) + ")").sum()
            ),
        },
        "split": split,
        "rating_distribution_filtered": {
            str(k): int(v) for k, v in ratings["rating"].value_counts().sort_index().items()
        },
        "movies_per_genre_filtered": {k: int(v) for k, v in genre_counts.items()},
    }
    stats_path = config.RESULTS_DIR / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    for path, rows in ((config.MOVIES_BASE_PATH, len(movies)), (config.TRAIN_PATH, len(train)),
                       (config.TEST_PATH, len(test))):
        print(f"  wrote {path.relative_to(config.BASE_DIR)} ({rows:,} rows)")
    print(f"  wrote {stats_path.relative_to(config.BASE_DIR)}")

    examples = movies[movies["ml_title"].str.contains(", The \\(|, A \\(|\\(a\\.k\\.a", regex=True)].head(5)
    print("  title cleaning examples:")
    for _, row in examples.iterrows():
        print(f"    {row['ml_title']!r} -> {row['title']!r} ({row['year']})")
    print(f"Done in {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
