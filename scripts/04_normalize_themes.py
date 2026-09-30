"""Phase 2, step 2: normalize the raw LLM themes (spec 6.6).

Usage:
    python scripts/04_normalize_themes.py
    python scripts/04_normalize_themes.py --threshold 0.75 --min-movies 3
    python scripts/04_normalize_themes.py --no-embeddings     # exact seed matching only
Outputs: data/processed/movie_themes.csv (movieId, theme), data/processed/theme_mapping.csv,
         results/theme_stats.json, results/theme_frequency.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config and src importable

import config  # noqa: E402
from src import theme_normalization as tn  # noqa: E402

STATS_PATH = config.RESULTS_DIR / "theme_stats.json"
CHART_PATH = config.RESULTS_DIR / "theme_frequency.png"
HEALTHY_RANGE = (50, 150)  # spec 6.6: a healthy result has roughly 50-150 distinct themes


def save_chart(final: pd.DataFrame, path: Path, top_n: int = 30) -> None:
    """Horizontal bar chart of the most common themes (blue = seed theme, orange = new theme)."""
    counts = final.groupby("theme")["movieId"].nunique().sort_values(ascending=False).head(top_n)[::-1]
    colors = ["#4C72B0" if t in config.SEED_THEMES else "#DD8452" for t in counts.index]
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel("Number of movies")
    ax.set_title(f"Top {len(counts)} themes: {final['theme'].nunique()} distinct themes over "
                 f"{final['movieId'].nunique():,} movies\n(blue = seed theme, orange = new theme)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw LLM themes into a shared vocabulary.")
    parser.add_argument("--threshold", type=float, default=config.THEME_SIM_THRESHOLD,
                        help="cosine similarity needed to map a new theme onto a seed (default %(default)s)")
    parser.add_argument("--min-movies", type=int, default=config.MIN_MOVIES_PER_THEME,
                        help="drop themes used by fewer movies than this (default %(default)s)")
    parser.add_argument("--no-embeddings", action="store_true", help="skip the embedding step")
    args = parser.parse_args()

    if not config.MOVIE_THEMES_RAW_PATH.exists():
        raise SystemExit("movie_themes_raw.csv not found. Run scripts/03_extract_themes.py first.")
    raw = pd.read_csv(config.MOVIE_THEMES_RAW_PATH)
    print(f"[1/4] Loaded {len(raw):,} (movie, theme) pairs: {raw['movieId'].nunique():,} movies, "
          f"{raw['theme'].nunique():,} distinct raw themes")

    embedder = None if args.no_embeddings else tn.load_embedder()
    if embedder is None:
        print("[2/4] Mapping to seed themes: exact matches only (embeddings off or sentence-transformers missing)")
    else:
        print(f"[2/4] Mapping to seed themes: exact matches, then {config.EMBEDDING_MODEL} "
              f"cosine similarity >= {args.threshold}")
    final, mapping, stats = tn.normalize(raw, config.SEED_THEMES, embedder, args.threshold, args.min_movies)
    print(f"  {stats['raw_themes_exact_seed']} raw themes match a seed exactly, "
          f"{stats['raw_themes_mapped_by_embedding']} mapped by embedding, "
          f"{stats['raw_themes_kept_as_new']} kept as new themes")
    examples = mapping[mapping["method"] == "embedding"].sort_values("similarity", ascending=False)
    for _, row in examples.head(8).iterrows():
        print(f"    {row['raw_theme']!r} -> {row['canonical_theme']!r} (similarity {row['similarity']:.2f})")

    print(f"[3/4] Dropping themes used by fewer than {args.min_movies} movies")
    rescued = stats["movies_rescued"]
    print(f"  {stats['distinct_after_mapping']} themes after mapping, {stats['themes_dropped_as_rare']} dropped as rare; "
          f"{rescued} movie{'' if rescued == 1 else 's'} kept its most common theme so it isn't left with none")

    print("[4/4] Writing files")
    final.to_csv(config.MOVIE_THEMES_PATH, index=False)
    mapping.to_csv(config.THEME_MAPPING_PATH, index=False)
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")
    save_chart(final, CHART_PATH)
    for path in (config.MOVIE_THEMES_PATH, config.THEME_MAPPING_PATH, STATS_PATH, CHART_PATH):
        print(f"  wrote {path.relative_to(config.BASE_DIR)}")

    print(f"\nTop 20 themes (movies):")
    for rank, (theme, n) in enumerate(stats["top20"], start=1):
        marker = "" if theme in config.SEED_THEMES else "  (new)"
        print(f"  {rank:>2}. {theme:<26} {n:>4}{marker}")
    low, high = HEALTHY_RANGE
    verdict = "within" if low <= stats["distinct_final_themes"] <= high else "outside"
    print(f"\nDistinct themes: {stats['distinct_final_themes']} ({verdict} the healthy range of {low}-{high}); "
          f"{stats['movies_with_themes']:,} movies, {stats['themes_per_movie']['mean']} themes per movie on average")


if __name__ == "__main__":
    main()
