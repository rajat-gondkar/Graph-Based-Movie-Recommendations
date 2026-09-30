"""Theme normalization (spec 6.6): merge near-duplicates into seed themes and drop rare themes."""
import re

import numpy as np
import pandas as pd

import config

MAPPING_COLUMNS = ["raw_theme", "canonical_theme", "method", "closest_seed", "similarity"]


def basic_clean(theme) -> str:
    """Step 1: lowercase, trim, collapse whitespace."""
    return re.sub(r"\s+", " ", str(theme).lower()).strip()


def _key(theme: str) -> str:
    """Matching key that treats hyphens like spaces."""
    return re.sub(r"[-\s]+", " ", theme).strip()


def load_embedder():
    """The sentence-transformers model on CPU, or None if the package isn't installed (step 5 fallback)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    return SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")


def map_to_seeds(themes, seeds: list[str], embedder, threshold: float) -> pd.DataFrame:
    """Steps 2-3: one row per distinct raw theme with its canonical theme, the method, and the best seed match."""
    seed_by_key = {_key(s): s for s in seeds}
    rows, others = [], []
    for theme in sorted(set(themes)):
        seed = seed_by_key.get(_key(theme))
        if seed:
            rows.append({"raw_theme": theme, "canonical_theme": seed, "method": "exact_seed",
                         "closest_seed": seed, "similarity": 1.0})
        else:
            others.append(theme)
    if others and embedder is not None:
        a = embedder.encode(others, normalize_embeddings=True, show_progress_bar=False)
        b = embedder.encode(list(seeds), normalize_embeddings=True, show_progress_bar=False)
        sims = a @ b.T  # cosine similarity, since all vectors have length 1
        best = sims.argmax(axis=1)
        for theme, j, sim in zip(others, best, sims[np.arange(len(others)), best]):
            mapped = bool(sim >= threshold)
            rows.append({"raw_theme": theme, "canonical_theme": seeds[j] if mapped else theme,
                         "method": "embedding" if mapped else "kept_new",
                         "closest_seed": seeds[j], "similarity": round(float(sim), 3)})
    else:
        rows += [{"raw_theme": t, "canonical_theme": t, "method": "kept_new",
                  "closest_seed": None, "similarity": None} for t in others]
    return pd.DataFrame(rows, columns=MAPPING_COLUMNS)


def drop_rare(pairs: pd.DataFrame, min_movies: int) -> tuple[pd.DataFrame, int]:
    """Step 4: drop themes used by fewer than min_movies movies.

    A movie that would be left with no theme keeps its most common one. Returns (pairs, movies rescued).
    """
    counts = pairs.groupby("theme")["movieId"].nunique()
    kept = pairs[pairs["theme"].map(counts) >= min_movies]
    orphans = pairs[~pairs["movieId"].isin(kept["movieId"])]
    rescued = (
        orphans.assign(n=orphans["theme"].map(counts))
        .sort_values(["movieId", "n", "theme"], ascending=[True, False, True])
        .drop_duplicates("movieId")[["movieId", "theme"]]
    )
    result = pd.concat([kept, rescued]).sort_values(["movieId", "theme"]).reset_index(drop=True)
    return result, len(rescued)


def normalize(raw_pairs: pd.DataFrame, seeds: list[str], embedder, threshold: float,
              min_movies: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Spec 6.6 steps 1-5. Returns (movie_themes, mapping table, stats)."""
    pairs = raw_pairs.assign(theme=raw_pairs["theme"].map(basic_clean))
    pairs = pairs[pairs["theme"] != ""]
    mapping = map_to_seeds(pairs["theme"], seeds, embedder, threshold)
    canonical = dict(zip(mapping["raw_theme"], mapping["canonical_theme"]))
    mapped = pairs.assign(theme=pairs["theme"].map(canonical)).drop_duplicates(["movieId", "theme"])
    final, rescued = drop_rare(mapped, min_movies)

    after_mapping = mapped.groupby("theme")["movieId"].nunique()
    after_drop = final.groupby("theme")["movieId"].nunique().sort_values(ascending=False)
    mapping["movies_after_mapping"] = mapping["canonical_theme"].map(after_mapping)
    mapping["kept"] = mapping["canonical_theme"].isin(after_drop.index)
    per_movie = final.groupby("movieId").size()
    method_counts = mapping["method"].value_counts()
    stats = {
        "used_embeddings": embedder is not None,
        "embedding_model": config.EMBEDDING_MODEL if embedder is not None else None,
        "similarity_threshold": threshold,
        "min_movies_per_theme": min_movies,
        "movies": int(raw_pairs["movieId"].nunique()),
        "raw_pairs": len(raw_pairs),
        "distinct_raw_themes": len(mapping),
        "raw_themes_exact_seed": int(method_counts.get("exact_seed", 0)),
        "raw_themes_mapped_by_embedding": int(method_counts.get("embedding", 0)),
        "raw_themes_kept_as_new": int(method_counts.get("kept_new", 0)),
        "distinct_after_mapping": int(after_mapping.size),
        "themes_dropped_as_rare": int((~after_mapping.index.isin(after_drop.index)).sum()),
        "movies_rescued": rescued,
        "distinct_final_themes": int(after_drop.size),
        "final_pairs": len(final),
        "movies_with_themes": int(final["movieId"].nunique()),
        "themes_per_movie": {"mean": round(float(per_movie.mean()), 2), "min": int(per_movie.min()),
                             "max": int(per_movie.max())} if len(per_movie) else None,
        "seed_themes_used": int(after_drop.index.isin(seeds).sum()),
        "seed_themes_unused": sorted(set(seeds) - set(after_drop.index)),
        "new_themes_kept": sorted(set(after_drop.index) - set(seeds)),
        "top20": [[theme, int(n)] for theme, n in after_drop.head(20).items()],
    }
    return final, mapping, stats
