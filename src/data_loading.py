"""Load, sample, filter, and split MovieLens (spec section 5, adapted to MovieLens 32M: decision D37)."""
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import config

ML_FILES = ("movies.csv", "ratings.csv", "tags.csv", "links.csv")
# Small integer types keep the 32M ratings table around 600 MB in memory.
RATING_DTYPES = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}

# "(1995)" or a range such as "(1975-1979)" or "(2007-)" at the very end of a MovieLens title.
_YEAR_RE = re.compile(r"\((\d{4})(?:\s*[-–]\s*\d{0,4})?\)\s*$")
# Articles that MovieLens moves to the end of a title: "Matrix, The" -> "The Matrix".
_ARTICLES = ("The", "A", "An", "La", "Le", "Les", "L'", "Il", "El", "Los", "Las", "Das", "Der", "Die")


def download_movielens(raw_dir: Path = config.RAW_DIR, url: str = config.MOVIELENS_URL) -> Path:
    """Download and unzip the MovieLens zip into data/raw/ unless it is already there."""
    target = raw_dir / Path(url).stem
    if all((target / name).exists() for name in ML_FILES):
        print(f"  MovieLens already present in {target}")
        return target
    zip_path = raw_dir / Path(url).name
    print(f"  Downloading {url} ...")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        with zip_path.open("wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc="Download") as bar:
            for chunk in response.iter_content(1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    print("  Unzipping ...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir)
    zip_path.unlink()  # the zip is no longer needed once extracted
    print(f"  Extracted to {target}")
    return target


def load_movielens(ml_dir: Path = config.MOVIELENS_DIR) -> dict[str, pd.DataFrame]:
    """Read the movies, ratings, tags, and links CSVs. tmdbId becomes a nullable integer."""
    data = {}
    for name in ML_FILES:
        key = name.removesuffix(".csv")
        dtypes = RATING_DTYPES if key == "ratings" else None
        data[key] = pd.read_csv(ml_dir / name, dtype=dtypes)
    data["links"]["tmdbId"] = data["links"]["tmdbId"].astype("Int64")
    return data


def parse_title(raw: str) -> tuple[str, int | None]:
    """Clean a MovieLens title (D7): 'Matrix, The (1999)' -> ('The Matrix', 1999)."""
    text = str(raw).strip()
    year = None
    match = _YEAR_RE.search(text)
    if match:
        year = int(match.group(1))
        text = text[: match.start()].strip()
        # Alternate or original-language titles sit in brackets before the year: "Seven (a.k.a. Se7en)".
        while text.endswith(")") and text.rfind("(") > 0:
            text = text[: text.rfind("(")].strip()
    for article in _ARTICLES:
        suffix = ", " + article
        if text.endswith(suffix):
            joiner = "" if article.endswith("'") else " "
            text = article + joiner + text[: -len(suffix)]
            break
    return text, year


def clean_genres(genres: str) -> str:
    """Drop non-genres (D8): 'Adventure|Animation|IMAX' -> 'Adventure|Animation'."""
    kept = [g for g in str(genres).split("|") if g and g not in config.DROP_GENRES]
    return "|".join(kept)


def filter_subset(
    ratings: pd.DataFrame, links: pd.DataFrame, min_movie_ratings: int, min_user_ratings: int
) -> pd.DataFrame:
    """One-pass rating-count filter (spec 5.2, D6): popular movies with a tmdbId, then active users."""
    per_movie = ratings.groupby("movieId").size()
    with_tmdb = set(links.loc[links["tmdbId"].notna(), "movieId"])
    keep_movies = set(per_movie[per_movie >= min_movie_ratings].index) & with_tmdb
    kept = ratings[ratings["movieId"].isin(keep_movies)]
    per_user = kept.groupby("userId").size()
    keep_users = per_user[per_user >= min_user_ratings].index
    return kept[kept["userId"].isin(keep_users)].reset_index(drop=True)


def sample_subset(
    ratings: pd.DataFrame,
    links: pd.DataFrame,
    max_movies: int | None,
    n_users: int | None,
    min_movie_ratings: int,
    min_user_ratings: int,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Shrink a big MovieLens table to demo size (D37). Returns (ratings, funnel counts for the report).

    1. keep the `max_movies` most-rated movies that have a tmdbId (ties broken by movieId);
    2. among users with >= min_user_ratings ratings on those movies, draw `n_users` at random (fixed seed);
    3. apply the usual filters inside that sample: movies >= min_movie_ratings, then users >= min_user_ratings.
    None for max_movies or n_users means "no limit".
    """
    funnel = {}
    with_tmdb = set(links.loc[links["tmdbId"].notna(), "movieId"])
    counts = ratings.groupby("movieId").size().rename("n").reset_index()
    counts = counts[counts["movieId"].isin(with_tmdb)].sort_values(["n", "movieId"], ascending=[False, True])
    funnel["movies_with_tmdbId_and_ratings"] = len(counts)
    top = counts if max_movies is None else counts.head(max_movies)
    funnel["top_movies_kept"] = len(top)
    funnel["rating_count_of_last_top_movie"] = int(top["n"].min())
    on_top = ratings[ratings["movieId"].isin(set(top["movieId"]))]
    funnel["ratings_on_top_movies"] = len(on_top)

    per_user = on_top.groupby("userId").size()
    eligible = np.sort(per_user[per_user >= min_user_ratings].index.to_numpy())
    funnel["users_total"] = int(ratings["userId"].nunique())
    funnel["users_eligible"] = len(eligible)
    if n_users is not None and n_users < len(eligible):
        chosen = np.random.default_rng(seed).choice(eligible, size=n_users, replace=False)
    else:
        chosen = eligible
    funnel["users_sampled"] = len(chosen)
    sampled = on_top[on_top["userId"].isin(set(chosen.tolist()))]
    funnel["ratings_in_sample"] = len(sampled)

    final = filter_subset(sampled, links, min_movie_ratings, min_user_ratings)
    funnel["movies_after_filters"] = int(final["movieId"].nunique())
    funnel["users_after_filters"] = int(final["userId"].nunique())
    funnel["ratings_after_filters"] = len(final)
    return final, funnel


def build_movies_base(movies: pd.DataFrame, links: pd.DataFrame, movie_ids) -> pd.DataFrame:
    """movies_base table: movieId, title (clean), year, genres (clean), tmdbId, ml_title (original)."""
    base = movies[movies["movieId"].isin(set(movie_ids))].merge(links[["movieId", "tmdbId"]], on="movieId")
    parsed = base["title"].map(parse_title)
    out = pd.DataFrame(
        {
            "movieId": base["movieId"],
            "title": parsed.map(lambda p: p[0]),
            "year": parsed.map(lambda p: p[1]).astype("Int64"),
            "genres": base["genres"].map(clean_genres),
            "tmdbId": base["tmdbId"].astype("Int64"),
            "ml_title": base["title"],
        }
    )
    return out.sort_values("movieId").reset_index(drop=True)


def split_by_time(
    ratings: pd.DataFrame, test_fraction: float = config.TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user temporal split (spec 5.3, D9): the latest max(1, round(f * n)) ratings go to test."""
    ordered = ratings.sort_values(["userId", "timestamp", "movieId"], kind="mergesort")
    position = ordered.groupby("userId").cumcount()
    n = ordered.groupby("userId")["movieId"].transform("size")
    n_test = (n * test_fraction).round().clip(lower=1).astype(int)
    is_test = position >= (n - n_test)
    return ordered[~is_test].reset_index(drop=True), ordered[is_test].reset_index(drop=True)


def top_tags(tags: pd.DataFrame, movie_ids, n: int = 5) -> dict[int, list[str]]:
    """Up to n most frequent user tags per movie (D10): lowercased, trimmed, ties alphabetical."""
    t = tags[tags["movieId"].isin(set(movie_ids))].copy()
    t["tag"] = t["tag"].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    t = t[t["tag"] != ""]
    counts = t.groupby(["movieId", "tag"]).size().reset_index(name="count")
    counts = counts.sort_values(["movieId", "count", "tag"], ascending=[True, False, True])
    return {int(mid): list(group["tag"].head(n)) for mid, group in counts.groupby("movieId")}


def rating_stats(ratings: pd.DataFrame) -> dict:
    """Size and sparsity numbers for a ratings table (used for the report)."""
    users = int(ratings["userId"].nunique())
    movies = int(ratings["movieId"].nunique())
    n = len(ratings)
    return {
        "users": users,
        "movies": movies,
        "ratings": n,
        "density": round(n / (users * movies), 4),
        "mean_ratings_per_user": round(n / users, 1),
        "mean_ratings_per_movie": round(n / movies, 1),
        "mean_rating": round(float(ratings["rating"].mean()), 3),
        "share_ratings_ge_4": round(float((ratings["rating"] >= 4.0).mean()), 3),
    }
