"""Load, filter, and split MovieLens ml-latest-small (spec section 5)."""
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

import config

ML_FILES = ("movies.csv", "ratings.csv", "tags.csv", "links.csv")

# "(1995)" or a range such as "(1975-1979)" or "(2007-)" at the very end of a MovieLens title.
_YEAR_RE = re.compile(r"\((\d{4})(?:\s*[-–]\s*\d{0,4})?\)\s*$")
# Articles that MovieLens moves to the end of a title: "Matrix, The" -> "The Matrix".
_ARTICLES = ("The", "A", "An", "La", "Le", "Les", "L'", "Il", "El", "Los", "Las", "Das", "Der", "Die")


def download_movielens(raw_dir: Path = config.RAW_DIR, url: str = config.MOVIELENS_URL) -> Path:
    """Download and unzip ml-latest-small into data/raw/ unless it is already there."""
    target = raw_dir / "ml-latest-small"
    if all((target / name).exists() for name in ML_FILES):
        print(f"  MovieLens already present in {target}")
        return target
    print(f"  Downloading {url} ...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(raw_dir)
    print(f"  Extracted to {target}")
    return target


def load_movielens(ml_dir: Path = config.MOVIELENS_DIR) -> dict[str, pd.DataFrame]:
    """Read the movies, ratings, tags, and links CSVs. tmdbId becomes a nullable integer."""
    data = {name.removesuffix(".csv"): pd.read_csv(ml_dir / name) for name in ML_FILES}
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
    """One-pass subset filter (spec 5.2, D6): popular movies with a tmdbId, then active users."""
    per_movie = ratings.groupby("movieId").size()
    with_tmdb = set(links.loc[links["tmdbId"].notna(), "movieId"])
    keep_movies = set(per_movie[per_movie >= min_movie_ratings].index) & with_tmdb
    kept = ratings[ratings["movieId"].isin(keep_movies)]
    per_user = kept.groupby("userId").size()
    keep_users = per_user[per_user >= min_user_ratings].index
    return kept[kept["userId"].isin(keep_users)].reset_index(drop=True)


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
