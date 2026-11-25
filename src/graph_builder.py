"""Build the Neo4j knowledge graph (spec section 7, D18, D19).

(User)-[:RATED {rating, timestamp}]->(Movie)-[:HAS_THEME]->(Theme), (Movie)-[:HAS_GENRE]->(Genre),
plus one (:Meta {mode, builtAt}) node that records whether the graph holds train ratings only (eval) or all (full).
"""
import math

import pandas as pd
from neo4j import GraphDatabase

import config

BATCH_SIZE = 1000
MODES = ("eval", "full")

CONSTRAINTS = [
    "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE",
    "CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE",
    "CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
    "CREATE CONSTRAINT theme_name IF NOT EXISTS FOR (t:Theme) REQUIRE t.name IS UNIQUE",
]


def get_driver():
    """One shared driver; callers close it when done."""
    s = config.get_neo4j_settings()
    # Notifications off: probing an empty database for the Meta label would otherwise print warnings.
    return GraphDatabase.driver(s["uri"], auth=(s["user"], s["password"]), notifications_min_severity="OFF")


def _none_if_nan(value):
    return None if pd.isna(value) else value


def load_inputs(mode: str) -> dict[str, pd.DataFrame]:
    """Read the processed CSVs and keep only what belongs in the graph (D18).

    Movies need an overview (they are in movies_enriched.csv) and at least one theme. RATED edges come from
    train.csv only in eval mode, otherwise from train + test, and only point at graph movies.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    movies = pd.read_csv(config.MOVIES_ENRICHED_PATH)
    themes = pd.read_csv(config.MOVIE_THEMES_PATH)
    movies = movies[movies["movieId"].isin(themes["movieId"])]
    themes = themes[themes["movieId"].isin(movies["movieId"])].drop_duplicates()

    files = [config.TRAIN_PATH] if mode == "eval" else [config.TRAIN_PATH, config.TEST_PATH]
    ratings = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ratings = ratings[ratings["movieId"].isin(movies["movieId"])].drop_duplicates(["userId", "movieId"])

    genres = movies[["movieId", "genres"]].dropna()
    genres = genres.assign(genre=genres["genres"].str.split("|")).explode("genre")[["movieId", "genre"]]
    genres = genres[~genres["genre"].isin(config.DROP_GENRES) & (genres["genre"] != "")].drop_duplicates()
    return {"movies": movies, "themes": themes, "genres": genres, "ratings": ratings}


def _batches(rows: list[dict], size: int = BATCH_SIZE):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def _run_batches(session, query: str, rows: list[dict]) -> None:
    for batch in _batches(rows):
        session.execute_write(lambda tx, b=batch: tx.run(query, rows=b).consume())


def existing_mode(driver) -> str | None:
    """The mode recorded in the Meta node, or None for an empty database."""
    with driver.session() as session:
        record = session.run("MATCH (m:Meta) RETURN m.mode AS mode LIMIT 1").single()
    return record["mode"] if record else None


def wipe(driver) -> None:
    """Delete everything: relationships first, then nodes, in batches so no transaction gets too big for the
    1 GB heap (one batch of 10,000 nodes with their ~600k relationships ran out of memory in dataset v3)."""
    with driver.session() as session:
        session.run("MATCH ()-[r]->() CALL (r) { DELETE r } IN TRANSACTIONS OF 20000 ROWS").consume()
        session.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS").consume()


def create_constraints(driver) -> None:
    with driver.session() as session:
        for statement in CONSTRAINTS:
            session.run(statement).consume()


def load_graph(driver, data: dict[str, pd.DataFrame]) -> None:
    """Batched UNWIND ... MERGE loaders; safe to run twice (idempotent)."""
    movies = [
        {"movieId": int(r.movieId), "title": r.title, "year": _none_if_nan(r.year), "overview": r.overview,
         "posterPath": _none_if_nan(r.poster_path), "tmdbId": int(r.tmdbId)}
        for r in data["movies"].itertuples()
    ]
    for m in movies:
        m["year"] = None if m["year"] is None else int(m["year"])
    genre_rows = [{"movieId": int(r.movieId), "name": r.genre} for r in data["genres"].itertuples()]
    theme_rows = [{"movieId": int(r.movieId), "name": r.theme} for r in data["themes"].itertuples()]
    users = [{"userId": int(u)} for u in data["ratings"]["userId"].unique()]
    rating_rows = [{"userId": int(r.userId), "movieId": int(r.movieId), "rating": float(r.rating),
                    "timestamp": int(r.timestamp)} for r in data["ratings"].itertuples()]

    with driver.session() as session:
        _run_batches(session, """
            UNWIND $rows AS row MERGE (m:Movie {movieId: row.movieId})
            SET m.title = row.title, m.year = row.year, m.overview = row.overview,
                m.posterPath = row.posterPath, m.tmdbId = row.tmdbId""", movies)
        _run_batches(session, """
            UNWIND $rows AS row MATCH (m:Movie {movieId: row.movieId})
            MERGE (g:Genre {name: row.name}) MERGE (m)-[:HAS_GENRE]->(g)""", genre_rows)
        _run_batches(session, """
            UNWIND $rows AS row MATCH (m:Movie {movieId: row.movieId})
            MERGE (t:Theme {name: row.name}) MERGE (m)-[:HAS_THEME]->(t)""", theme_rows)
        _run_batches(session, "UNWIND $rows AS row MERGE (:User {userId: row.userId})", users)
        _run_batches(session, """
            UNWIND $rows AS row
            MATCH (u:User {userId: row.userId}) MATCH (m:Movie {movieId: row.movieId})
            MERGE (u)-[r:RATED]->(m) SET r.rating = row.rating, r.timestamp = row.timestamp""", rating_rows)


def compute_stats(driver) -> int:
    """Movie.ratingCount / avgRating and Theme.movieCount / idf = ln(totalMovies / movieCount). Returns totalMovies."""
    with driver.session() as session:
        session.run("""
            MATCH (m:Movie) OPTIONAL MATCH (:User)-[r:RATED]->(m)
            WITH m, count(r) AS n, avg(r.rating) AS avg
            SET m.ratingCount = n, m.avgRating = avg""").consume()
        total = session.run("MATCH (m:Movie) RETURN count(m) AS n").single()["n"]
        themes = session.run("""
            MATCH (t:Theme) OPTIONAL MATCH (t)<-[:HAS_THEME]-(m:Movie)
            RETURN t.name AS name, count(m) AS n""").data()
        rows = [{"name": t["name"], "movieCount": t["n"], "idf": math.log(total / t["n"])} for t in themes if t["n"]]
        session.run("""
            UNWIND $rows AS row MATCH (t:Theme {name: row.name})
            SET t.movieCount = row.movieCount, t.idf = row.idf""", rows=rows).consume()
    return total


def write_meta(driver, mode: str) -> None:
    with driver.session() as session:
        session.run("MATCH (m:Meta) DETACH DELETE m").consume()
        session.run("CREATE (:Meta {mode: $mode, builtAt: toString(datetime())})", mode=mode).consume()


def counts(driver) -> dict[str, dict[str, int]]:
    """Node counts by label and relationship counts by type."""
    with driver.session() as session:
        nodes = {r["label"]: r["n"] for r in session.run(
            "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY label")}
        rels = {r["type"]: r["n"] for r in session.run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY type")}
    return {"nodes": nodes, "relationships": rels}


def theme_idf_extremes(driver, n: int = 5) -> dict[str, list[tuple[str, int, float]]]:
    """Rarest and most common themes by IDF, for the report."""
    query = "MATCH (t:Theme) RETURN t.name AS name, t.movieCount AS n, t.idf AS idf ORDER BY t.idf {} , t.name LIMIT $n"
    with driver.session() as session:
        rare = [(r["name"], r["n"], r["idf"]) for r in session.run(query.format("DESC"), n=n)]
        common = [(r["name"], r["n"], r["idf"]) for r in session.run(query.format("ASC"), n=n)]
    return {"rarest": rare, "most_common": common}


def sanity_check(driver, data: dict[str, pd.DataFrame]) -> list[tuple[str, bool]]:
    """Compare graph counts with the input tables; returns (description, passed) pairs."""
    c = counts(driver)
    expected = [
        ("Movie nodes = movies with themes", c["nodes"].get("Movie") == len(data["movies"])),
        ("Theme nodes = distinct themes", c["nodes"].get("Theme") == data["themes"]["theme"].nunique()),
        ("Genre nodes = distinct genres", c["nodes"].get("Genre") == data["genres"]["genre"].nunique()),
        ("User nodes = distinct rating users", c["nodes"].get("User") == data["ratings"]["userId"].nunique()),
        ("HAS_THEME edges = movie-theme pairs", c["relationships"].get("HAS_THEME") == len(data["themes"])),
        ("HAS_GENRE edges = movie-genre pairs", c["relationships"].get("HAS_GENRE") == len(data["genres"])),
        ("RATED edges = loaded ratings", c["relationships"].get("RATED") == len(data["ratings"])),
    ]
    with driver.session() as session:
        no_theme = session.run("MATCH (m:Movie) WHERE NOT (m)-[:HAS_THEME]->() RETURN count(m) AS n").single()["n"]
        no_idf = session.run("MATCH (t:Theme) WHERE t.idf IS NULL RETURN count(t) AS n").single()["n"]
    return expected + [("Every movie has a theme", no_theme == 0), ("Every theme has an idf", no_idf == 0)]
