"""Theme-based recommender over the Neo4j graph (spec section 8, D20-D22, D24).

score = alpha * graph_score + (1 - alpha) * popularity
graph_score = (theme_norm + GENRE_WEIGHT * genre_norm) / (1 + GENRE_WEIGHT)
All three parts are min-max scaled over the candidate pool (1.0 when every value is equal).

Command line:
    python -m src.recommender --user 877
    python -m src.recommender --user 877 --explain template --k 5 --alpha 1.0
    python -m src.recommender --ratings "1:5,260:5,1196:4,2571:5,4993:4"
"""
import argparse
import math
import random
import time
from contextlib import contextmanager
from functools import lru_cache

import config
from src import graph_builder

POOL_QUERY = """
// Step 1: the user's theme affinity from liked movies
MATCH (u:User {userId: $uid})-[r:RATED]->(:Movie)-[:HAS_THEME]->(t:Theme)
WHERE r.rating >= $like_threshold
WITH u, t, sum(r.rating - 3.0) AS affinity
// Step 2: unseen candidates sharing those themes
MATCH (t)<-[:HAS_THEME]-(cand:Movie)
WHERE NOT (u)-[:RATED]->(cand) AND cand.ratingCount >= $min_candidate_ratings
WITH cand, sum(affinity * t.idf) AS themeScore,
     collect({theme: t.name, affinity: affinity, idf: t.idf, contribution: affinity * t.idf}) AS matchedThemes
RETURN cand.movieId AS movieId, cand.title AS title, cand.year AS year, cand.posterPath AS posterPath,
       cand.ratingCount AS ratingCount, themeScore, matchedThemes
ORDER BY themeScore DESC, movieId
LIMIT $pool_size
"""

GENRE_AFFINITY_QUERY = """
MATCH (:User {userId: $uid})-[r:RATED]->(:Movie)-[:HAS_GENRE]->(g:Genre)
WHERE r.rating >= $like_threshold
RETURN g.name AS genre, sum(r.rating - 3.0) AS affinity
"""

THEME_PROFILE_QUERY = """
MATCH (:User {userId: $uid})-[r:RATED]->(:Movie)-[:HAS_THEME]->(t:Theme)
WHERE r.rating >= $like_threshold
RETURN t.name AS theme, sum(r.rating - 3.0) AS affinity, t.idf AS idf, count(*) AS likedMovies
ORDER BY affinity DESC, theme
"""

CANDIDATE_GENRES_QUERY = """
UNWIND $ids AS id
MATCH (m:Movie {movieId: id})-[:HAS_GENRE]->(g:Genre)
RETURN m.movieId AS movieId, collect(g.name) AS genres
"""

POPULAR_QUERY = """
MATCH (m:Movie)
WHERE m.ratingCount >= $min_candidate_ratings AND NOT m.movieId IN $exclude
  AND NOT EXISTS { MATCH (:User {userId: $uid})-[:RATED]->(m) }
RETURN m.movieId AS movieId, m.title AS title, m.year AS year, m.posterPath AS posterPath,
       m.ratingCount AS ratingCount
ORDER BY m.ratingCount DESC, m.movieId
LIMIT $n
"""


# ---------------------------------------------------------------------------
# Shared driver and small helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_driver():
    """One shared Neo4j driver for the whole process (spec section 15)."""
    return graph_builder.get_driver()


def _read(query: str, **params) -> list[dict]:
    with get_driver().session() as session:
        return session.run(query, **params).data()


@lru_cache(maxsize=1)
def duplicate_titles() -> frozenset[str]:
    """Titles shared by two or more graph movies (remakes); these get a year in display text."""
    rows = _read("MATCH (m:Movie) WITH m.title AS t, count(*) AS n WHERE n > 1 RETURN t")
    return frozenset(r["t"] for r in rows)


def display_title(title: str, year) -> str:
    """'Title (Year)' only when the title alone is ambiguous (Phase 1 follow-up)."""
    if title in duplicate_titles() and year is not None:
        return f"{title} ({int(year)})"
    return title


def graph_mode() -> str | None:
    rows = _read("MATCH (m:Meta) RETURN m.mode AS mode LIMIT 1")
    return rows[0]["mode"] if rows else None


def user_exists(user_id: int) -> bool:
    return bool(_read("MATCH (u:User {userId: $uid}) RETURN 1 AS x LIMIT 1", uid=user_id))


def example_user_ids(n: int = 5) -> list[int]:
    return [r["id"] for r in _read("MATCH (u:User) WHERE u.userId > 0 RETURN u.userId AS id ORDER BY id LIMIT $n", n=n)]


def _minmax(values: list[float]) -> list[float]:
    """Scale to [0, 1]; 1.0 for every item when all values are equal (D21)."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [1.0] * len(values)
    return [(v - low) / (high - low) for v in values]


# ---------------------------------------------------------------------------
# Profiles (also used by the dashboard)
# ---------------------------------------------------------------------------
def theme_profile(user_id: int) -> list[dict]:
    """The user's themes by affinity (sum of rating - 3 over liked movies)."""
    return _read(THEME_PROFILE_QUERY, uid=user_id, like_threshold=config.LIKE_THRESHOLD)


def genre_profile(user_id: int) -> dict[str, float]:
    rows = _read(GENRE_AFFINITY_QUERY, uid=user_id, like_threshold=config.LIKE_THRESHOLD)
    return {r["genre"]: r["affinity"] for r in sorted(rows, key=lambda r: (-r["affinity"], r["genre"]))}


# ---------------------------------------------------------------------------
# Candidate pool and ranking
# ---------------------------------------------------------------------------
def fetch_pool(user_id: int, pool_size: int = config.POOL_SIZE) -> dict:
    """Everything needed to rank for one user: theme pool, candidate genres, and the user's genre affinity.

    The evaluation fetches this once per user and calls rerank() for every alpha.
    """
    pool = _read(POOL_QUERY, uid=user_id, like_threshold=config.LIKE_THRESHOLD,
                 min_candidate_ratings=config.MIN_CANDIDATE_RATINGS, pool_size=pool_size)
    genres = {r["movieId"]: r["genres"] for r in _read(CANDIDATE_GENRES_QUERY, ids=[c["movieId"] for c in pool])}
    for cand in pool:
        cand["genres"] = sorted(genres.get(cand["movieId"], []))
        cand["matchedThemes"] = sorted(cand["matchedThemes"], key=lambda t: (-t["contribution"], t["theme"]))
    return {"userId": user_id, "pool": pool, "genre_affinity": genre_profile(user_id)}


def popular_fill(user_id: int, n: int, exclude: list[int]) -> list[dict]:
    """Most-rated unseen movies, used when the theme pool has fewer than k movies (D22)."""
    if n <= 0:
        return []
    rows = _read(POPULAR_QUERY, uid=user_id, exclude=exclude, n=n,
                 min_candidate_ratings=config.MIN_CANDIDATE_RATINGS)
    return [{**r, "score": None, "themeScore": 0.0, "genreScore": 0.0, "graphScore": 0.0, "popularity": None,
             "matchedThemes": [], "genres": [], "filled": True} for r in rows]


def rerank(pool_data: dict, k: int = 10, alpha: float = config.ALPHA, fill: bool = True) -> list[dict]:
    """Score the pool (D20, D21) and return the top-k, padded with popular movies if needed (D22)."""
    pool, genre_aff = pool_data["pool"], pool_data["genre_affinity"]
    theme_raw = [c["themeScore"] for c in pool]
    genre_raw = [sum(genre_aff.get(g, 0.0) for g in c["genres"]) for c in pool]
    pop_raw = [math.log1p(c["ratingCount"]) for c in pool]
    ranked = []
    for c, tn, gr, gn, pn in zip(pool, _minmax(theme_raw), genre_raw, _minmax(genre_raw), _minmax(pop_raw)):
        graph = (tn + config.GENRE_WEIGHT * gn) / (1 + config.GENRE_WEIGHT)
        ranked.append({**c, "genreScore": gr, "themeNorm": tn, "genreNorm": gn, "graphScore": graph,
                       "popularity": pn, "score": alpha * graph + (1 - alpha) * pn, "filled": False})
    ranked.sort(key=lambda c: (-c["score"], c["movieId"]))
    top = ranked[:k]
    if fill and len(top) < k:
        top += popular_fill(pool_data["userId"], k - len(top), [c["movieId"] for c in top])
    for c in top:
        c["displayTitle"] = display_title(c["title"], c["year"])
    return top


def recommend(user_id: int, k: int = 10, alpha: float = config.ALPHA) -> list[dict]:
    """Top-k recommendations for a user in the graph.

    Each item has movieId, title, displayTitle, year, posterPath, score, matchedThemes, plus the score parts
    (themeScore, genreScore, graphScore, popularity) and `filled` (True for popularity fill-ins).
    """
    return rerank(fetch_pool(user_id), k=k, alpha=alpha)


# ---------------------------------------------------------------------------
# Cold-start users (D24)
# ---------------------------------------------------------------------------
def cleanup_temporary_users() -> int:
    """Delete leftover temporary users (negative ids), e.g. after a crash. Returns how many were removed."""
    with get_driver().session() as session:
        return session.run("MATCH (u:User) WHERE u.userId < 0 DETACH DELETE u RETURN count(*) AS n").single()["n"]


@contextmanager
def temporary_user(ratings: dict[int, float]):
    """Create a throwaway User with a random negative id and the given ratings; always deleted afterwards."""
    if not ratings:
        raise ValueError("Give at least one rating.")
    uid = -random.randint(1, 2**31 - 1)
    rows = [{"movieId": int(m), "rating": float(r)} for m, r in ratings.items()]
    with get_driver().session() as session:
        found = session.run("""
            CREATE (u:User {userId: $uid})
            WITH u UNWIND $rows AS row
            MATCH (m:Movie {movieId: row.movieId})
            CREATE (u)-[:RATED {rating: row.rating, timestamp: timestamp() / 1000}]->(m)
            RETURN count(*) AS n""", uid=uid, rows=rows).single()["n"]
    try:
        if found < len(rows):
            print(f"  Note: {len(rows) - found} of the given movies are not in the graph and were ignored.")
        yield uid
    finally:
        with get_driver().session() as session:
            session.run("MATCH (u:User {userId: $uid}) DETACH DELETE u", uid=uid).consume()


def recommend_for_ratings(ratings: dict[int, float], k: int = 10, alpha: float = config.ALPHA) -> list[dict]:
    """Recommendations for someone who is not in MovieLens, from a {movieId: rating} dict."""
    with temporary_user(ratings) as uid:
        return recommend(uid, k=k, alpha=alpha)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _parse_ratings(text: str) -> dict[int, float]:
    pairs = [p.split(":") for p in text.split(",") if p.strip()]
    return {int(m): float(r) for m, r in pairs}


def _print_recs(recs: list[dict], explanations: list[dict] | None) -> None:
    for i, rec in enumerate(recs, start=1):
        if rec["filled"]:
            print(f"{i:>2}. {rec['displayTitle']} ({rec['year'] or '?'})  [popular fill-in, no theme path]")
        else:
            themes = ", ".join(t["theme"] for t in rec["matchedThemes"][:3])
            print(f"{i:>2}. {rec['displayTitle']} ({rec['year'] or '?'})  score {rec['score']:.3f}  "
                  f"(graph {rec['graphScore']:.3f}, popularity {rec['popularity']:.3f})  themes: {themes}")
        if explanations:
            print(f"    -> {explanations[i - 1]['text']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Theme-based movie recommendations from the knowledge graph.")
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--user", type=int, help="MovieLens userId in the graph")
    who.add_argument("--ratings", help='cold-start profile, e.g. "1:5,260:5,2571:4" (movieId:rating)')
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=config.ALPHA, help="1.0 = pure graph (default %(default)s)")
    parser.add_argument("--explain", choices=["llm", "template", "none"], default="llm")
    args = parser.parse_args()

    from src import explainer  # imported here to avoid a circular import

    removed = cleanup_temporary_users()
    if removed:
        print(f"Removed {removed} leftover temporary user(s).")
    mode = graph_mode()
    if mode != "full":
        print(f"Note: the graph is in '{mode}' mode (train ratings only).")
    start = time.time()
    if args.user is not None:
        if not user_exists(args.user):
            raise SystemExit(f"User {args.user} is not in the graph. Example ids: {example_user_ids()}")
        profile = theme_profile(args.user)[:5]
        print(f"User {args.user}, top themes: " + ", ".join(f"{p['theme']} ({p['affinity']:.1f})" for p in profile))
        recs = recommend(args.user, k=args.k, alpha=args.alpha)
        rec_seconds = time.time() - start
        exps = None if args.explain == "none" else explainer.explain_many(args.user, recs, mode=args.explain)
    else:
        ratings = _parse_ratings(args.ratings)
        with temporary_user(ratings) as uid:
            profile = theme_profile(uid)[:5]
            print("Custom profile, top themes: " + ", ".join(f"{p['theme']} ({p['affinity']:.1f})" for p in profile))
            recs = recommend(uid, k=args.k, alpha=args.alpha)
            rec_seconds = time.time() - start
            exps = None if args.explain == "none" else explainer.explain_many(uid, recs, mode=args.explain,
                                                                               custom=True)
    print(f"\nTop {len(recs)} (alpha {args.alpha}, explanations: {args.explain}):")
    _print_recs(recs, exps)
    fills = sum(r["filled"] for r in recs)
    print(f"\nRecommendation time {rec_seconds:.2f} s; total {time.time() - start:.2f} s; "
          f"{fills} popularity fill-in(s)")
    if exps:
        stats = explainer.last_batch_stats()
        print(f"Explanations: {stats['llm_calls']} new LLM call(s), {stats['cache_hits']} from cache, "
              f"{stats['fallbacks']} template fallback(s)")
    left = _read("MATCH (u:User) WHERE u.userId < 0 RETURN count(u) AS n")[0]["n"]
    print(f"Temporary users left in the graph: {left}")


if __name__ == "__main__":
    main()
