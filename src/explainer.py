"""Grounded explanations from the graph path (spec section 9, D23, D24).

1. evidence(): the liked movies -> shared theme -> recommended movie path, as a structured object.
2. The text is written either by the LLM (told to use only facts in the evidence) or by a fixed template.
3. LLM texts are cached in data/cache/explanations_cache.json, keyed by user, movie, and the evidence.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import config
from src import cache, llm_client
from src.recommender import _read, display_title, temporary_user

EVIDENCE_QUERY = """
// D23: count liked movies per theme before cutting to 3, and order themes by score contribution
MATCH (u:User {userId: $uid})-[r:RATED]->(seen:Movie)-[:HAS_THEME]->(t:Theme)<-[:HAS_THEME]-(rec:Movie {movieId: $mid})
WHERE r.rating >= $like_threshold AND seen <> rec
WITH t, seen, r
ORDER BY r.rating DESC, seen.ratingCount DESC, seen.movieId
WITH t, collect({title: seen.title, year: seen.year, rating: r.rating}) AS liked, sum(r.rating - 3.0) AS affinity
RETURN t.name AS theme, affinity * t.idf AS contribution, size(liked) AS likedCount, liked[..$max_liked] AS liked
ORDER BY contribution DESC, theme
LIMIT $max_themes
"""

SHARED_GENRES_QUERY = """
MATCH (u:User {userId: $uid})-[r:RATED]->(seen:Movie)-[:HAS_GENRE]->(g:Genre)<-[:HAS_GENRE]-(rec:Movie {movieId: $mid})
WHERE r.rating >= $like_threshold AND seen <> rec
RETURN g.name AS genre, count(DISTINCT seen) AS n
ORDER BY n DESC, genre
"""

MOVIE_QUERY = "MATCH (m:Movie {movieId: $mid}) RETURN m.title AS title, m.year AS year"

SIMILAR_QUERY = """
// D50: liked movies that the recommended movie is linked to by SIMILAR edges, strongest contribution first
MATCH (u:User {userId: $uid})-[r:RATED]->(liked:Movie)<-[s:SIMILAR]-(rec:Movie {movieId: $mid})
WHERE r.rating >= $like_threshold
RETURN liked.title AS title, liked.year AS year, r.rating AS rating, s.coLikes AS sharedFans
ORDER BY s.score * (r.rating - u.meanRating) DESC, s.coLikes DESC, title
LIMIT $max_liked
"""

SIMILAR_KEY = "liked_by_the_same_people"

# Spec 9.2 prompt, extended in D50 to describe the similarity evidence.
SYSTEM_PROMPT = """You write short, friendly explanations for movie recommendations.
You are given STRUCTURED EVIDENCE from a knowledge graph. Write 1 to 2 sentences explaining why the movie is recommended.
The evidence can contain shared themes (with the user's liked movies that have each theme) and "liked_by_the_same_people": movies the user liked that many of the same viewers also loved ("shared_fans" = how many viewers loved both).
Rules:
- Use ONLY the movie titles, themes, genres, and numbers present in the evidence. Do not add any other facts, plot details, actors, or awards.
- Mention at least one of the user's previously liked movies by title, and at least one shared theme if the evidence lists any.
- Plain, natural language. No bullet points, no markdown, no emojis."""

MODES = ("llm", "template")
_stats = {"llm_calls": 0, "cache_hits": 0, "fallbacks": 0}


# ---------------------------------------------------------------------------
# Evidence (the ground truth of every explanation)
# ---------------------------------------------------------------------------
def evidence(user_id: int, movie_id: int) -> dict:
    """The structured evidence object in the exact shape of spec 9.1."""
    params = {"uid": user_id, "mid": movie_id, "like_threshold": config.LIKE_THRESHOLD}
    movie = _read(MOVIE_QUERY, mid=movie_id)
    if not movie:
        raise ValueError(f"Movie {movie_id} is not in the graph.")
    rows = _read(EVIDENCE_QUERY, **params, max_themes=config.EVIDENCE_MAX_THEMES, max_liked=config.EVIDENCE_MAX_LIKED)
    year = movie[0]["year"]
    return {
        "recommended": {"title": display_title(movie[0]["title"], year), "year": int(year) if year else None},
        "evidence": [
            {"theme": row["theme"],
             "because_you_liked": [{"title": display_title(m["title"], m["year"]), "rating": m["rating"]}
                                   for m in row["liked"]]}
            for row in rows
        ],
        SIMILAR_KEY: [{"title": display_title(r["title"], r["year"]), "rating": r["rating"], "shared_fans": r["sharedFans"]}
                      for r in _read(SIMILAR_QUERY, **params, max_liked=config.EVIDENCE_MAX_LIKED)],
        "shared_genres": [r["genre"] for r in _read(SHARED_GENRES_QUERY, **params)],
    }


def has_path(ev: dict) -> bool:
    """True if the recommendation is linked to the user's liked movies by a theme or a SIMILAR path."""
    return bool(ev["evidence"] or ev.get(SIMILAR_KEY))


# ---------------------------------------------------------------------------
# Text: template and LLM
# ---------------------------------------------------------------------------
def _join_titles(titles: list[str]) -> str:
    titles = [f"*{t}*" for t in titles]
    return titles[0] if len(titles) == 1 else ", ".join(titles[:-1]) + " and " + titles[-1]


def template_text(ev: dict) -> str:
    """Spec 9.3 template (strongest theme), plus the strongest similarity path (D50); fill-ins have neither."""
    title = ev["recommended"]["title"]
    similar = ev.get(SIMILAR_KEY) or []
    if similar:
        fans = [m["title"] for m in similar[:2]]
        text = f"Recommended because many people who loved {_join_titles(fans)} also loved *{title}*"
        if ev["evidence"]:
            top = ev["evidence"][0]
            liked = [m["title"] for m in top["because_you_liked"][:2]]
            return text + f", and it shares the theme **{top['theme']}** with {_join_titles(liked)}."
        return text + "."
    if not ev["evidence"]:
        if ev["shared_genres"]:
            return (f"*{title}* is a popular movie in genres you like ({', '.join(ev['shared_genres'][:2])}); "
                    f"it doesn't share a theme with the movies you rated highly.")
        return f"*{title}* is one of the most popular movies you haven't rated yet."
    top = ev["evidence"][0]
    liked = [m["title"] for m in top["because_you_liked"][:2]]
    verb = "share" if len(liked) > 1 else "shares"
    return (f"Recommended because you liked {_join_titles(liked)}, which {verb} the theme "
            f"**{top['theme']}** with *{title}*.")


def _cache_key(user_key: str, movie_id: int, ev: dict) -> str:
    payload = json.dumps(ev, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(f"{user_key}|{movie_id}|{payload}".encode()).hexdigest()


def _messages(ev: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(ev, indent=2, ensure_ascii=False)}]


def _call_llm(ev: dict) -> str | Exception:
    """One LLM call; returns the text, or the exception so the caller can fall back to the template."""
    try:
        return " ".join(llm_client.chat(_messages(ev), max_tokens=120, temperature=0.3, purpose="explanation").split())
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - any failure (even a missing key) means "use the template"
        return exc


def _explain_batch(user_id: int, movie_ids: list[int], mode: str, user_key: str, workers: int = 1) -> list[dict]:
    """Evidence and text for several movies. Missing LLM texts are fetched in parallel (workers threads);
    the cache is read once and written once, from this thread only."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    store = cache.load_json(config.EXPLANATIONS_CACHE_PATH)
    results, todo = [], []
    for movie_id in movie_ids:
        ev = evidence(user_id, movie_id)
        if mode == "template" or not has_path(ev):  # no path at all: nothing for the LLM to ground on (D41)
            results.append({"text": template_text(ev), "evidence": ev, "mode": mode, "source": "template"})
            continue
        key = _cache_key(user_key, movie_id, ev)
        if key in store:
            _stats["cache_hits"] += 1
            results.append({"text": store[key]["text"], "evidence": ev, "mode": mode, "source": "cache"})
            continue
        results.append({"text": None, "evidence": ev, "mode": mode, "source": "llm"})
        todo.append((len(results) - 1, key, movie_id))
    if todo:
        _stats["llm_calls"] += len(todo)
        evs = [results[i]["evidence"] for i, _, _ in todo]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            replies = list(pool.map(_call_llm, evs))
        for (i, key, movie_id), reply in zip(todo, replies):
            if isinstance(reply, Exception) or not reply:
                print(f"  LLM explanation failed for movie {movie_id} ({reply}); using the template.")
                _stats["fallbacks"] += 1
                results[i].update(text=template_text(results[i]["evidence"]), source="template-fallback")
            else:
                results[i]["text"] = reply
                store[key] = {"text": reply, "userKey": user_key, "movieId": movie_id, "model": llm_client.model_name()}
        cache.save_json(store, config.EXPLANATIONS_CACHE_PATH)  # D13: atomic write
    return results


# ---------------------------------------------------------------------------
# Public functions (spec 9.5)
# ---------------------------------------------------------------------------
def explain(user_id: int, movie_id: int, mode: str = "llm") -> dict:
    """{"text", "evidence", "mode", "source"} for a user in the graph."""
    return _explain_batch(user_id, [movie_id], mode, str(user_id))[0]


def explain_for_ratings(ratings: dict[int, float], movie_id: int, mode: str = "llm") -> dict:
    """Same for a cold-start profile; the cache key uses a fixed 'custom' marker plus the evidence (D24)."""
    with temporary_user(ratings) as uid:
        return _explain_batch(uid, [movie_id], mode, "custom")[0]


def explain_many(user_id: int, recs: list[dict], mode: str = "llm", custom: bool = False,
                 workers: int = 5) -> list[dict]:
    """Explanations for a list of recommendations, with up to `workers` LLM calls at a time.

    Pass custom=True when user_id is a temporary user inside temporary_user(), so one temporary node serves all cards.
    """
    for key in _stats:
        _stats[key] = 0
    return _explain_batch(user_id, [r["movieId"] for r in recs], mode, "custom" if custom else str(user_id), workers)


def last_batch_stats() -> dict:
    """LLM calls, cache hits, and fallbacks during the last explain_many() call."""
    return dict(_stats)
