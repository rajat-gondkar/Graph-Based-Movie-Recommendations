"""Web app backend (Phase 5): a small JSON API over the recommender, plus the static page in app/static.

Run from the project folder:
    uvicorn app.server:app --port 8000
then open http://localhost:8000

Endpoints
    GET  /api/status                         graph mode, LLM availability, counts
    GET  /api/catalog                        every movie with genres and themes (for browsing)
    GET  /api/users                          MovieLens user ids
    POST /api/recommend   {liked, k, alpha}  recommendations for a set of liked movies (cold start)
    GET  /api/users/{id}/recommend?k&alpha   recommendations for a MovieLens user
Every recommendation comes with an LLM explanation (template if the LLM fails) and its evidence (the graph path).
"""
import sys
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config and src importable

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import config  # noqa: E402
from src import explainer, recommender  # noqa: E402
from src.tmdb_client import poster_url  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
LIKE_RATING = 5.0   # a Like counts as a 5-star rating (D42)
MIN_LIKES = 3
DB_ERRORS = (ServiceUnavailable, SessionExpired, Neo4jError, OSError)
_compute_lock = threading.Lock()  # one recommendation run at a time: keeps the explanation cache writes safe


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        removed = recommender.cleanup_temporary_users()  # leftovers from a crash (D24)
        if removed:
            print(f"Removed {removed} leftover temporary user(s).")
    except DB_ERRORS as exc:
        print(f"Neo4j is not reachable yet ({exc}); the page will show how to start it.")
    yield
    recommender.get_driver().close()


app = FastAPI(title="Movie KG Recommender", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)  # the catalogue is ~630 KB of JSON


@app.exception_handler(ServiceUnavailable)
@app.exception_handler(SessionExpired)
@app.exception_handler(OSError)
async def neo4j_down(_request: Request, _exc: Exception):
    return JSONResponse(status_code=503, content={
        "detail": "Can't reach the Neo4j database. Start it with `colima start` (macOS) and `docker compose up -d` "
                  "in the project folder, then reload the page."})


def llm_available() -> bool:
    try:
        config.get_llm_settings()
        return True
    except SystemExit:
        return False


# ---------------------------------------------------------------------------
# Shaping results for the page
# ---------------------------------------------------------------------------
def _year(value):
    return int(value) if value is not None else None


def _card(rec: dict, exp: dict) -> dict:
    return {
        "movieId": rec["movieId"], "title": rec["displayTitle"], "year": _year(rec["year"]),
        "poster": poster_url(rec["posterPath"]), "score": rec["score"], "graphScore": rec["graphScore"],
        "popularity": rec["popularity"], "filled": rec["filled"],
        "themes": [t["theme"] for t in rec["matchedThemes"][:config.EVIDENCE_MAX_THEMES]],
        "explanation": exp["text"], "source": exp["source"], "evidence": exp["evidence"],
    }


def _profile(uid: int) -> dict:
    themes = [{"name": t["theme"], "value": t["affinity"]} for t in recommender.theme_profile(uid)[:8]]
    genres = [{"name": g, "value": v} for g, v in list(recommender.genre_profile(uid).items())[:8]]
    return {"themes": themes, "genres": genres}


def _run(uid: int, k: int, alpha: float, custom: bool) -> dict:
    start = time.time()
    recs = recommender.recommend(uid, k=k, alpha=alpha)
    exps = explainer.explain_many(uid, recs, mode="llm", custom=custom)
    stats = explainer.last_batch_stats()
    return {"profile": _profile(uid), "recommendations": [_card(r, e) for r, e in zip(recs, exps)],
            "stats": {**stats, "seconds": round(time.time() - start, 2),
                      "filled": sum(r["filled"] for r in recs)}}


@lru_cache(maxsize=256)
def _profile_result(liked: tuple[int, ...], k: int, alpha: float) -> dict:
    with recommender.temporary_user({m: LIKE_RATING for m in liked}) as uid:
        return _run(uid, k, alpha, custom=True)


@lru_cache(maxsize=256)
def _user_result(user_id: int, k: int, alpha: float) -> dict:
    result = _run(user_id, k, alpha, custom=False)
    top = recommender.user_top_movies(user_id, n=8)
    result["topRated"] = [{"title": m["displayTitle"], "rating": m["rating"], "poster": poster_url(m["posterPath"])}
                          for m in top]
    return result


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/status")
def status():
    try:
        mode = recommender.graph_mode()
    except DB_ERRORS:
        raise HTTPException(503, "Can't reach the Neo4j database. Start it with `colima start` (macOS) and "
                                 "`docker compose up -d` in the project folder, then reload the page.") from None
    return {"graphMode": mode, "llm": llm_available(), "minLikes": MIN_LIKES, "defaultAlpha": config.ALPHA}


@lru_cache(maxsize=1)
def _catalog() -> list[dict]:
    return [{"id": m["movieId"], "title": m["displayTitle"], "year": _year(m["year"]),
             "poster": poster_url(m["posterPath"]), "n": m["ratingCount"], "avg": round(m["avgRating"], 2),
             "genres": m["genres"], "themes": m["themes"]} for m in recommender.movie_catalog()]


@app.get("/api/catalog")
def catalog():
    return _catalog()


@app.get("/api/users")
def users():
    return recommender.user_ids()


class ProfileRequest(BaseModel):
    liked: list[int] = Field(min_length=MIN_LIKES, max_length=200)
    k: int = Field(10, ge=1, le=30)
    alpha: float = Field(config.ALPHA, ge=0.0, le=1.0)


@app.post("/api/recommend")
def recommend_profile(req: ProfileRequest):
    known = {m["id"] for m in _catalog()}
    liked = tuple(sorted({m for m in req.liked if m in known}))
    if len(liked) < MIN_LIKES:
        raise HTTPException(422, f"Like at least {MIN_LIKES} movies from the catalogue.")
    with _compute_lock:
        return _profile_result(liked, req.k, round(req.alpha, 2))


@app.get("/api/users/{user_id}/recommend")
def recommend_user(user_id: int, k: int = 10, alpha: float = config.ALPHA):
    if user_id <= 0 or not recommender.user_exists(user_id):
        raise HTTPException(404, f"User {user_id} is not in the graph.")
    if not (1 <= k <= 30 and 0.0 <= alpha <= 1.0):
        raise HTTPException(422, "k must be 1-30 and alpha 0-1.")
    with _compute_lock:
        return _user_result(user_id, k, round(alpha, 2))


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
