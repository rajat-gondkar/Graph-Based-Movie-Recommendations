"""TMDB client: movie details with retries and backoff (spec section 5.4).

Responses are cached by the caller in data/cache/tmdb_cache.json (see scripts/02_fetch_tmdb.py).
Error messages never include request URLs, because a v3 key travels in the query string.
"""
import time

import requests

import config

RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 5


def make_session(credential: str) -> requests.Session:
    """HTTP session for a v3 API key (api_key parameter) or a v4 read token (Bearer), auto-detected (D11)."""
    session = requests.Session()
    session.headers["Accept"] = "application/json"
    if credential.startswith("eyJ") and credential.count(".") == 2:  # JWT-shaped v4 read access token
        session.headers["Authorization"] = f"Bearer {credential}"
    else:
        session.params = {"api_key": credential}
    return session


def _retry_wait(response: requests.Response, default: float) -> float:
    """Seconds to wait before retrying: TMDB's Retry-After header if present, else the backoff default."""
    try:
        return max(float(response.headers.get("Retry-After", default)), 0.0)
    except ValueError:
        return default


def fetch_movie(session: requests.Session, tmdb_id: int) -> dict:
    """GET /movie/{id} and return a cache entry: {'status': 200, 'data': {...}} or {'status': 404}.

    Retries network errors, HTTP 429 and 5xx with exponential backoff (1, 2, 4, 8 s).
    Raises RuntimeError when it gives up, so the caller can skip the movie without caching it.
    """
    url = f"{config.TMDB_BASE_URL}/movie/{int(tmdb_id)}"
    error = "unknown error"
    for attempt in range(MAX_ATTEMPTS):
        wait = float(2 ** attempt)
        try:
            response = session.get(url, params={"language": "en-US"}, timeout=20)
        except requests.RequestException as exc:
            error = type(exc).__name__  # type only: the message would contain the URL and key
        else:
            if response.status_code == 200:
                return {"status": 200, "data": response.json()}
            if response.status_code == 404:
                return {"status": 404}
            if response.status_code == 401:
                raise SystemExit("TMDB rejected the credential (HTTP 401). Check TMDB_API_KEY in .env.")
            if response.status_code not in RETRY_STATUSES:
                raise RuntimeError(f"HTTP {response.status_code}")
            error = f"HTTP {response.status_code}"
            wait = _retry_wait(response, wait)
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(wait)
    raise RuntimeError(f"gave up after {MAX_ATTEMPTS} attempts ({error})")


def movie_fields(entry: dict | None) -> dict:
    """The TMDB fields kept for each movie (spec 5.4), taken from a cache entry. Missing values are empty."""
    data = (entry or {}).get("data") or {}
    return {
        "overview": (data.get("overview") or "").strip(),
        "tagline": (data.get("tagline") or "").strip(),
        "poster_path": data.get("poster_path") or "",
        "release_date": data.get("release_date") or "",
        "vote_average": data.get("vote_average"),
    }


def poster_url(poster_path) -> str | None:
    """Full poster URL on TMDB's image CDN, or None when there is no poster."""
    if isinstance(poster_path, str) and poster_path.strip():
        return f"{config.TMDB_POSTER_BASE}{poster_path.strip()}"
    return None
