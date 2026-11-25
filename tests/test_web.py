"""Web app tests (Phase 5): the FastAPI backend and the static page. Need Neo4j in full mode; no paid LLM calls.

The LLM client is replaced by a function that raises, so a profile that isn't in the explanation cache exercises
the template fallback instead of spending money.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from src import llm_client, recommender  # noqa: E402


@pytest.fixture(scope="module")
def client():
    try:
        recommender.get_driver().verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not reachable: {exc}")
    from app.server import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("LLM disabled in tests")
    monkeypatch.setattr(llm_client, "chat", fail)


def _temp_users() -> int:
    return recommender._read("MATCH (u:User) WHERE u.userId < 0 RETURN count(u) AS n")[0]["n"]


def test_page_and_assets(client):
    assert client.get("/").status_code == 200 and "<main" in client.get("/").text
    for asset in ("/static/styles.css", "/static/app.js"):
        assert client.get(asset).status_code == 200


def test_status_catalog_users(client):
    status = client.get("/api/status").json()
    assert status["graphMode"] == "full" and status["minLikes"] == 3
    catalog = client.get("/api/catalog").json()
    assert len(catalog) == 2426
    assert {"id", "title", "year", "poster", "n", "avg", "genres", "themes"} <= set(catalog[0])
    assert all(m["themes"] for m in catalog)
    assert len(client.get("/api/users").json()) == 999


def test_validation(client):
    assert client.post("/api/recommend", json={"liked": [1, 2]}).status_code == 422          # too few likes
    assert client.post("/api/recommend", json={"liked": [1, 2, 999999999]}).status_code == 422  # unknown id
    assert client.post("/api/recommend", json={"liked": [1, 2, 3], "alpha": 2}).status_code == 422
    assert client.get("/api/users/1/recommend").status_code == 404
    assert client.get("/api/users/877/recommend?k=99").status_code == 422


def test_user_recommendations_from_cache(client):
    data = client.get("/api/users/877/recommend?k=10&alpha=0.8").json()
    recs = data["recommendations"]
    assert len(recs) == 10 and all(r["source"] == "cache" for r in recs)   # explained in Phase 4, no new calls
    assert recs[0]["evidence"]["evidence"] and recs[0]["themes"]
    assert len(data["topRated"]) == 8 and data["profile"]["themes"]


def test_profile_recommendations_and_fallback(client):
    liked = [1, 2571, 4993, 260]
    data = client.post("/api/recommend", json={"liked": liked, "k": 5, "alpha": 1.0}).json()
    recs = data["recommendations"]
    assert len(recs) == 5 and not {r["movieId"] for r in recs} & set(liked)
    assert all(r["source"] in {"cache", "template-fallback"} for r in recs)  # LLM disabled -> template text
    assert all(r["explanation"] for r in recs)
    assert _temp_users() == 0
