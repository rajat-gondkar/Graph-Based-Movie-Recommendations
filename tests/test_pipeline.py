"""Checks on the real outputs: processed data files (Phases 1-2) and the live Neo4j graph and recommender (Phases 3-4).

No LLM calls: the LLM client is replaced by a function that fails the test if it is ever called.
Graph tests are skipped when Neo4j is not reachable. Run with `pytest -q` (about 30 s).
"""
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import explainer, graph_builder, llm_client, recommender  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return {
        "movies": pd.read_csv(config.MOVIES_ENRICHED_PATH),
        "themes": pd.read_csv(config.MOVIE_THEMES_PATH),
        "train": pd.read_csv(config.TRAIN_PATH),
        "test": pd.read_csv(config.TEST_PATH),
    }


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("A test tried to call the LLM")
    monkeypatch.setattr(llm_client, "chat", fail)


# --- Phase 1 and 2 outputs -------------------------------------------------------
def test_split_has_no_leakage(data):
    train, test = data["train"], data["test"]
    assert not set(zip(train.userId, train.movieId)) & set(zip(test.userId, test.movieId))
    last_train = train.groupby("userId").timestamp.max()
    first_test = test.groupby("userId").timestamp.min()
    assert (last_train <= first_test.reindex(last_train.index)).all()   # test is always later in time
    assert set(train.userId) == set(test.userId)
    share = len(test) / (len(train) + len(test))
    assert 0.19 < share < 0.21


def test_movies_and_themes(data):
    movies, themes = data["movies"], data["themes"]
    assert movies.movieId.is_unique and movies.overview.notna().all()
    per_movie = themes.groupby("movieId").theme.nunique()
    assert set(per_movie.index) == set(movies.movieId)                 # every usable movie has themes
    assert per_movie.between(1, 5).all()
    assert not themes.duplicated().any()
    assert 50 <= themes.theme.nunique() <= 150
    assert set(config.SEED_THEMES) <= set(themes.theme)
    assert not set(themes.theme) & set(config.THEME_BLOCKLIST)
    assert not set(themes.theme) & {k for k, v in config.THEME_OVERRIDES.items() if k != v}


# --- Phase 3: the graph ----------------------------------------------------------
@pytest.fixture(scope="module")
def driver():
    try:
        drv = recommender.get_driver()
        drv.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not reachable: {exc}")
    if recommender.graph_mode() != "full":
        pytest.skip("These tests expect the full graph (python scripts/05_build_graph.py --mode full --wipe)")
    return drv


def test_graph_matches_input_files(driver):
    checks = graph_builder.sanity_check(driver, graph_builder.load_inputs("full"))
    failed = [text for text, ok in checks if not ok]
    assert not failed, failed


def test_idf_and_rating_stats(driver, data):
    total = recommender._read("MATCH (m:Movie) RETURN count(m) AS n")[0]["n"]
    for t in recommender._read("MATCH (t:Theme) RETURN t.name AS name, t.movieCount AS n, t.idf AS idf"):
        assert math.isclose(t["idf"], math.log(total / t["n"]))
    graph = pd.DataFrame(recommender._read(
        "MATCH (m:Movie) RETURN m.movieId AS movieId, m.ratingCount AS n, m.avgRating AS avg")).set_index("movieId")
    ratings = pd.concat([data["train"], data["test"]])
    ratings = ratings[ratings.movieId.isin(graph.index)]
    expected = ratings.groupby("movieId").rating.agg(["size", "mean"])
    assert (graph.n == expected["size"].reindex(graph.index)).all()
    assert ((graph.avg - expected["mean"].reindex(graph.index)).abs() < 1e-9).all()


# --- Phase 4: recommender and explanations ---------------------------------------
@pytest.fixture(scope="module")
def sample_users(driver):
    rows = recommender._read("MATCH (u:User) WHERE u.userId > 0 RETURN u.userId AS id ORDER BY id")
    ids = [r["id"] for r in rows]
    return ids[::max(1, len(ids) // 20)][:20]                           # 20 users spread over the id range


def test_recommendations_are_valid(sample_users):
    for uid in sample_users:
        seen = {r["m"] for r in recommender._read(
            "MATCH (:User {userId: $u})-[:RATED]->(m:Movie) RETURN m.movieId AS m", u=uid)}
        profile = {p["theme"] for p in recommender.theme_profile(uid)}
        recs = recommender.recommend(uid, k=10)
        ids = [r["movieId"] for r in recs]
        assert len(recs) == 10 and len(set(ids)) == 10
        assert not set(ids) & seen, f"user {uid} got a movie they already rated"
        scores = [r["score"] for r in recs if not r["filled"]]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)
        for r in recs:
            assert r["ratingCount"] >= config.MIN_CANDIDATE_RATINGS
            assert {t["theme"] for t in r["matchedThemes"]} <= profile


def test_alpha_one_ranks_by_graph_score(sample_users):
    recs = recommender.recommend(sample_users[0], k=10, alpha=1.0)
    graph = [r["graphScore"] for r in recs]
    assert graph == sorted(graph, reverse=True)


def test_evidence_is_true_in_the_graph(sample_users):
    for uid in sample_users[:10]:
        rec = recommender.recommend(uid, k=1)[0]
        ev = explainer.evidence(uid, rec["movieId"])
        assert explainer.has_path(ev), f"user {uid}: top recommendation has no theme or similarity path"
        sims = {(r["t"], r["liked"]) for r in recommender._read("""
            MATCH (:User {userId: $u})-[r:RATED]->(l:Movie)<-[s:SIMILAR]-(:Movie {movieId: $m})
            WHERE r.rating >= $like RETURN l.title AS liked, l.year AS y, s.coLikes AS t""",
            u=uid, m=rec["movieId"], like=config.LIKE_THRESHOLD)}
        for m in ev[explainer.SIMILAR_KEY]:
            assert m["rating"] >= config.LIKE_THRESHOLD
            assert any(m["shared_fans"] == fans and m["title"].startswith(t) for fans, t in sims)
        rec_themes = {r["t"] for r in recommender._read(
            "MATCH (:Movie {movieId: $m})-[:HAS_THEME]->(t) RETURN t.name AS t", m=rec["movieId"])}
        liked = recommender._read("""
            MATCH (:User {userId: $u})-[r:RATED]->(m:Movie)-[:HAS_THEME]->(t:Theme)
            WHERE r.rating >= $like RETURN m.title AS title, m.year AS year, t.name AS theme""",
                                  u=uid, like=config.LIKE_THRESHOLD)
        liked_pairs = {(recommender.display_title(x["title"], x["year"]), x["theme"]) for x in liked}
        for item in ev["evidence"]:
            assert item["theme"] in rec_themes
            for movie in item["because_you_liked"]:
                assert movie["rating"] >= config.LIKE_THRESHOLD
                assert (movie["title"], item["theme"]) in liked_pairs
        text = explainer.template_text(ev)
        assert ev["recommended"]["title"] in text
        if ev["evidence"]:
            assert ev["evidence"][0]["theme"] in text


def test_explanation_cache_avoids_repeat_calls(sample_users, monkeypatch, tmp_path):
    """First run calls the (stubbed) LLM once per film; the second run is served entirely from the cache."""
    monkeypatch.setattr(config, "EXPLANATIONS_CACHE_PATH", tmp_path / "explanations.json")
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: "Stub explanation.")
    uid = sample_users[0]
    recs = recommender.recommend(uid, k=5)
    first = explainer.explain_many(uid, recs, mode="llm")
    assert explainer.last_batch_stats()["llm_calls"] == 5 and all(r["source"] == "llm" for r in first)
    second = explainer.explain_many(uid, recs, mode="llm")
    assert explainer.last_batch_stats()["llm_calls"] == 0 and all(r["source"] == "cache" for r in second)


def test_evidence_lists_up_to_the_configured_themes(sample_users):
    for uid in sample_users[:5]:
        rec = recommender.recommend(uid, k=1)[0]
        ev = explainer.evidence(uid, rec["movieId"])
        assert len(ev["evidence"]) <= config.EVIDENCE_MAX_THEMES and explainer.has_path(ev)
        assert all(len(e["because_you_liked"]) <= config.EVIDENCE_MAX_LIKED for e in ev["evidence"])
        shared = {t["theme"] for t in rec["matchedThemes"]}
        assert len(ev["evidence"]) == min(len(shared), config.EVIDENCE_MAX_THEMES)


def _temp_users() -> int:
    return recommender._read("MATCH (u:User) WHERE u.userId < 0 RETURN count(u) AS n")[0]["n"]


def test_cold_start_and_cleanup(driver):
    ratings = {1: 5.0, 260: 5.0, 2571: 4.5}
    recs = recommender.recommend_for_ratings(ratings, k=10)
    assert len(recs) == 10 and not {r["movieId"] for r in recs} & set(ratings)
    assert _temp_users() == 0
    with pytest.raises(RuntimeError):
        with recommender.temporary_user(ratings):
            raise RuntimeError("simulated crash")
    assert _temp_users() == 0                                           # deleted even after an error


def test_profile_without_likes_has_no_liked_path(driver):
    """No rating >= 4: no theme profile and no liked-film path. Since D48 the SIMILAR edges still supply
    candidates, and their explanations fall back to the plain 'popular' sentence (D41)."""
    ratings = {1: 2.0, 260: 3.0}
    with recommender.temporary_user(ratings) as uid:
        recs = recommender.recommend(uid, k=5)
        assert len(recs) == 5 and not {r["movieId"] for r in recs} & set(ratings)
        assert all(not r["matchedThemes"] for r in recs)
        for r in recs:
            ev = explainer.evidence(uid, r["movieId"])
            assert not explainer.has_path(ev)
            assert "popular" in explainer.template_text(ev)
    assert _temp_users() == 0
