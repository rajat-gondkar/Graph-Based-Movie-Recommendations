"""Fast unit tests: no network, no Neo4j, no LLM. Run with `pytest -q`."""
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import data_loading as dl  # noqa: E402
from src import explainer, recommender  # noqa: E402
from src import llm_client  # noqa: E402
from src import theme_extraction as te  # noqa: E402
from src import theme_normalization as tn  # noqa: E402


# --- Phase 1: data preparation -------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("Matrix, The (1999)", ("The Matrix", 1999)),
    ("Seven (a.k.a. Se7en) (1995)", ("Seven", 1995)),
    ("Toy Story (1995)", ("Toy Story", 1995)),
    ("Some Movie", ("Some Movie", None)),
])
def test_parse_title(raw, expected):
    assert dl.parse_title(raw) == expected


def test_clean_genres_drops_non_genres():
    assert dl.clean_genres("Adventure|IMAX|Animation") == "Adventure|Animation"
    assert dl.clean_genres("(no genres listed)") == ""


def test_split_by_time_takes_latest_20_percent():
    ratings = pd.DataFrame({"userId": [1] * 10 + [2] * 3, "movieId": list(range(10)) + [1, 2, 3],
                            "rating": 4.0, "timestamp": list(range(100, 110)) + [5, 3, 4]})
    train, test = dl.split_by_time(ratings, 0.2)
    assert sorted(test[test.userId == 1].movieId) == [8, 9]          # the two latest of ten
    assert list(test[test.userId == 2].movieId) == [1]                # max(1, round(0.6)) = 1, latest timestamp
    assert len(train) + len(test) == len(ratings)
    assert not set(zip(train.userId, train.movieId)) & set(zip(test.userId, test.movieId))


def test_filter_subset_needs_tmdb_and_counts():
    ratings = pd.DataFrame({"userId": [1, 1, 2, 2, 3], "movieId": [10, 11, 10, 11, 10], "rating": 4.0, "timestamp": 1})
    links = pd.DataFrame({"movieId": [10, 11], "tmdbId": [100, None]})
    kept = dl.filter_subset(ratings, links, min_movie_ratings=2, min_user_ratings=1)
    assert set(kept.movieId) == {10}                                  # movie 11 has no tmdbId


# --- Phase 2: LLM reply parsing and theme cleaning ------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("Man vs. Nature", "man vs nature"),
    ("Coming-of-Age", "coming of age"),
    ("Loss & Grief", "loss and grief"),
    ("fear of the unknown", "fear of the unknown"),                  # 4-word seed is kept
    ("a very long made up theme", None),                             # non-seed over 3 words
    ("", None),
    (42, None),
])
def test_clean_theme(raw, expected):
    assert te.clean_theme(raw) == expected


def test_parse_json_object_handles_messy_replies():
    assert llm_client.parse_json_object('```json\n{"themes": ["a"]}\n```') == {"themes": ["a"]}
    assert llm_client.parse_json_object('<think>hmm {x}</think> Sure: {"themes": []} done') == {"themes": []}
    assert llm_client.parse_json_object("no json here") is None
    assert llm_client.parse_json_object("") is None


def test_parse_themes_validates_and_limits():
    reply = '{"themes": ["Redemption", "redemption", "Survival", "Greed", "Revenge", "Betrayal", "Loyalty"]}'
    assert te.parse_themes(reply) == ["redemption", "survival", "greed", "revenge", "betrayal"]
    assert te.parse_themes('{"themes": "redemption"}') is None
    assert te.parse_themes('{"themes": [1, 2]}') is None


def test_normalization_review_and_rare_drop():
    raw = pd.DataFrame({
        "movieId": [1, 1, 2, 2, 3, 3, 4],
        "theme": ["Revenge", "romance", "vengeance", "greed", "greed", "redemption", "one-off theme"],
    })
    final, mapping, stats = tn.normalize(raw, ["revenge", "redemption"], embedder=None, threshold=0.75,
                                         min_movies=2, overrides={"vengeance": "revenge"}, blocklist={"romance"})
    assert "romance" not in set(final.theme)                          # blocked genre word
    assert set(final[final.movieId == 2].theme) == {"revenge", "greed"}  # manual override
    assert set(final.movieId) == {1, 2, 3, 4}                         # movie 4 rescued, not left empty
    assert stats["raw_themes_blocked"] == 1 and stats["raw_themes_manual_override"] == 1


def test_config_review_settings_are_consistent():
    assert not set(config.THEME_BLOCKLIST) & set(config.SEED_THEMES)
    assert all(v not in config.THEME_BLOCKLIST for v in config.THEME_OVERRIDES.values())


# --- Phase 4: scoring and explanation text --------------------------------------
def test_minmax():
    assert recommender._minmax([]) == []
    assert recommender._minmax([3, 3]) == [1.0, 1.0]                  # D21: all equal -> 1.0
    assert recommender._minmax([0, 5, 10]) == [0.0, 0.5, 1.0]


def _toy_pool():
    themes = [{"theme": "redemption", "affinity": 2.0, "idf": 1.0, "contribution": 2.0}]
    return {"userId": 1, "genre_affinity": {"Drama": 2.0}, "pool": [
        {"movieId": 1, "title": "A", "year": 2000, "posterPath": None, "ratingCount": 10, "themeScore": 9.0,
         "matchedThemes": themes, "genres": ["Drama"]},
        {"movieId": 2, "title": "B", "year": 2001, "posterPath": None, "ratingCount": 500, "themeScore": 1.0,
         "matchedThemes": themes, "genres": []},
    ]}


def test_rerank_formula_and_alpha(monkeypatch):
    monkeypatch.setattr(recommender, "duplicate_titles", lambda: frozenset())
    pure = recommender.rerank(_toy_pool(), k=2, alpha=1.0, fill=False)
    popular = recommender.rerank(_toy_pool(), k=2, alpha=0.0, fill=False)
    assert [r["movieId"] for r in pure] == [1, 2]                     # higher theme + genre score wins
    assert [r["movieId"] for r in popular] == [2, 1]                  # higher rating count wins
    top = pure[0]
    expected_graph = (1.0 + config.GENRE_WEIGHT * 1.0) / (1 + config.GENRE_WEIGHT)
    assert math.isclose(top["graphScore"], expected_graph)
    assert all(0.0 <= r["score"] <= 1.0 for r in pure + popular)
    hybrid = recommender.rerank(_toy_pool(), k=2, alpha=0.8, fill=False)
    for r in hybrid:
        assert math.isclose(r["score"], 0.8 * r["graphScore"] + 0.2 * r["popularity"])


EVIDENCE = {
    "recommended": {"title": "Rec", "year": 2000},
    "evidence": [{"theme": "redemption", "because_you_liked": [{"title": "A", "rating": 5.0},
                                                               {"title": "B", "rating": 4.0}]}],
    "shared_genres": ["Drama"],
}


def test_template_text():
    assert explainer.template_text(EVIDENCE) == (
        "Recommended because you liked *A* and *B*, which share the theme **redemption** with *Rec*.")
    one = {**EVIDENCE, "evidence": [{"theme": "greed", "because_you_liked": [{"title": "A", "rating": 5.0}]}]}
    assert "which shares the theme **greed**" in explainer.template_text(one)
    none = {**EVIDENCE, "evidence": [], "shared_genres": []}
    assert explainer.template_text(none) == "*Rec* is one of the most popular movies you haven't rated yet."


def test_cache_key_depends_on_user_movie_and_evidence():
    key = explainer._cache_key("877", 1, EVIDENCE)
    assert key == explainer._cache_key("877", 1, dict(EVIDENCE))
    assert key != explainer._cache_key("878", 1, EVIDENCE)
    assert key != explainer._cache_key("877", 2, EVIDENCE)
    assert key != explainer._cache_key("877", 1, {**EVIDENCE, "shared_genres": []})
