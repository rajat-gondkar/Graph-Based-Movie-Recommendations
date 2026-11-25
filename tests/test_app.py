"""Dashboard tests with Streamlit's AppTest (Phase 5). Need Neo4j in full mode; no LLM calls."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import llm_client, recommender  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    try:
        recommender.get_driver().verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not reachable: {exc}")
    monkeypatch.setattr(llm_client, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.radio(key="explain_mode").set_value("Template").run()
    return at


def _temp_users() -> int:
    return recommender._read("MATCH (u:User) WHERE u.userId < 0 RETURN count(u) AS n")[0]["n"]


def test_browse_like_and_recommend():
    at = _app()
    assert not at.exception and at.header[0].value == "Pick movies you like"
    likes = [b.key for b in at.button if b.key and b.key.startswith("like_")]
    assert len(likes) == 24
    get = [b for b in at.sidebar.button if b.label == "Get recommendations"][0]
    assert get.disabled                                                  # fewer than 3 likes
    for key in likes[:3]:
        at.button(key=key).click().run()
    assert len(at.session_state.liked) == 3
    [b for b in at.sidebar.button if b.label == "Get recommendations"][0].click().run()
    assert not at.exception and at.header[0].value == "Recommended for you"
    assert len(at.expander) >= 1
    assert sum(m.value.startswith("Recommended because") for m in at.markdown) >= 1
    assert _temp_users() == 0


def test_theme_browse_and_search():
    at = _app()
    at.segmented_control(key="browse_by").set_value("Theme").run()
    assert not at.exception
    at.text_input(key="search").input("Matrix").run()
    assert any(c.value.startswith('Showing 4 of 4 movies matching "Matrix"') for c in at.caption)


def test_movielens_user_mode():
    at = _app()
    at.radio(key="who").set_value("A MovieLens user").run()
    assert not at.exception and at.header[0].value == "Recommendations for a MovieLens user"
    assert any("10 recommendations" in c.value for c in at.caption)
