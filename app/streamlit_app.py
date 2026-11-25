"""Streamlit dashboard for the movie knowledge-graph recommender (Phase 5).

Run from the project folder:  streamlit run app/streamlit_app.py

Two modes:
- Build my own profile: browse movies by genre or theme (or search), like the ones you enjoyed, then get
  recommendations. Liked movies are treated as 5-star ratings on a temporary graph user that is deleted afterwards.
- A MovieLens user: pick a user from the dataset and see their recommendations.
Every recommendation shows its explanation and the graph path (liked movies -> shared theme -> recommended movie).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `config` and `src` import when run by streamlit

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired  # noqa: E402

import config  # noqa: E402
from src import explainer, recommender  # noqa: E402
from src.tmdb_client import poster_url  # noqa: E402

LIKE_RATING = 5.0          # a "like" counts as a 5-star rating (D42)
MIN_LIKES = 3              # fewer likes give a thin taste profile
PAGE_SIZE = 24             # movies per "page" in the browser
GRID_COLUMNS = 6
DB_ERRORS = (ServiceUnavailable, SessionExpired, Neo4jError, OSError)
EXPLAIN_MODES = {"LLM": "llm", "Template": "template", "None": "none"}

st.set_page_config(page_title="Movie KG Recommender", page_icon="🎬", layout="wide")


# ---------------------------------------------------------------------------
# Data access (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def startup() -> dict:
    """Connect once, delete leftover temporary users (D24), and read the graph mode."""
    recommender.get_driver().verify_connectivity()
    return {"removed": recommender.cleanup_temporary_users(), "mode": recommender.graph_mode()}


@st.cache_data(show_spinner="Loading the movie catalog...")
def load_catalog() -> pd.DataFrame:
    return pd.DataFrame(recommender.movie_catalog())


@st.cache_data(show_spinner=False)
def load_user_ids() -> list[int]:
    return recommender.user_ids()


def llm_available() -> bool:
    try:
        config.get_llm_settings()
        return True
    except SystemExit:
        return False


def _explain(uid: int, recs: list[dict], mode: str, custom: bool) -> tuple[list[dict] | None, dict]:
    if mode == "none":
        return None, {}
    exps = explainer.explain_many(uid, recs, mode=mode, custom=custom)
    return exps, explainer.last_batch_stats()


@st.cache_data(show_spinner=False)
def results_for_profile(liked: tuple[int, ...], k: int, alpha: float, mode: str) -> dict:
    """Recommendations for a set of liked movies, via one temporary user for profile, ranking and explanations."""
    start = time.time()
    with recommender.temporary_user({mid: LIKE_RATING for mid in liked}) as uid:
        themes, genres = recommender.theme_profile(uid), recommender.genre_profile(uid)
        recs = recommender.recommend(uid, k=k, alpha=alpha)
        exps, stats = _explain(uid, recs, mode, custom=True)
    return {"themes": themes, "genres": genres, "recs": recs, "exps": exps, "stats": stats,
            "seconds": time.time() - start}


@st.cache_data(show_spinner=False)
def results_for_user(user_id: int, k: int, alpha: float, mode: str) -> dict:
    start = time.time()
    recs = recommender.recommend(user_id, k=k, alpha=alpha)
    exps, stats = _explain(user_id, recs, mode, custom=False)
    return {"themes": recommender.theme_profile(user_id), "genres": recommender.genre_profile(user_id),
            "recs": recs, "exps": exps, "stats": stats, "seconds": time.time() - start,
            "top": recommender.user_top_movies(user_id)}


# ---------------------------------------------------------------------------
# Small view helpers
# ---------------------------------------------------------------------------
def show_poster(path, title: str) -> None:
    url = poster_url(path)
    if url:
        st.image(url, width="stretch")
    else:
        st.markdown(f"<div style='aspect-ratio:2/3;display:flex;align-items:center;justify-content:center;"
                    f"text-align:center;padding:8px;border-radius:6px;background:rgba(128,128,128,.15)'>"
                    f"{title}</div>", unsafe_allow_html=True)


def year_text(year) -> str:
    return "" if year is None or pd.isna(year) else f" ({int(year)})"


def toggle_like(movie_id: int) -> None:
    liked = st.session_state.liked
    if movie_id in liked:
        liked.remove(movie_id)
    else:
        liked.append(movie_id)


def path_graph_html(ev: dict) -> str:
    """pyvis network of the evidence: You -> liked movies -> theme <- recommended movie."""
    from pyvis.network import Network

    dark = getattr(getattr(st.context, "theme", None), "type", None) == "dark"
    net = Network(height="340px", width="100%", directed=True, cdn_resources="in_line",
                  bgcolor="#0e1117" if dark else "#ffffff", font_color="#e6e6e6" if dark else "#262730")
    net.add_node("you", label="You", color="#4C72B0", shape="dot", size=18, level=0)
    net.add_node("rec", label=ev["recommended"]["title"], color="#DD8452", shape="star", size=24, level=3)
    for item in ev["evidence"]:
        theme_id = f"theme:{item['theme']}"
        net.add_node(theme_id, label=item["theme"], color="#55A868", shape="box", level=2)
        net.add_edge("rec", theme_id, label="HAS_THEME", color="#DD8452")
        for movie in item["because_you_liked"]:
            movie_id = f"movie:{movie['title']}"
            if movie_id not in net.get_nodes():
                net.add_node(movie_id, label=movie["title"], color="#8172B3", shape="dot", size=12, level=1)
                net.add_edge("you", movie_id, label=f"RATED {movie['rating']:g}", color="#4C72B0")
            net.add_edge(movie_id, theme_id, label="HAS_THEME", color="#8172B3")
    net.set_options("""{"layout": {"hierarchical": {"enabled": true, "direction": "LR", "levelSeparation": 190,
                       "nodeSpacing": 70}}, "physics": {"enabled": false},
                       "edges": {"font": {"size": 10, "align": "middle"}, "smooth": false},
                       "interaction": {"zoomView": false}}""")
    return net.generate_html()


def profile_charts(themes: list[dict], genres: dict) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Top themes in this taste profile**")
        if themes:
            df = pd.DataFrame(themes[:8]).set_index("theme")[["affinity"]]
            st.bar_chart(df, horizontal=True, sort="-affinity", height=260, x_label="affinity", y_label="")
        else:
            st.caption("No movie rated 4 stars or more, so there is no theme profile.")
    with right:
        st.markdown("**Top genres**")
        if genres:
            df = pd.DataFrame({"genre": list(genres)[:8], "affinity": list(genres.values())[:8]}).set_index("genre")
            st.bar_chart(df, horizontal=True, sort="-affinity", height=260, x_label="affinity", y_label="")


SOURCE_NOTE = {"llm": "written by the LLM from the evidence", "cache": "LLM text (cached)",
               "template": "template (no LLM)", "template-fallback": "template (LLM unavailable)"}


def recommendation_cards(data: dict) -> None:
    recs, exps = data["recs"], data["exps"]
    for start in range(0, len(recs), 3):
        cols = st.columns(3)
        for col, i in zip(cols, range(start, min(start + 3, len(recs)))):
            rec = recs[i]
            with col.container(border=True):
                img, body = st.columns([1, 2])
                with img:
                    show_poster(rec["posterPath"], rec["displayTitle"])
                with body:
                    st.markdown(f"**{i + 1}. {rec['displayTitle']}**{year_text(rec['year'])}")
                    if rec["filled"]:
                        st.caption("Popular pick (no theme match)")
                    else:
                        st.caption(f"Match score {rec['score']:.2f} · graph {rec['graphScore']:.2f} · "
                                   f"popularity {rec['popularity']:.2f}")
                        st.markdown(" ".join(f":violet-badge[{t['theme']}]" for t in rec["matchedThemes"][:3]))
                if exps:
                    st.markdown(exps[i]["text"])
                    st.caption(SOURCE_NOTE.get(exps[i]["source"], exps[i]["source"]))
                    if exps[i]["evidence"]["evidence"]:
                        with st.expander("Show graph path"):
                            # The HTML is generated here from graph data (MovieLens titles, cleaned theme names).
                            st.iframe(path_graph_html(exps[i]["evidence"]), height=360)
                            st.json(exps[i]["evidence"], expanded=False)


def results_summary(data: dict, mode: str) -> None:
    stats, fills = data["stats"], sum(r["filled"] for r in data["recs"])
    parts = [f"{len(data['recs'])} recommendations in {data['seconds']:.1f} s"]
    if mode == "llm" and stats:
        parts.append(f"{stats['llm_calls']} new LLM call(s), {stats['cache_hits']} cached")
        if stats["fallbacks"]:
            parts.append(f"{stats['fallbacks']} template fallback(s)")
    if fills:
        parts.append(f"{fills} popular pick(s) added because the theme match ran short")
    st.caption(" · ".join(parts))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def browse_page(catalog: pd.DataFrame) -> None:
    st.header("Pick movies you like")
    st.caption(f"Browse by genre or theme, press **Like** on movies you enjoyed (they count as 5 stars), then press "
               f"**Get recommendations** in the sidebar. Pick at least {MIN_LIKES}.")
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        browse_by = st.segmented_control("Browse by", ["Genre", "Theme"], default="Genre", key="browse_by") or "Genre"
    with c2:
        query = st.text_input("Search by title", placeholder="e.g. Inception", key="search").strip()
    with c3:
        sort = st.selectbox("Sort by", ["Most popular", "Highest rated", "Newest"], key="sort")

    if browse_by == "Genre":
        counts = catalog.explode("genres").genres.value_counts()
        choice = st.pills("Genre", ["All"] + list(counts.index), default="All", key="genre") or "All"
        view = catalog if choice == "All" else catalog[catalog.genres.apply(lambda g: choice in g)]
    else:
        counts = catalog.explode("themes").themes.value_counts()
        choice = st.selectbox("Theme", list(counts.index), key="theme",
                              format_func=lambda t: f"{t} ({counts[t]} movies)")
        view = catalog[catalog.themes.apply(lambda t: choice in t)]
    if query:
        view = catalog[catalog.title.str.contains(query, case=False, regex=False)]
    by = {"Most popular": (["ratingCount", "avgRating"], False), "Highest rated": (["avgRating", "ratingCount"], False),
          "Newest": (["year", "ratingCount"], False)}[sort]
    view = view.sort_values(by[0], ascending=by[1], na_position="last")

    filter_key = (browse_by, choice, query, sort)
    if st.session_state.get("filter_key") != filter_key:
        st.session_state.filter_key, st.session_state.shown = filter_key, PAGE_SIZE
    shown = view.head(st.session_state.shown)
    label = f"matching \"{query}\"" if query else (f"in {choice}" if choice != "All" else "in the catalog")
    st.caption(f"Showing {len(shown)} of {len(view)} movies {label}")

    liked = set(st.session_state.liked)
    rows = shown.to_dict("records")
    for start in range(0, len(rows), GRID_COLUMNS):
        cols = st.columns(GRID_COLUMNS)
        for col, movie in zip(cols, rows[start:start + GRID_COLUMNS]):
            with col.container(border=True):
                show_poster(movie["posterPath"], movie["displayTitle"])
                st.markdown(f"**{movie['displayTitle']}**{year_text(movie['year'])}")
                st.caption(f"★ {movie['avgRating']:.1f} · {movie['ratingCount']} ratings")
                is_liked = movie["movieId"] in liked
                st.button("♥ Liked" if is_liked else "♡ Like", key=f"like_{movie['movieId']}",
                          type="primary" if is_liked else "secondary", width="stretch",
                          on_click=toggle_like, args=(movie["movieId"],))
    if len(view) > len(shown) and st.button("Show more movies"):
        st.session_state.shown += PAGE_SIZE
        st.rerun()


def profile_results_page(catalog: pd.DataFrame, k: int, alpha: float, mode: str) -> None:
    if st.button("← Back to browsing"):
        st.session_state.view = "browse"
        st.rerun()
    liked = tuple(sorted(st.session_state.liked))
    titles = catalog.set_index("movieId").loc[list(liked), "displayTitle"].tolist()
    st.header("Recommended for you")
    st.caption("Based on " + ", ".join(titles))
    with st.spinner("Walking the knowledge graph and writing explanations..."):
        data = results_for_profile(liked, k, alpha, mode)
    results_summary(data, mode)
    profile_charts(data["themes"], data["genres"])
    recommendation_cards(data)


def movielens_user_page(k: int, alpha: float, mode: str) -> None:
    st.header("Recommendations for a MovieLens user")
    ids = load_user_ids()
    user_id = st.selectbox("MovieLens user", ids, index=ids.index(877) if 877 in ids else 0, key="ml_user")
    with st.spinner("Walking the knowledge graph and writing explanations..."):
        data = results_for_user(user_id, k, alpha, mode)
    with st.expander(f"Movies user {user_id} rated highest", expanded=False):
        st.write(", ".join(f"{m['displayTitle']} ({m['rating']:g}★)" for m in data["top"]))
    results_summary(data, mode)
    profile_charts(data["themes"], data["genres"])
    recommendation_cards(data)


def sidebar(catalog: pd.DataFrame, has_llm: bool) -> tuple[str, int, float, str]:
    with st.sidebar:
        st.title("🎬 Movie KG")
        who = st.radio("Who is watching?", ["Build my own profile", "A MovieLens user"], key="who")
        st.subheader("Settings")
        k = st.slider("Number of recommendations", 5, 20, 10)
        alpha = st.slider("Theme match vs popularity (α)", 0.0, 1.0, float(config.ALPHA), 0.1,
                          help="1.0 = rank only by the knowledge graph (themes and genres); "
                               "lower values blend in how popular a movie is.")
        options = list(EXPLAIN_MODES) if has_llm else ["Template", "None"]
        mode = EXPLAIN_MODES[st.radio("Explanations", options, horizontal=True, key="explain_mode")]
        if not has_llm:
            st.caption("No LLM key in .env, so explanations use the template.")

        if who == "Build my own profile":
            st.subheader(f"Your picks ({len(st.session_state.liked)})")
            titles = catalog.set_index("movieId").displayTitle
            for mid in list(st.session_state.liked):
                left, right = st.columns([5, 1])
                left.write(titles.get(mid, mid))
                right.button("✕", key=f"remove_{mid}", on_click=toggle_like, args=(mid,), help="Remove")
            enough = len(st.session_state.liked) >= MIN_LIKES
            if st.button("Get recommendations", type="primary", disabled=not enough, width="stretch"):
                st.session_state.view = "results"
                st.rerun()
            if not enough:
                st.caption(f"Like at least {MIN_LIKES} movies to continue.")
            if st.session_state.liked and st.button("Clear all", width="stretch"):
                st.session_state.liked, st.session_state.view = [], "browse"
                st.rerun()
    return who, k, alpha, mode


def footer() -> None:
    st.divider()
    st.caption("Every explanation is built from a real path in the knowledge graph: movies you liked → a theme they "
               "share → the recommended movie. In LLM mode the model only rephrases that evidence and is told not to "
               "add any other facts; template mode uses no LLM at all. Themes were extracted from TMDB plot summaries "
               "by an LLM.")
    st.caption("Ratings: MovieLens 32M (GroupLens). Posters and plots: TMDB. This product uses the TMDB API but is "
               "not endorsed or certified by TMDB.")


def main() -> None:
    st.session_state.setdefault("liked", [])
    st.session_state.setdefault("view", "browse")
    try:
        info = startup()
        catalog = load_catalog()
    except DB_ERRORS as exc:
        st.error("Can't reach the Neo4j database. Start it from the project folder with `colima start` (macOS) and "
                 "`docker compose up -d`, then reload this page.")
        st.caption(f"Details: {exc}")
        st.stop()
    if info["mode"] != "full":
        st.warning(f"The graph is in '{info['mode']}' mode (train ratings only). For the demo, rebuild it with "
                   "`python scripts/05_build_graph.py --mode full --wipe`.")

    who, k, alpha, mode = sidebar(catalog, llm_available())
    try:
        if who == "A MovieLens user":
            movielens_user_page(k, alpha, mode)
        elif st.session_state.view == "results" and len(st.session_state.liked) >= MIN_LIKES:
            profile_results_page(catalog, k, alpha, mode)
        else:
            browse_page(catalog)
    except DB_ERRORS as exc:
        st.error("Lost the connection to Neo4j. Check that it is running (`docker compose ps`) and reload.")
        st.caption(f"Details: {exc}")
    footer()


main()
