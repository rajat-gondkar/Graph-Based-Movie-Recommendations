"""Evaluation helpers (Phase 6): the shared candidate catalogue, ranking metrics, and the faithfulness check.

D27: one catalogue for every method = graph movies with >= MIN_CANDIDATE_RATINGS train ratings, minus each user's
train items. Relevant = a test rating >= RELEVANT_THRESHOLD on a catalogue movie; users need >= 1 relevant item.
D30: faithfulness = every movie title and theme an explanation mentions is in its evidence object.
"""
import math
import re

import pandas as pd

import config


# ---------------------------------------------------------------------------
# Catalogue and relevant items
# ---------------------------------------------------------------------------
def build_catalog(train: pd.DataFrame, graph_movies: set[int]) -> set[int]:
    counts = train[train["movieId"].isin(graph_movies)].groupby("movieId").size()
    return set(counts[counts >= config.MIN_CANDIDATE_RATINGS].index)


def relevant_items(test: pd.DataFrame, catalog: set[int]) -> dict[int, set[int]]:
    rel = test[(test["rating"] >= config.RELEVANT_THRESHOLD) & test["movieId"].isin(catalog)]
    return {int(u): set(g["movieId"]) for u, g in rel.groupby("userId")}


# ---------------------------------------------------------------------------
# Ranking metrics (binary relevance)
# ---------------------------------------------------------------------------
def precision_at_k(recs: list[int], rel: set[int], k: int) -> float:
    return sum(m in rel for m in recs[:k]) / k


def recall_at_k(recs: list[int], rel: set[int], k: int) -> float:
    return sum(m in rel for m in recs[:k]) / len(rel) if rel else 0.0


def hit_rate_at_k(recs: list[int], rel: set[int], k: int) -> float:
    return float(any(m in rel for m in recs[:k]))


def ndcg_at_k(recs: list[int], rel: set[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, m in enumerate(recs[:k]) if m in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(rel))))
    return dcg / ideal if ideal else 0.0


def evaluate_lists(lists: dict[int, list[int]], relevant: dict[int, set[int]], k: int, catalog: set[int],
                   popularity: dict[int, int]) -> dict:
    """Mean metrics over users who have relevant items, plus catalogue coverage and average popularity."""
    users = [u for u in relevant if u in lists]
    rows = [(precision_at_k(lists[u], relevant[u], k), recall_at_k(lists[u], relevant[u], k),
             ndcg_at_k(lists[u], relevant[u], k), hit_rate_at_k(lists[u], relevant[u], k)) for u in users]
    recommended = [m for u in users for m in lists[u][:k]]
    n = len(rows) or 1
    return {
        "users": len(rows),
        "precision": sum(r[0] for r in rows) / n,
        "recall": sum(r[1] for r in rows) / n,
        "ndcg": sum(r[2] for r in rows) / n,
        "hit_rate": sum(r[3] for r in rows) / n,
        "coverage": len(set(recommended) & catalog) / len(catalog),
        "avg_popularity": sum(popularity.get(m, 0) for m in recommended) / (len(recommended) or 1),
    }


# ---------------------------------------------------------------------------
# Faithfulness (D30)
# ---------------------------------------------------------------------------
_YEAR = re.compile(r"\s*\(\d{4}\)$")
# Theme names that are also ordinary English words in explanations ("you will love ..."); skipped (D30).
AMBIGUOUS_THEMES = {"love"}


def normalize(text: str, lower: bool = True) -> str:
    """Lowercase (optional), straight quotes, '&' -> 'and', drop punctuation except word characters and spaces.
    Keeps the same length mapping either way, so cased and lowercased versions line up character by character."""
    text = text.replace("’", "'").replace("‘", "'").replace("&", " and ")
    text = text.lower() if lower else text
    text = re.sub(r"[*_\"“”]", " ", text)
    text = re.sub(r"[^\w' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def base_title(title: str) -> str:
    return _YEAR.sub("", title)


class MentionFinder:
    """Finds catalogue titles, themes and genres mentioned in a text (whole words, longest match first)."""

    def __init__(self, titles: list[str], themes: list[str], genres: list[str]):
        skip = {normalize(t) for t in themes} | {normalize(g) for g in genres}
        self.titles, self.one_word_titles = {}, {}
        for title in titles:
            key = normalize(base_title(title))
            if not key or key in skip:          # a title equal to a theme or genre ("Identity") is ambiguous
                continue
            if len(key.split()) > 1:
                self.titles.setdefault(key, base_title(title))
            elif len(key) > 3:                  # "Up", "Her", "It": too short to tell from ordinary words
                # one-word titles must appear capitalised ("Alien"), so "taken" or "home" in a sentence don't count
                self.one_word_titles.setdefault(normalize(base_title(title), lower=False), base_title(title))
        self.themes = {normalize(t): t for t in themes if t not in AMBIGUOUS_THEMES}
        self.genres = {normalize(g): g for g in genres}

    def mentions(self, text: str) -> dict[str, set[str]]:
        """All title/theme/genre matches; where two overlap (the film "Unknown" inside the theme "fear of the
        unknown", or "Star Wars" inside a longer title), the longest one wins."""
        text = re.sub(r"\(\d{4}\)", " ", text)  # "The Avengers (2012)": the year is not the film "2012"
        norm, cased = f" {normalize(text)} ", f" {normalize(text, lower=False)} "
        candidates = []
        for kind, names, hay in (("titles", self.titles, norm), ("titles", self.one_word_titles, cased),
                                 ("themes", self.themes, norm), ("genres", self.genres, norm)):
            for key, name in names.items():
                for m in re.finditer(r"(?<![\w'])" + re.escape(key) + r"(?![\w'])", hay):
                    candidates.append((m.end() - m.start(), m.start(), m.end(), kind, name))
        taken = [False] * len(norm)
        found = {"titles": set(), "themes": set(), "genres": set()}
        for _length, start, end, kind, name in sorted(candidates, key=lambda c: (-c[0], c[1])):
            if not any(taken[start:end]):
                taken[start:end] = [True] * (end - start)
                found[kind].add(name)
        return found


def check_faithfulness(text: str, evidence: dict, finder: MentionFinder) -> dict:
    """Which mentions are backed by the evidence. `faithful` covers titles and themes (spec 11.2);
    `genres_ok` separately checks genre words against the shared genres."""
    found = finder.mentions(text)
    ev_titles = {base_title(evidence["recommended"]["title"])}
    liked = set()
    for item in evidence["evidence"]:
        for m in item["because_you_liked"]:
            liked.add(base_title(m["title"]))
    ev_titles |= liked
    ev_themes = {item["theme"] for item in evidence["evidence"]}
    norm_ev_titles = {normalize(t) for t in ev_titles}
    bad_titles = {t for t in found["titles"] if normalize(t) not in norm_ev_titles}
    bad_themes = found["themes"] - ev_themes
    bad_genres = found["genres"] - set(evidence["shared_genres"])
    return {
        "faithful": not bad_titles and not bad_themes,
        "genres_ok": not bad_genres,
        "mentions_liked_title": bool({normalize(t) for t in found["titles"]} & {normalize(t) for t in liked}),
        "mentions_evidence_theme": bool(found["themes"] & ev_themes),
        "unsupported_titles": sorted(bad_titles),
        "unsupported_themes": sorted(bad_themes),
        "unsupported_genres": sorted(bad_genres),
        "titles": sorted(found["titles"]),
        "themes": sorted(found["themes"]),
    }
