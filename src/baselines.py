"""Baseline recommenders for Phase 6 (D28), all CPU-only and fitted on train ratings.

Every baseline has fit(train) and recommend(user_id, k, exclude) and only ranks movies in `catalog`.
- Popularity: most train ratings first.
- Item-kNN: adjusted cosine (ratings centred by user mean) between items, 50 neighbours per item;
  score(u, i) = sum over the user's items j of sim(i, j) * (r_uj - mean_u).
- SVD: TruncatedSVD with 50 components on the user-mean-centred rating matrix (seed 42); score = reconstruction.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as l2_normalize

import config


class _MatrixModel:
    """Shared user/item indexing and top-k selection over the catalogue."""

    def __init__(self, catalog: set[int]):
        self.catalog = catalog

    def _index(self, train: pd.DataFrame):
        self.users = {u: i for i, u in enumerate(sorted(train["userId"].unique()))}
        self.items = {m: j for j, m in enumerate(sorted(train["movieId"].unique()))}
        self.item_ids = np.array(sorted(self.items, key=self.items.get))
        rows = train["userId"].map(self.users).to_numpy()
        cols = train["movieId"].map(self.items).to_numpy()
        means = train.groupby("userId")["rating"].mean()
        centred = train["rating"].to_numpy() - train["userId"].map(means).to_numpy()
        shape = (len(self.users), len(self.items))
        self.centred = sparse.csr_matrix((centred, (rows, cols)), shape=shape)
        self.in_catalog = np.isin(self.item_ids, list(self.catalog))

    def _top(self, scores: np.ndarray, k: int, exclude: set[int]) -> list[int]:
        scores = np.where(self.in_catalog, scores, -np.inf)
        if exclude:
            mask = np.isin(self.item_ids, list(exclude))
            scores = np.where(mask, -np.inf, scores)
        order = np.lexsort((self.item_ids, -scores))  # score descending, ties by movieId
        return [int(self.item_ids[j]) for j in order[:k] if np.isfinite(scores[j])]


class Popularity:
    name = "Popularity"

    def __init__(self, catalog: set[int]):
        self.catalog = catalog

    def fit(self, train: pd.DataFrame) -> "Popularity":
        counts = train[train["movieId"].isin(self.catalog)].groupby("movieId").size()
        self.ranked = [int(m) for m in counts.reset_index(name="n").sort_values(["n", "movieId"],
                                                                                 ascending=[False, True])["movieId"]]
        return self

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        return [m for m in self.ranked if m not in exclude][:k]


class ItemKNN(_MatrixModel):
    name = "Item-kNN"

    def __init__(self, catalog: set[int], neighbours: int = 50):
        super().__init__(catalog)
        self.neighbours = neighbours

    def fit(self, train: pd.DataFrame) -> "ItemKNN":
        self._index(train)
        items = l2_normalize(self.centred.T.tocsr(), axis=1)          # item vectors, unit length
        sim = (items @ items.T).toarray()
        np.fill_diagonal(sim, 0.0)
        # keep the `neighbours` most similar items per row
        if sim.shape[0] > self.neighbours:
            cut = np.argpartition(-sim, self.neighbours, axis=1)[:, self.neighbours:]
            np.put_along_axis(sim, cut, 0.0, axis=1)
        self.sim = sim
        return self

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        row = self.centred.getrow(self.users[user_id])
        scores = np.asarray(row @ self.sim.T).ravel()   # sum_j sim(i, j) * centred rating of j
        return self._top(scores, k, exclude)


class SVD(_MatrixModel):
    name = "SVD"

    def __init__(self, catalog: set[int], components: int = 50):
        super().__init__(catalog)
        self.components = components

    def fit(self, train: pd.DataFrame) -> "SVD":
        self._index(train)
        svd = TruncatedSVD(n_components=self.components, random_state=config.RANDOM_SEED)
        self.user_factors = svd.fit_transform(self.centred)
        self.item_factors = svd.components_
        return self

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        scores = self.user_factors[self.users[user_id]] @ self.item_factors
        return self._top(scores, k, exclude)
