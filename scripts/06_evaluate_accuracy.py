"""Phase 6A: ranking accuracy of the graph recommender against baselines (spec 11.1, D27, D28, D48, D49).

Final test evaluation (eval graph = train ratings only):
    python scripts/05_build_graph.py --mode eval --wipe
    python scripts/06_evaluate_accuracy.py
Tuning on the validation split (val graph = earlier 80% of train; scored against the later 20% of train, D49):
    python scripts/05_build_graph.py --mode val --wipe
    python scripts/06_evaluate_accuracy.py --split val
Outputs (test): results/accuracy_results.csv, alpha_sweep.csv, accuracy_chart.png, alpha_sweep.png, accuracy_stats.json
Outputs (val):  results/validation_results.csv, results/validation_grid.csv, results/validation_stats.json
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import baselines, graph_builder, recommender  # noqa: E402
from src import evaluation as ev  # noqa: E402

KS = (5, 10)
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CF_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)   # D48 grid, validation split only
GRID_ALPHAS = (0.4, 0.6, 0.8, 1.0)
METRICS = ["precision", "recall", "ndcg", "hit_rate", "coverage", "avg_popularity"]


def fetch_pools(users: list[int], workers: int, cf_pool: int) -> dict[int, dict]:
    """Each user's candidate pool, fetched once: theme pool (marked) plus, if cf_pool > 0, similarity candidates."""
    def one(uid):
        data = recommender.fetch_pool(uid, cf_pool=0)
        theme_ids = {c["movieId"] for c in data["pool"]}
        if cf_pool:
            data = recommender.fetch_pool(uid, cf_pool=cf_pool)
        for c in data["pool"]:
            c["inThemePool"] = c["movieId"] in theme_ids
        return uid, data
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(tqdm(pool.map(one, users), total=len(users), desc="graph pools", unit="user"))


def ranked_lists(pools: dict[int, dict], alpha: float, cf_weight: float) -> tuple[dict[int, list[int]], int]:
    """Top-10 per user. cf_weight = 0 uses the theme pool only, i.e. exactly the theme-based recommender."""
    lists, fills = {}, 0
    for uid, data in pools.items():
        if cf_weight == 0:
            data = {**data, "pool": [c for c in data["pool"] if c["inThemePool"]]}
        recs = recommender.rerank(data, k=max(KS), alpha=alpha, cf_weight=cf_weight)
        lists[uid] = [r["movieId"] for r in recs]
        fills += sum(r["filled"] for r in recs)
    return lists, fills


def chart_methods(df: pd.DataFrame, path: Path) -> None:
    sub = df[df["k"] == 10].set_index("method")[["precision", "recall", "ndcg", "hit_rate"]]
    ax = sub.T.plot(kind="bar", figsize=(9, 4.5), width=0.8, rot=0)
    ax.set_title("Accuracy at K = 10 (eval graph, train ratings only)")
    ax.set_ylabel("score")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def chart_alpha(df: pd.DataFrame, path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for metric in ("precision", "recall", "ndcg", "hit_rate"):
        left.plot(df["alpha"], df[metric], marker="o", label=metric)
    left.set_xlabel("α (1.0 = graph only)")
    left.set_title("Accuracy at K = 10")
    left.legend(fontsize=8, frameon=False)
    right.plot(df["alpha"], df["coverage"], marker="o", color="tab:green", label="coverage")
    right.set_xlabel("α")
    right.set_ylabel("catalogue coverage", color="tab:green")
    twin = right.twinx()
    twin.plot(df["alpha"], df["avg_popularity"], marker="s", color="tab:red", label="avg popularity")
    twin.set_ylabel("avg train ratings of recommended films", color="tab:red")
    right.set_title("Coverage and popularity at K = 10")
    for a in (left, right):
        a.spines[["top"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ranking accuracy on the eval graph.")
    parser.add_argument("--workers", type=int, default=4, help="parallel Neo4j sessions for the graph pools")
    parser.add_argument("--limit-users", type=int, default=0, help="evaluate only the first N users (0 = all)")
    parser.add_argument("--split", choices=["test", "val"], default="test",
                        help="test = final evaluation on the eval graph; val = tuning on the val graph (D49)")
    args = parser.parse_args()
    started = time.time()
    val = args.split == "val"

    mode, needed = recommender.graph_mode(), ("val" if val else "eval")
    if mode != needed:
        raise SystemExit(f"The graph is in '{mode}' mode; --split {args.split} needs the '{needed}' graph (D19, D49):\n"
                         f"  python scripts/05_build_graph.py --mode {needed} --wipe")

    print(f"[1/4] Catalogue and relevant items (D27), split: {args.split}")
    train, test = pd.read_csv(config.TRAIN_PATH), pd.read_csv(config.TEST_PATH)
    if val:
        train, test = graph_builder.validation_split(train)   # fit on the earlier 80% of train, score on the rest
    graph_movies = {r["id"] for r in recommender._read("MATCH (m:Movie) RETURN m.movieId AS id")}
    train = train[train["movieId"].isin(graph_movies)]
    test = test[test["movieId"].isin(graph_movies)]
    catalog = ev.build_catalog(train, graph_movies)
    relevant = ev.relevant_items(test, catalog)
    users = sorted(relevant)
    if args.limit_users:
        users = users[:args.limit_users]
        relevant = {u: relevant[u] for u in users}
    popularity = train.groupby("movieId").size().to_dict()
    seen = {int(u): set(g["movieId"]) for u, g in train.groupby("userId")}
    print(f"  catalogue: {len(catalog):,} films (graph films with >= {config.MIN_CANDIDATE_RATINGS} train ratings); "
          f"users with a relevant test item: {len(users):,}; relevant items: {sum(map(len, relevant.values())):,}")

    print("[2/4] Baselines (D28)")
    lists: dict[str, dict[int, list[int]]] = {}
    for model in (baselines.Popularity(catalog), baselines.ItemKNN(catalog), baselines.SVD(catalog)):
        t = time.time()
        model.fit(train)
        lists[model.name] = {u: model.recommend(u, max(KS), seen.get(u, set())) for u in users}
        print(f"  {model.name}: {time.time() - t:.1f} s")

    print("[3/4] Graph recommender (one pool per user, re-ranked for every setting)")
    t = time.time()
    use_cf = val or config.CF_WEIGHT > 0
    pools = fetch_pools(users, args.workers, config.CF_POOL_SIZE if use_cf else 0)
    graph_seconds = time.time() - t
    by_alpha, fills = {}, {}
    for a in ALPHAS:
        by_alpha[a], fills[a] = ranked_lists(pools, a, 0.0)
    graph_info = {"users_with_pool_under_10": sum(sum(c["inThemePool"] for c in d["pool"]) < max(KS)
                                                  for d in pools.values()),
                  "fill_ins_per_alpha": fills}
    lists["Graph pure (α = 1.0)"] = by_alpha[1.0]
    lists["Graph hybrid (α = 0.8)"] = by_alpha[0.8]
    if not val and config.CF_WEIGHT > 0:
        name = f"Graph themes + similar (w = {config.CF_WEIGHT}, α = {config.ALPHA_SIMILAR})"
        lists[name], graph_info["fill_ins_similar"] = ranked_lists(pools, config.ALPHA_SIMILAR, config.CF_WEIGHT)
        themed = {uid: {c["movieId"] for c in d["pool"] if c["matchedThemes"]} for uid, d in pools.items()}
        items = [(uid, m) for uid, lst in lists[name].items() for m in lst]
        graph_info["theme_path_share_similar"] = round(sum(m in themed[uid] for uid, m in items) / len(items), 4)
    leaks = sum(bool(set(lst) & seen.get(u, set())) for lst in lists.values() for u, lst in lst.items())
    left_catalog = sum(bool(set(l_) - catalog) for lst in lists.values() for l_ in lst.values())
    print(f"  {graph_seconds:.1f} s; lists containing a train item: {leaks}; lists with a film outside the catalogue: "
          f"{left_catalog}; users with a pool under 10: {graph_info['users_with_pool_under_10']}")

    print("[4/4] Metrics")
    rows = [{"method": name, "k": k, **ev.evaluate_lists(lst, relevant, k, catalog, popularity)}
            for k in KS for name, lst in lists.items()]
    results = pd.DataFrame(rows)
    if val:
        grid = []
        for w in CF_WEIGHTS:
            for a in GRID_ALPHAS:
                lst, n_fill = ranked_lists(pools, a, w)
                grid.append({"cf_weight": w, "alpha": a, "fill_ins": n_fill,
                             **ev.evaluate_lists(lst, relevant, 10, catalog, popularity)})
        grid = pd.DataFrame(grid).sort_values("ndcg", ascending=False)
        results.to_csv(config.RESULTS_DIR / "validation_results.csv", index=False, float_format="%.4f")
        grid.to_csv(config.RESULTS_DIR / "validation_grid.csv", index=False, float_format="%.4f")
        stats = {"split": "val", "catalog": len(catalog), "users": len(users),
                 "relevant_items": sum(map(len, relevant.values())), "graph_seconds": round(graph_seconds, 1),
                 "total_seconds": round(time.time() - started, 1), "lists_with_train_items": leaks,
                 "lists_outside_catalog": left_catalog, "best": grid.iloc[0].to_dict(), **graph_info}
        (config.RESULTS_DIR / "validation_stats.json").write_text(json.dumps(stats, indent=2, default=float) + "\n")
        pd.set_option("display.width", 160)
        print("\nValidation, K = 10")
        print(results[results["k"] == 10].drop(columns="k").to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nGrid (K = 10, sorted by NDCG): similarity weight w inside the graph score, α = graph vs popularity")
        print(grid.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nWrote results/validation_results.csv, validation_grid.csv, validation_stats.json "
              f"({time.time() - started:.1f} s)")
        return
    sweep = pd.DataFrame([{"alpha": a, **ev.evaluate_lists(by_alpha[a], relevant, 10, catalog, popularity)}
                          for a in ALPHAS])
    results.to_csv(config.RESULTS_DIR / "accuracy_results.csv", index=False, float_format="%.4f")
    sweep.to_csv(config.RESULTS_DIR / "alpha_sweep.csv", index=False, float_format="%.4f")
    chart_methods(results, config.RESULTS_DIR / "accuracy_chart.png")
    chart_alpha(sweep, config.RESULTS_DIR / "alpha_sweep.png")
    stats = {"catalog": len(catalog), "users": len(users), "relevant_items": sum(map(len, relevant.values())),
             "graph_seconds": round(graph_seconds, 1), "total_seconds": round(time.time() - started, 1),
             "lists_with_train_items": leaks, "lists_outside_catalog": left_catalog, **graph_info}
    (config.RESULTS_DIR / "accuracy_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    pd.set_option("display.width", 140)
    for k in KS:
        print(f"\nK = {k}")
        print(results[results["k"] == k].drop(columns="k").to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nα sweep (K = 10)")
    print(sweep.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote results/accuracy_results.csv, alpha_sweep.csv, accuracy_chart.png, alpha_sweep.png, "
          f"accuracy_stats.json ({time.time() - started:.1f} s)")


if __name__ == "__main__":
    main()
