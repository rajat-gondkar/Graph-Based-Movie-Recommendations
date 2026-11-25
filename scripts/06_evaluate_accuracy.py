"""Phase 6A: ranking accuracy of the graph recommender against baselines (spec 11.1, D27, D28).

Needs the eval graph (train ratings only):
    python scripts/05_build_graph.py --mode eval --wipe
    python scripts/06_evaluate_accuracy.py
Outputs: results/accuracy_results.csv, results/alpha_sweep.csv, results/accuracy_chart.png, results/alpha_sweep.png,
         results/accuracy_stats.json
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
from src import baselines, recommender  # noqa: E402
from src import evaluation as ev  # noqa: E402

KS = (5, 10)
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
METRICS = ["precision", "recall", "ndcg", "hit_rate", "coverage", "avg_popularity"]


def graph_lists(users: list[int], workers: int) -> tuple[dict[float, dict[int, list[int]]], dict]:
    """Fetch each user's theme pool once, then re-rank it for every alpha (top-10 each)."""
    def one(uid):
        return uid, recommender.fetch_pool(uid)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pools = dict(tqdm(pool.map(one, users), total=len(users), desc="graph pools", unit="user"))
    lists, fills, short_pools = {a: {} for a in ALPHAS}, {a: 0 for a in ALPHAS}, 0
    for uid, data in pools.items():
        short_pools += len(data["pool"]) < max(KS)
        for a in ALPHAS:
            recs = recommender.rerank(data, k=max(KS), alpha=a)
            lists[a][uid] = [r["movieId"] for r in recs]
            fills[a] += sum(r["filled"] for r in recs)
    return lists, {"users_with_pool_under_10": short_pools, "fill_ins_per_alpha": fills,
                   "users_needing_fill_alpha_0.8": sum(
                       1 for uid, d in pools.items() if len(d["pool"]) < max(KS))}


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
    args = parser.parse_args()
    started = time.time()

    mode = recommender.graph_mode()
    if mode != "eval":
        raise SystemExit(f"The graph is in '{mode}' mode. Evaluation needs train ratings only (D19):\n"
                         "  python scripts/05_build_graph.py --mode eval --wipe")

    print("[1/4] Catalogue and relevant items (D27)")
    train, test = pd.read_csv(config.TRAIN_PATH), pd.read_csv(config.TEST_PATH)
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

    print("[3/4] Graph recommender (one pool per user, re-ranked for each α)")
    t = time.time()
    by_alpha, graph_info = graph_lists(users, args.workers)
    graph_seconds = time.time() - t
    lists["Graph pure (α = 1.0)"] = by_alpha[1.0]
    lists["Graph hybrid (α = 0.8)"] = by_alpha[0.8]
    leaks = sum(bool(set(lst) & seen.get(u, set())) for lst in lists.values() for u, lst in lst.items())
    left_catalog = sum(bool(set(l_) - catalog) for lst in lists.values() for l_ in lst.values())
    print(f"  {graph_seconds:.1f} s; lists containing a train item: {leaks}; lists with a film outside the catalogue: "
          f"{left_catalog}; users with a pool under 10: {graph_info['users_with_pool_under_10']}")

    print("[4/4] Metrics")
    rows = [{"method": name, "k": k, **ev.evaluate_lists(lst, relevant, k, catalog, popularity)}
            for k in KS for name, lst in lists.items()]
    results = pd.DataFrame(rows)
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
