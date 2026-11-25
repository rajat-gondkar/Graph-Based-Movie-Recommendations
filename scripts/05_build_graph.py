"""Phase 3: build the Neo4j knowledge graph (spec section 7).

Usage:
    python scripts/05_build_graph.py --mode full --wipe     # demo graph: all ratings
    python scripts/05_build_graph.py --mode eval --wipe     # evaluation graph: train ratings only
    python scripts/05_build_graph.py --mode full            # re-run without --wipe: idempotent, same counts
Switching between modes needs --wipe (D19). Neo4j must be running (docker compose up -d).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config and src importable

import config  # noqa: E402
from src import graph_builder as gb  # noqa: E402

STATS_PATH = config.RESULTS_DIR / "graph_stats.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the movie knowledge graph in Neo4j.")
    parser.add_argument("--mode", choices=gb.MODES, default="full",
                        help="val = earlier 80%% of train (tuning); eval = train.csv only; full = all ratings "
                             "(default %(default)s)")
    parser.add_argument("--wipe", action="store_true", help="delete everything in the database first")
    args = parser.parse_args()

    start = time.time()
    print(f"[1/6] Reading processed files (mode: {args.mode})")
    data = gb.load_inputs(args.mode)
    print(f"  {len(data['movies']):,} movies, {data['themes']['theme'].nunique()} themes "
          f"({len(data['themes']):,} movie-theme pairs), {data['genres']['genre'].nunique()} genres "
          f"({len(data['genres']):,} pairs), {data['ratings']['userId'].nunique():,} users, "
          f"{len(data['ratings']):,} ratings; {len(data['similar']):,} SIMILAR edges "
          f"({config.SIMILAR_NEIGHBOURS} neighbours per movie)")

    driver = gb.get_driver()
    try:
        driver.verify_connectivity()
        current = gb.existing_mode(driver)
        if current and current != args.mode and not args.wipe:
            raise SystemExit(f"The graph is in '{current}' mode. Re-run with --wipe to rebuild it as '{args.mode}'.")
        if args.wipe:
            print("[2/6] Wiping the database")
            gb.wipe(driver)
        else:
            print("[2/6] Keeping existing data (MERGE makes the load idempotent)")
        print("[3/6] Creating constraints")
        gb.create_constraints(driver)
        print("[4/6] Loading movies, genres, themes, users and ratings in batches of "
              f"{gb.BATCH_SIZE}")
        load_start = time.time()
        gb.load_graph(driver, data)
        load_seconds = time.time() - load_start
        print("[5/6] Computing ratingCount, avgRating, movieCount and idf")
        total = gb.compute_stats(driver)
        gb.write_meta(driver, args.mode)
        print("[6/6] Checking the graph against the input files")
        checks = gb.sanity_check(driver, data)
        c = gb.counts(driver)
        extremes = gb.theme_idf_extremes(driver)
    finally:
        driver.close()

    print("\nNodes:")
    for label, n in c["nodes"].items():
        print(f"  {label:<8} {n:>8,}")
    print("Relationships:")
    for rel, n in c["relationships"].items():
        print(f"  {rel:<10} {n:>8,}")
    print("\nChecks:")
    for text, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {text}")
    print("\nMost common themes (lowest idf): " + ", ".join(f"{n} ({m} movies, idf {i:.2f})" for n, m, i in extremes["most_common"]))
    print("Rarest themes (highest idf):     " + ", ".join(f"{n} ({m}, idf {i:.2f})" for n, m, i in extremes["rarest"]))

    seconds = time.time() - start
    stats = {"mode": args.mode, "total_movies": total, "nodes": c["nodes"], "relationships": c["relationships"],
             "checks_passed": all(ok for _, ok in checks), "load_seconds": round(load_seconds, 1),
             "total_seconds": round(seconds, 1),
             "most_common_themes": [list(x) for x in extremes["most_common"]],
             "rarest_themes": [list(x) for x in extremes["rarest"]]}
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"\nWrote {STATS_PATH.relative_to(config.BASE_DIR)}; load {load_seconds:.1f} s, total {seconds:.1f} s")
    print("\nTry these in the Neo4j browser (http://localhost:7474):")
    print("  MATCH (m:Movie)-[:HAS_THEME]->(t:Theme) RETURN m, t LIMIT 50")
    print("  MATCH (m:Movie {title: 'The Matrix'})-[r]->(x) RETURN m, r, x")
    print("  MATCH (t:Theme) RETURN t.name, t.movieCount, t.idf ORDER BY t.idf DESC LIMIT 10")
    if not all(ok for _, ok in checks):
        raise SystemExit("Some checks failed; see above.")


if __name__ == "__main__":
    main()
