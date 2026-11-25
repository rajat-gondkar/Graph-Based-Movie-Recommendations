"""Phase 6B: explanation quality (spec 11.2, D29, D30).

For 50 random evaluation users (seed 42), take the top-1 hybrid recommendation, write its explanation with the LLM
and with the template, and check both: faithfulness (every title and theme mentioned is in the evidence), rule
compliance (names a liked title and an evidence theme), length, and LLM fallbacks. Path coverage comes from the
accuracy run (share of top-10 items that have a theme path).

Usage (eval graph, after 06):
    python scripts/07_evaluate_explanations.py            # asks before the LLM calls
    python scripts/07_evaluate_explanations.py --yes
Outputs: results/explanation_eval.csv, results/explanation_stats.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import explainer, recommender  # noqa: E402

SAMPLE = 50


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM and template explanations.")
    parser.add_argument("--yes", action="store_true", help="don't ask before the LLM calls")
    parser.add_argument("--sample", type=int, default=SAMPLE)
    args = parser.parse_args()

    mode = recommender.graph_mode()
    if mode != "eval":
        raise SystemExit(f"The graph is in '{mode}' mode; run this on the eval graph, after 06_evaluate_accuracy.py.")
    users = sorted(recommender.user_ids())
    sample = sorted(random.Random(config.RANDOM_SEED).sample(users, args.sample))
    print(f"[1/3] Sample: {len(sample)} users (seed {config.RANDOM_SEED}), top-1 hybrid recommendation each "
          f"(α = {config.ALPHA}); graph mode: {mode}")

    recs = {u: recommender.recommend(u, k=1, alpha=config.ALPHA)[0] for u in sample}
    if not args.yes:
        answer = input(f"  Up to {len(sample)} LLM calls (cached ones are free). Continue? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            raise SystemExit("Cancelled. No LLM calls were made.")

    print("[2/3] Explanations (LLM and template)")
    started = time.time()
    rows, calls, fallbacks = [], 0, 0
    titles = [r["t"] for r in recommender._read("MATCH (m:Movie) RETURN m.title AS t")]
    themes = [r["t"] for r in recommender._read("MATCH (t:Theme) RETURN t.name AS t")]
    genres = [r["g"] for r in recommender._read("MATCH (g:Genre) RETURN g.name AS g")]
    finder = ev.MentionFinder(titles, themes, genres)
    for uid in sample:
        rec = recs[uid]
        llm = explainer.explain_many(uid, [rec], mode="llm")[0]
        stats = explainer.last_batch_stats()
        calls += stats["llm_calls"]
        fallbacks += stats["fallbacks"]
        tpl = explainer.explain(uid, rec["movieId"], mode="template")
        evidence = llm["evidence"]
        row = {"userId": uid, "movieId": rec["movieId"], "recommended": rec["displayTitle"],
               "filled": rec["filled"], "has_path": bool(evidence["evidence"]),
               "evidence_themes": "; ".join(e["theme"] for e in evidence["evidence"]),
               "llm_source": llm["source"]}
        for name, text in (("llm", llm["text"]), ("template", tpl["text"])):
            check = ev.check_faithfulness(text, evidence, finder)
            row.update({
                f"{name}_text": text,
                f"{name}_words": len(text.split()),
                f"{name}_faithful": check["faithful"],
                f"{name}_genres_ok": check["genres_ok"],
                f"{name}_mentions_liked_title": check["mentions_liked_title"],
                f"{name}_mentions_evidence_theme": check["mentions_evidence_theme"],
                f"{name}_unsupported": "; ".join(check["unsupported_titles"] + check["unsupported_themes"]),
                f"{name}_unsupported_genres": "; ".join(check["unsupported_genres"]),
            })
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "explanation_eval.csv", index=False)

    print("[3/3] Summary")
    acc = json.loads((config.RESULTS_DIR / "accuracy_stats.json").read_text())
    fills = acc["fill_ins_per_alpha"].get("0.8", acc["fill_ins_per_alpha"].get(0.8))
    total_items = acc["users"] * 10
    summary = {"sample": len(df), "graph_mode": mode, "llm_calls": calls, "llm_fallbacks": fallbacks,
               "seconds": round(time.time() - started, 1),
               "path_coverage_sample": float(df["has_path"].mean()),
               "path_coverage_all_top10_alpha_0.8": round(1 - fills / total_items, 6)}
    for name in ("llm", "template"):
        summary[name] = {
            "faithful": float(df[f"{name}_faithful"].mean()),
            "genres_ok": float(df[f"{name}_genres_ok"].mean()),
            "mentions_liked_title": float(df[f"{name}_mentions_liked_title"].mean()),
            "mentions_evidence_theme": float(df[f"{name}_mentions_evidence_theme"].mean()),
            "both_rules": float((df[f"{name}_mentions_liked_title"] & df[f"{name}_mentions_evidence_theme"]).mean()),
            "mean_words": round(float(df[f"{name}_words"].mean()), 1),
        }
    (config.RESULTS_DIR / "explanation_stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name in ("llm", "template"):
        s = summary[name]
        print(f"  {name:>8}: faithful {s['faithful']:.0%} · genres ok {s['genres_ok']:.0%} · names a liked title "
              f"{s['mentions_liked_title']:.0%} · names an evidence theme {s['mentions_evidence_theme']:.0%} · "
              f"mean {s['mean_words']} words")
    print(f"  LLM calls {calls}, fallbacks {fallbacks}; path coverage: sample {summary['path_coverage_sample']:.0%}, "
          f"all top-10 lists {summary['path_coverage_all_top10_alpha_0.8']:.2%}")
    bad = df[~df["llm_faithful"] | ~df["llm_genres_ok"]]
    if len(bad):
        print("  LLM explanations flagged:")
        for _, r in bad.iterrows():
            print(f"    user {r.userId} -> {r.recommended}: unsupported [{r.llm_unsupported}] genres "
                  f"[{r.llm_unsupported_genres}]\n      {r.llm_text}")
    print("  Wrote results/explanation_eval.csv and results/explanation_stats.json")


if __name__ == "__main__":
    main()
