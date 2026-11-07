#!/usr/bin/env python3
"""E04: decision tree (depth 3) for the machine fault type, compared with E01 (rules) and E02 (naive Bayes).

Same training rows as E02 (train events only). Alerts on train pumps use leave-one-pump-out models for E02 and E04.
Writes results/e04_tree.png, results/e04_tree_rules.txt (rules and comparison with the symptom table), results/e04_trainval_predictions.csv.
Validation is scored once; test pumps are not used."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT.parent / "baseline-reproduction" / "scripts"))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree
from assist import _ref  # noqa: F401
from assist.config import load_machine
from assist.evidence.bayes import GaussianNB
from assist.evidence.tree import FaultTree
from assist.pipeline import predictions_frame, run_pipeline
from assist.splits import pumps_in
from retrofit_sim.baselines import CH
from tune_e02 import training_rows
import eval as ev


def main():
    tr, va = pumps_in("train"), pumps_in("validation")
    events = pd.read_csv(ROOT.parent / "data" / "full" / "events_train.csv")
    det_tr, _ = run_pipeline(tr, ("train",), use_e02=False, with_what_if=False)
    rows, labels, pumps = training_rows(det_tr, events)
    channels = list(CH.values())
    print("training rows", len(rows), pd.Series(labels).value_counts().to_dict())

    def fit(make, exclude=None):
        keep = [i for i, p in enumerate(pumps) if p != exclude]
        return make().fit([rows[i] for i in keep], [labels[i] for i in keep])

    models = {"E02 naive Bayes": lambda: GaussianNB(channels), "E04 decision tree": lambda: FaultTree(channels, 3)}
    full = {n: fit(m) for n, m in models.items()}
    lopo = {n: {p: fit(m, p) for p in tr} for n, m in models.items()}
    results, preds = {}, {}
    for name in ("E01 rules", "E02 naive Bayes", "E04 decision tree"):
        if name == "E01 rules":
            c_tr = c_va = None
            use = False
        else:
            c_tr = (lambda n: lambda a, z, t: lopo[n][a].predict_proba(z))(name)
            c_va = (lambda n: lambda a, z, t: full[n].predict_proba(z))(name)
            use = True
        ptr = predictions_frame(run_pipeline(tr, ("train",), use_e02=use, classifier=c_tr, with_what_if=False)[1])
        pva = predictions_frame(run_pipeline(va, ("validation",), use_e02=use, classifier=c_va, with_what_if=False)[1])
        p = pd.concat([ptr, pva], ignore_index=True)
        r = ev.score_splits(p)
        preds[name] = p
        results[name] = r
        print(f"{name}: train {r['train']['composite_score']:.1f} type recall {r['train']['fault_type_recall']:.2f} class recall {r['train']['class_recall']:.2f} | "
              f"validation {r['validation']['composite_score']:.1f} type recall {r['validation']['fault_type_recall']:.2f} class recall {r['validation']['class_recall']:.2f}", flush=True)
    preds["E04 decision tree"].to_csv(ROOT / "results" / "e04_trainval_predictions.csv", index=False)

    tree = full["E04 decision tree"]
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_tree(tree.tree, feature_names=[c.replace("_", " ") for c in channels], class_names=list(tree.tree.classes_), filled=True,
              impurity=False, proportion=False, fontsize=8, ax=ax)
    ax.set_title("E04 decision tree (depth 3) for the machine fault type, trained on train events")
    fig.tight_layout(); fig.savefig(ROOT / "results" / "e04_tree.png", dpi=130)

    machine = load_machine()
    sym = {t: {e["channel"] for e in spec["evidence"]} for t, spec in machine["symptoms"].items()}
    lines = ["# E04 tree rules (trained on the same rows as E02)\n", "```", tree.rules(), "```\n", "## Comparison with the symptom table\n",
             "| leaf (path) | predicted type | rows | channels on the path | in the symptom table of that type |", "|---|---|---|---|---|"]
    for lf in tree.leaves():
        path = " AND ".join(f"{c.replace('_', ' ')} {op} {th:.1f}" for c, op, th in lf["conditions"])
        inn = sorted(c for c in lf["channels"] if c in sym.get(lf["fault_type"], set()))
        out = sorted(c for c in lf["channels"] if c not in sym.get(lf["fault_type"], set()))
        lines.append(f"| {path} | {lf['fault_type']} | {lf['n']} | {', '.join(sorted(lf['channels']))} | in: {', '.join(inn) or '-'}; not in: {', '.join(out) or '-'} |")
    (ROOT / "results" / "e04_tree_rules.txt").write_text("\n".join(lines) + "\n")
    print(tree.rules())


if __name__ == "__main__":
    main()
