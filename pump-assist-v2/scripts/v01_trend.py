#!/usr/bin/env python3
"""V01: snapshot statistics per pump, trend table and figure. Reads only the vibration snapshots (no labels, no test pumps)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from assist.detect.vibration import snapshot_stats
from assist.splits import pumps_in

BEARING = {"PUMP-02": "minor", "PUMP-04": "moderate", "PUMP-07": "minor", "PUMP-08": "minor"}   # from events_train/validation


def main():
    ids = pumps_in("train") + pumps_in("validation")
    s = snapshot_stats(ids, splits=("train", "validation"))
    s.drop(columns=["snapshot_id"]).to_csv(ROOT / "results" / "v01_snapshot_stats.csv", index=False)
    rows = []
    for aid, g in s.groupby("asset_id"):
        rows.append({"asset_id": aid, "bearing_wear_event": BEARING.get(aid, ""), "snapshots": len(g),
                     "max_rms_ratio": round(g.rms_ratio.max(), 2), "max_kurt_delta": round(g.kurt_delta.max(), 2),
                     "max_crest_delta": round(g.crest_delta.max(), 2), "snapshots_flagged": int(g.changed.sum())})
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "results" / "v01_trends.csv", index=False)
    print(t.to_string(index=False))
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=False)
    for ax, col, lab in zip(axes, ["rms_ratio", "kurt_delta", "crest_delta"], ["RMS / start value", "Kurtosis change", "Crest factor change"]):
        for aid, g in s.groupby("asset_id"):
            hit = aid in BEARING
            ax.plot(g.timestamp_utc, g[col], lw=1.6 if hit else 0.7, alpha=1 if hit else 0.5, label=f"{aid} (bearing wear)" if hit else None)
        ax.set_ylabel(lab, fontsize=8); ax.grid(alpha=0.2); ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_title("Vibration snapshot statistics vs the pump's own first snapshots (thin: other pumps)", fontsize=9)
    fig.tight_layout(); fig.savefig(ROOT / "results" / "v01_trends.png", dpi=130)


if __name__ == "__main__":
    main()
