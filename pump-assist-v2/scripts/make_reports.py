#!/usr/bin/env python3
"""U02: write one static HTML report per pump into results/reports/ (open index.html, no server needed).

  python scripts/make_reports.py                 # train + validation pumps (default)
  python scripts/make_reports.py --splits train validation test
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assist.report import write_reports
from assist.splits import pumps_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["v2", "b4"], default=None, help="override configs/pipeline.yaml")
    ap.add_argument("--splits", nargs="+", default=["train", "validation"], choices=["train", "validation", "test"])
    a = ap.parse_args()
    ids = [p for s in a.splits for p in pumps_in(s)]
    out = ROOT / "results" / ("reports_b4" if a.engine == "b4" else "reports")
    paths = write_reports(ids, tuple(a.splits), out_dir=out, engine=a.engine)
    print(f"{len(paths)} reports in {paths[0].parent}")


if __name__ == "__main__":
    main()
