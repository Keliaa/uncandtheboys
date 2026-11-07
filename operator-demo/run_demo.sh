#!/usr/bin/env bash
# One-command launcher for the B4 operator demo.
#   ./run_demo.sh            start the presentation (creates .venv on first run; needs network only then)
#   ./run_demo.sh --prepare  regenerate replay bundles from the shared data first (runs B4 offline, ~20 s)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if [ ! -x .venv/bin/python ]; then
  echo "Creating .venv and installing requirements (one-time, needs network) ..."
  "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi

if [ "${1:-}" = "--prepare" ] || [ ! -f bundles/manifest.json ]; then
  .venv/bin/python prepare_demo.py
fi

echo "Opening http://localhost:8501  (Ctrl+C to stop)"
( sleep 2; command -v open >/dev/null && open "http://localhost:8501" ) >/dev/null 2>&1 &
exec .venv/bin/streamlit run app.py --server.headless true --browser.gatherUsageStats false
