#!/bin/bash
# HE FinPlanner — Mac launcher
# Double-click in Finder to start.
# First run: installs Python environment automatically (requires uv).
# Subsequent runs: starts in ~3 seconds.
#
# Prerequisite (one-off, in Terminal):
#   brew install uv
#
# If double-click is blocked by Gatekeeper:
#   right-click → Open  (first time only)

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# ── Check uv ──────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    osascript <<'APPLESCRIPT'
display dialog "uv is not installed.

Install it once in Terminal:

    brew install uv

Then double-click this file again." with title "HE FinPlanner" buttons {"OK"} default button "OK" with icon caution
APPLESCRIPT
    exit 1
fi

# ── Create / update Python environment (idempotent) ───────────────────────────
if [ ! -d ".venv" ]; then
    echo "First run — creating Python environment (takes ~30 s) …"
    uv venv .venv
fi

echo "Checking dependencies …"
uv pip install --quiet -r DEMO/requirements.txt

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
echo "Starting HE FinPlanner …"
echo "Browser will open at http://localhost:8501"
echo "Close this window (or press Ctrl-C) to stop the app."
echo ""

# Open the browser once the server is ready
(
    for i in $(seq 1 20); do
        curl -sf http://localhost:8501/_stcore/health &>/dev/null && break
        sleep 1
    done
    open "http://localhost:8501"
) &

uv run streamlit run DEMO/app.py \
    --browser.serverAddress localhost \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.port 8501
