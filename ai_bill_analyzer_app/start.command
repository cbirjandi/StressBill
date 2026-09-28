#!/bin/bash
# AI Bill Analyzer launcher: double-click this file in Finder.
# First run: installs what it needs (a few minutes) and builds the AGORA index (about a minute).
# After that: opens the dashboard in a few seconds.

cd "$(dirname "$0")" || exit 1

pause_and_exit() {
    echo
    read -r -p "Press Enter to close this window."
    exit 1
}

# 1. Find Python 3.11 or newer (the Mac's built-in Python 3.9 is too old)
PY=""
for candidate in python3.12 python3.13 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
        PY="$candidate"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "Python 3.11 or newer is needed. Opening python.org:"
    echo "download the macOS installer for Python 3.12, run it, then double-click start.command again."
    open "https://www.python.org/downloads/"
    pause_and_exit
fi

# 2. Private environment with the app's packages (first run only)
if [ ! -f .venv/.installed ]; then
    echo "First-time setup: installing packages (a few minutes)..."
    rm -rf .venv
    "$PY" -m venv .venv || pause_and_exit
    .venv/bin/python -m pip install --upgrade pip || pause_and_exit
    .venv/bin/python -m pip install -r requirements.txt || { echo "Package installation failed."; pause_and_exit; }
    touch .venv/.installed
fi

# 3. Open the dashboard. The first start also builds the AGORA index (about a minute).
#    Close this window or press Ctrl+C to stop it.
echo "Starting the dashboard. It opens in your browser; keep this window open while you use it."
.venv/bin/python -m streamlit run app.py --server.fileWatcherType none
