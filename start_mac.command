#!/bin/bash
# Double-click to set up (first time only) and open the notebooks or the app.
# Needs an internet connection the first time. Nothing is installed system-wide:
# uv goes in ~/.local/bin, and Python plus the packages go in this folder (.venv).
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv, the tool that sets up Python for this project. One time only..."
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        echo
        echo 'Could not install uv automatically. See START_HERE.md, section "If something goes wrong".'
        read -r -p "Press Enter to close this window."
        exit 1
    fi
fi

echo "Checking the project setup. The first time this downloads Python and the packages, about 2-5 minutes..."
if ! uv run --locked --no-dev python scripts/launch.py "$@"; then
    echo
    echo 'Something went wrong. See the messages above and START_HERE.md, section "If something goes wrong".'
    read -r -p "Press Enter to close this window."
fi
