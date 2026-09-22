#!/usr/bin/env sh

# If you cannot download the model weights from the official site, use a mirror.
# Just remove the comment of the following line.
# export HF_ENDPOINT=https://hf-mirror.com

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# OmniVoice needs torch, which the main app venv deliberately does not install.
# Point OMNIVOICE_PYTHON at the interpreter of the environment you installed
# services/omnivoice_server/requirements.txt into. See that folder's README.md.
#
# The .venv/Scripts/python.exe branch matters on Git Bash / MSYS under Windows:
# that layout has no .venv/bin/python, and a bare python3 there is usually the
# Windows Store stub, which fails in ways that have nothing to do with this app.
if [ -n "$OMNIVOICE_PYTHON" ]; then
  PY="$OMNIVOICE_PYTHON"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
elif [ -x "$ROOT/.venv/Scripts/python.exe" ]; then
  PY="$ROOT/.venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "***** No Python interpreter found. Run 'uv sync --frozen' first. *****"
  exit 1
fi

echo "***** Starting OmniVoice TTS service on http://127.0.0.1:8890 *****"
"$PY" -m services.omnivoice_server "$@"
