#!/bin/sh
# TrainKit installer for macOS and Linux.
#   curl -LsSf https://raw.githubusercontent.com/Miracle197/model-toybox/main/bootstrap/install.sh | sh
# Install from a local checkout instead:  TRAINKIT_SOURCE=/path/to/trainkit sh bootstrap/install.sh
set -eu

TRAINKIT_SOURCE="${TRAINKIT_SOURCE:-git+https://github.com/Miracle197/model-toybox}"

say() { printf '\033[1;36m==>\033[0m %s\n' "$1"; }

if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv (fast Python installer, also provides Python itself)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

say "Installing TrainKit from $TRAINKIT_SOURCE …"
uv tool install --python 3.12 --force "$TRAINKIT_SOURCE"
uv tool update-shell >/dev/null 2>&1 || true

say "Checking this computer…"
trainkit doctor || true

cat <<'MSG'

TrainKit is installed. Open a new terminal, then:
  trainkit setup          # pick a model; installs PyTorch for your GPU, weights and a test dataset
  trainkit train …        # train with a live view and web dashboard
  trainkit play           # try models on your camera, videos or images
MSG
