# Deterministic dependency install. Used by CI and by local development so both
# resolve the same tree.
#
#   bash tools/install_deps.sh            # runtime only
#   bash tools/install_deps.sh --dev      # runtime + dev/test tooling
#
# Why this exists instead of `pip install -r requirements.txt`:
#
# Three packages in the graph declare three *different* OpenCV distributions,
# and every one of them unpacks into the same `cv2/` package directory:
#
#   scenedetect        -> opencv-python           (unpinned)
#   rapidocr_onnxruntime -> opencv-python        (unpinned, >=4.5.1.48)
#   mediapipe          -> opencv-contrib-python   (unpinned)
#
# pip installs all of them, last writer wins, so the effective `cv2` build
# depends on install order. We have seen all three land at once in CI:
#
#   opencv-contrib-python-5.0.0.93
#   opencv-python-5.0.0.93
#   opencv-python-headless-4.14.0.94
#
# ...which also silently violates requirements.txt's own `<5.0.0.0` pin.
#
# Only the headless build is actually needed: scenedetect and rapidocr `import cv2`,
# they do not require the GUI build's symbols. So we install the normal way and
# then remove the two non-headless variants, leaving one deterministic `cv2`.

set -euo pipefail

DEV=0
if [ "${1:-}" = "--dev" ]; then
  DEV=1
fi

# Keep this in sync with requirements.txt. The lower bound is deliberately loose;
# the point is to *choose* one distribution, not to freeze a patch version.
HEADLESS_SPEC="opencv-python-headless>=4.8.0.76,<5.0.0.0"

python -m pip install --upgrade pip

# System libraries OpenCV's headless build and ffmpeg need.
if command -v apt-get >/dev/null 2>&1; then
  if [ "$(id -u)" = "0" ]; then SUDO=""; else SUDO="sudo"; fi
  $SUDO apt-get update -qq
  $SUDO apt-get install -y --no-install-recommends ffmpeg libglib2.0-0
fi

python -m pip install -r requirements.txt

if [ "$DEV" = "1" ]; then
  python -m pip install -r requirements-dev.txt
fi

# Collapse the OpenCV distributions down to the single headless build.
python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless >/dev/null 2>&1 || true
python -m pip install --no-deps --force-reinstall "$HEADLESS_SPEC"

# Fail loudly if more than one OpenCV distribution survived.
INSTALLED=$(python -m pip list --format=freeze 2>/dev/null | grep -iE '^opencv' || true)
COUNT=$(printf '%s\n' "$INSTALLED" | grep -c . || true)
if [ "$COUNT" -ne 1 ]; then
  echo "ERROR: expected exactly 1 OpenCV distribution, found $COUNT:" >&2
  printf '%s\n' "$INSTALLED" >&2
  exit 1
fi
echo "[*] OpenCV resolved to a single distribution: $INSTALLED"

python - <<'PY'
import cv2
print(f"[*] cv2 {cv2.__version__} imported from {cv2.__file__}")
PY
