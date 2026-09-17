#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd)"
YOLO_VENV="${YOLO_VENV:-$PROJECT_ROOT/.venv}"
INSTALL_SCRIPT="${INSTALL_SCRIPT:-$PROJECT_ROOT/install.sh}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

cd "$PROJECT_ROOT"

if [[ -z "${MODEL:-}" ]]; then
    printf 'Error: MODEL must point to a trained .pt weights file.\n' >&2
    printf 'Example: MODEL=artifacts/runs/my-run/weights/best.pt %q\n' "$0" >&2
    exit 1
fi

if [[ ! -x "$YOLO_VENV/bin/uvicorn" ]]; then
    YOLO_VENV="$YOLO_VENV" "$INSTALL_SCRIPT"
fi

printf 'Serving the annotated stream at http://%s:%s\n' "$HOST" "$PORT"
printf 'Video source: %s\n' "${VIDEO_SOURCE:-0}"
exec "$YOLO_VENV/bin/uvicorn" scripts.stream_server:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1
