#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd)"
PYTHON="${PYTHON:-python3}"
YOLO_VENV="${YOLO_VENV:-$PROJECT_ROOT/.venv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$PROJECT_ROOT/requirements.txt}"

cd "$PROJECT_ROOT"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    printf 'Error: Python executable not found: %s\n' "$PYTHON" >&2
    exit 1
fi

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    printf 'Error: requirements file not found: %s\n' "$REQUIREMENTS_FILE" >&2
    exit 1
fi

if [[ ! -x "$YOLO_VENV/bin/python" ]]; then
    printf 'Creating virtual environment at %s...\n' "$YOLO_VENV"
    "$PYTHON" -m venv "$YOLO_VENV"
fi

printf 'Installing Python dependencies...\n'
"$YOLO_VENV/bin/python" -m pip install --upgrade pip
"$YOLO_VENV/bin/python" -m pip install --requirement "$REQUIREMENTS_FILE"

printf '\nInstallation complete.\n'
printf 'Virtual environment: %s\n' "$YOLO_VENV"
printf 'Activate it with: source %q/bin/activate\n' "$YOLO_VENV"

if ! command -v unzip >/dev/null 2>&1; then
    printf 'Warning: unzip is required for CVAT exports but is not installed.\n' >&2
fi
