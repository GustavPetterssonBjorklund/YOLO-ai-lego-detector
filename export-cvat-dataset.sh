#!/usr/bin/env bash
set -Eeuo pipefail

# Override any of these values through environment variables when needed.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd)"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$PROJECT_ROOT/artifacts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ARTIFACTS_ROOT/exports}"
CVAT_URL="${CVAT_URL:-https://cvat.spetsen.se}"
CVAT_USERNAME="${CVAT_USERNAME:-gustav.pettersson.bjorklund}"
CVAT_ORG="${CVAT_ORG:-Hitachigym}"
CVAT_PROJECT_ID="${CVAT_PROJECT_ID:-2}"
CVAT_FORMAT="${CVAT_FORMAT:-Ultralytics YOLO Detection 1.0}"

# Make all relative overrides deterministic and repository-relative.
cd "$PROJECT_ROOT"

for command_name in cvat-cli unzip; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'Error: required command not found: %s\n' "$command_name" >&2
        exit 1
    fi
done

# A CVAT access token is preferred. If none is available, prompt once for the
# CVAT password and let cvat-cli read it from PASS.
if [[ -z "${CVAT_ACCESS_TOKEN:-}" ]]; then
    read -rsp "CVAT password for ${CVAT_USERNAME}: " PASS
    export PASS
    printf '\n'
    trap 'unset PASS' EXIT
fi

timestamp="$(date -u +'%Y%m%dT%H%M%SZ')"
archive_dir="${OUTPUT_ROOT}/archives"
datasets_dir="${OUTPUT_ROOT}/datasets"
dataset_dir="${datasets_dir}/dataset-${timestamp}"
archive_path="${archive_dir}/lego-dataset-${timestamp}.zip"

mkdir -p "$archive_dir" "$datasets_dir" "$dataset_dir"

printf 'Exporting CVAT project %s...\n' "$CVAT_PROJECT_ID"
cvat-cli \
    --server-host "$CVAT_URL" \
    --auth "$CVAT_USERNAME" \
    --org "$CVAT_ORG" \
    project export-dataset \
    --format "$CVAT_FORMAT" \
    --with-images yes \
    "$CVAT_PROJECT_ID" \
    "$archive_path"

unzip -tq "$archive_path" >/dev/null
unzip -q "$archive_path" -d "$dataset_dir"

# Point "latest" at the newest extracted snapshot without deleting older ones.
ln -sfn "datasets/$(basename "$dataset_dir")" "${OUTPUT_ROOT}/latest"

printf '\nExport complete.\n'
printf 'Archive: %s\n' "$archive_path"
printf 'Dataset: %s\n' "$dataset_dir"
printf 'Latest:  %s/latest\n' "$OUTPUT_ROOT"
