#!/bin/sh
set -eu

PORT="${PORT:-8080}"
export GO_API_ADDR="0.0.0.0:${PORT}"
export WEB_DIST_DIR="${WEB_DIST_DIR:-/app/web/dist}"
export SCORING_GRPC_ADDR="${SCORING_GRPC_ADDR:-127.0.0.1:47834}"
export CUEMATE_SHOWCASE_MODE="${CUEMATE_SHOWCASE_MODE:-1}"
export DATABASE_URL="${DATABASE_URL:-sqlite:file:/app/data/cuemate-showcase.db?mode=ro&immutable=1}"

python -m cuemate_analysis serve-scoring --host 127.0.0.1 --port 47834 &
scorer_pid="$!"

cleanup() {
    kill "$scorer_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

exec /app/apiserver
