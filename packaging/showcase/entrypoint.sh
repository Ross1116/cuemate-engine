#!/bin/sh
set -eu

PORT="${PORT:-8080}"
export GO_API_ADDR="0.0.0.0:${PORT}"
export WEB_DIST_DIR="${WEB_DIST_DIR:-/app/web/dist}"
export SCORING_GRPC_ADDR="${SCORING_GRPC_ADDR:-127.0.0.1:47834}"
export CUEMATE_SHOWCASE_MODE="${CUEMATE_SHOWCASE_MODE:-1}"
export DATABASE_URL="${DATABASE_URL:-sqlite:file:/app/data/cuemate-showcase.db?mode=ro&immutable=1}"

scoring_host="${SCORING_GRPC_ADDR%:*}"
scoring_port="${SCORING_GRPC_ADDR##*:}"
if [ "$scoring_host" = "$SCORING_GRPC_ADDR" ]; then
    scoring_host="127.0.0.1"
    scoring_port="$SCORING_GRPC_ADDR"
fi
if [ -z "$scoring_host" ]; then
    scoring_host="127.0.0.1"
fi
case "$scoring_port" in
    ''|*[!0-9]*) scoring_port="47834" ;;
esac

python -m cuemate_analysis serve-scoring --host "$scoring_host" --port "$scoring_port" &
scorer_pid="$!"

cleanup() {
    kill "$scorer_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

exec /app/apiserver
