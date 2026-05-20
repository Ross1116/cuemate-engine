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

wait_for_scorer() {
    python - "$scoring_host" "$scoring_port" <<'PY'
import sys

import grpc

from cuemate_analysis.scoring_service import load_scoring_proto_modules

try:
    host = sys.argv[1]
    port = sys.argv[2]
    pb2, pb2_grpc = load_scoring_proto_modules()
    with grpc.insecure_channel(f"{host}:{port}") as channel:
        grpc.channel_ready_future(channel).result(timeout=1.0)
        stub = pb2_grpc.ScoringServiceStub(channel)
        stub.GetScoringMetadata(pb2.GetScoringMetadataRequest(), timeout=1.0)
except Exception:
    sys.exit(1)
PY
}

for attempt in $(seq 1 30); do
    if wait_for_scorer; then
        break
    fi
    if ! kill -0 "$scorer_pid" 2>/dev/null; then
        echo "Scoring service exited before it became ready." >&2
        exit 1
    fi
    if [ "$attempt" -eq 30 ]; then
        echo "Scoring service did not become ready on ${scoring_host}:${scoring_port}." >&2
        exit 1
    fi
    sleep 1
done

cleanup() {
    kill "$scorer_pid" 2>/dev/null || true
    kill "${api_pid:-}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

/app/apiserver &
api_pid="$!"

(
    wait "$scorer_pid"
    echo "Scoring service exited; stopping API." >&2
    kill "$api_pid" 2>/dev/null || true
) &

wait "$api_pid"
