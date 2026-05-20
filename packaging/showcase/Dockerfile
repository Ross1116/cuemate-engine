FROM node:22.21.0-bookworm-slim AS web-build
WORKDIR /src/web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM golang:1.25-bookworm-slim AS go-build
WORKDIR /src/go
COPY go/go.mod go/go.sum ./
RUN go mod download
COPY go/ ./
RUN go build -o /out/apiserver ./cmd/apiserver

FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    GO_API_ADDR=0.0.0.0:8080 \
    WEB_DIST_DIR=/app/web/dist \
    SCORING_GRPC_ADDR=127.0.0.1:47834 \
    CUEMATE_SHOWCASE_MODE=1 \
    DATABASE_URL=sqlite:file:/app/data/cuemate-showcase.db?mode=ro&immutable=1

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system cuemate \
    && useradd --system --gid cuemate --home-dir /app --shell /usr/sbin/nologin cuemate

COPY python/ /app/python/
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir /app/python

COPY config/ /app/config/
COPY db/schema.sql /app/db/schema.sql
COPY --from=web-build /src/web/dist/ /app/web/dist/
COPY --from=go-build /out/apiserver /app/apiserver
COPY dist/showcase/cuemate-showcase.db /app/data/cuemate-showcase.db
COPY packaging/showcase/entrypoint.sh /app/entrypoint.sh

RUN chown -R cuemate:cuemate /app \
    && chmod +x /app/entrypoint.sh \
    && chmod -R u+rwX,go+rX /app /app/data /app/web/dist
EXPOSE 8080
USER cuemate
ENTRYPOINT ["/app/entrypoint.sh"]
