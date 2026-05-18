# CueMate Client

Responsive local-first PWA for the CueMate engine.

## Development

```powershell
npm install
npm run dev
```

The Vite dev server proxies `/api/*` to `http://127.0.0.1:8080`.

## Production Build

```powershell
npm run build
go run ./go/cmd/apiserver
```

The Go API serves `web/dist` by default. Override with `WEB_DIST_DIR` if needed.
