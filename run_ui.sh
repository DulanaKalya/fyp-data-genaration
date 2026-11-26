#!/usr/bin/env bash
# Start the UI: FastAPI backend + built React app on http://127.0.0.1:8765
# (port 8000 is already used by another program on this machine)
set -e
cd "$(dirname "$0")"
if [ ! -f ui/dist/index.html ] || [ "${1:-}" = "--build" ]; then
  (cd ui && npm install --silent && npm run build)
fi
echo "UI: http://127.0.0.1:8765"
exec python3 -m uvicorn api.server:app --host 127.0.0.1 --port 8765
