#!/usr/bin/env bash
# bash run.sh local    run without Docker (recommended)  -> http://localhost:8000
# bash run.sh docker   run with Docker (builds from the repo root)
# bash run.sh test     run the unit tests
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
if [ -f .env ]; then set -a; . ./.env; set +a; fi   # keys live in .env, never in code

case "${1:-local}" in
  local)  pip install -q -r requirements.txt && echo "Open http://localhost:8000" && python -m uvicorn app.main:app --port 8000 ;;
  test)   pip install -q -r requirements.txt && python -m pytest -q ;;
  docker) docker compose up --build ;;
  *)      echo "usage: bash run.sh [local|docker|test]" >&2; exit 1 ;;
esac
