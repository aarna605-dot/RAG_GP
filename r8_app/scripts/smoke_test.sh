#!/usr/bin/env bash
# Quick check against a running app:  bash scripts/smoke_test.sh [base_url]
set -euo pipefail
API="${1:-http://localhost:8000}"
echo "1) health";            curl -sf "$API/health"; echo
echo "2) web app served";    curl -sf "$API/" | grep -o "<title>[^<]*</title>"
echo "3) blocking /ask";     curl -sf -X POST "$API/ask" -H 'Content-Type: application/json' -d '{"question":"How much can a router save?"}' | python3 -m json.tool | head -12
echo "4) streaming";         curl -sfN -X POST "$API/ask/stream" -H 'Content-Type: application/json' -d '{"question":"hello agent"}' | grep '^event:'
echo "5) unregistered setup should be 400"; curl -s -o /dev/null -w "%{http_code}\n" -X POST "$API/ask" -H 'Content-Type: application/json' -d '{"question":"hello there","setup":"S4"}'
