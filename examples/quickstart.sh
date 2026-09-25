#!/usr/bin/env bash
# End-to-end demo against the docker compose stack (API on :58000).
# signup → create collection → upload the sample corpus → ask two questions (JSON + SSE).
set -euo pipefail
BASE="${BASE:-http://localhost:58000/v1}"
EMAIL="demo-$RANDOM@example.com"
json() { python3 -c "import sys,json; print(json.load(sys.stdin)$1)"; }

TOKEN=$(curl -sf -X POST "$BASE/auth/signup" -H 'content-type: application/json' \
  -d "{\"tenant_name\":\"demo\",\"email\":\"$EMAIL\",\"password\":\"correct-horse-battery\"}" | json '["access_token"]')
AUTH=(-H "authorization: Bearer $TOKEN")
CID=$(curl -sf -X POST "$BASE/collections" "${AUTH[@]}" -H 'content-type: application/json' \
  -d '{"name":"kestrel"}' | json '["id"]')

for f in "$(dirname "$0")"/../evaluation/datasets/corpus/kestrel/*; do
  curl -sf -X POST "$BASE/collections/$CID/documents" "${AUTH[@]}" -F "file=@$f" > /dev/null
done
echo "uploaded; waiting for the worker..."
sleep 4

echo "--- JSON answer"
curl -sf -X POST "$BASE/query" "${AUTH[@]}" -H 'content-type: application/json' \
  -d "{\"question\":\"What does error code E-4471 mean?\",\"collection_ids\":[\"$CID\"]}" | python3 -m json.tool

echo "--- streamed answer (SSE)"
curl -sfN -X POST "$BASE/query" "${AUTH[@]}" -H 'accept: text/event-stream' -H 'content-type: application/json' \
  -d "{\"question\":\"Is vacation unlimited?\",\"collection_ids\":[\"$CID\"]}"
