#!/bin/bash

QUESTION="$1"

API_KEY="supersecret123"
HMAC_SECRET="8102a5637ba3a5770975961d7746abb544d24b51168b6b6586653d7c56dc35e4"

# 1. JSON body EXACTLY as server expects
BODY="{\"question\":\"$QUESTION\"}"

# 2. Unix timestamp (must match server)
TS=$(date +%s)

# 3. Correct signature input: timestamp + "." + body
SIGN_INPUT="${TS}.${BODY}"

# 4. Compute signature
SIGNATURE=$(printf "%s" "$SIGN_INPUT" | openssl dgst -sha256 -hmac "$HMAC_SECRET" | awk '{print $2}')

echo "=============================="
echo "TIMESTAMP: $TS"
echo "BODY: $BODY"
echo "SIGN_INPUT: $SIGN_INPUT"
echo "SIGNATURE: $SIGNATURE"
echo "=============================="

# 5. Send request locally
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -H "api-key: $API_KEY" \
  -H "x-timestamp: $TS" \
  -H "x-signature: $SIGNATURE" \
  --data-binary "$BODY"

echo ""
echo "🎉 Done!"
