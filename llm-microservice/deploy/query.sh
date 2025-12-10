#!/bin/bash

QUESTION="$1"

# ----------------------------
# 1. Set secrets
# ----------------------------
API_KEY="supersecret123"
HMAC_SECRET="8102a5637ba3a5770975961d7746abb544d24b51168b6b6586653d7c56dc35e4"

# ----------------------------
# 2. Session ID (dummy for now)
# ----------------------------
SESSION_ID="session_test_user_001_540108f1"

# ----------------------------
# 3. JSON body EXACTLY as it will be sent
# ----------------------------
BODY="{\"question\":\"$QUESTION\",\"session_id\":\"$SESSION_ID\"}"

# ----------------------------
# 4. Unix timestamp
# ----------------------------
TS=$(date +%s)

# ----------------------------
# 5. SIGN EXACT STRING:
#       timestamp + "." + body
# ----------------------------
SIGN_INPUT="${TS}.${BODY}"

SIGNATURE=$(printf "%s" "$SIGN_INPUT" | \
  openssl dgst -sha256 -hmac "$HMAC_SECRET" | awk '{print $2}')

echo "=============================="
echo "TIMESTAMP: $TS"
echo "BODY: $BODY"
echo "SIGN_INPUT: $SIGN_INPUT"
echo "SIGNATURE: $SIGNATURE"
echo "=============================="

# ----------------------------
# 6. Make request
# ----------------------------
curl -s -X POST "http://34.148.239.255:8001/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -H "X-Timestamp: $TS" \
  -H "X-Signature: $SIGNATURE" \
  -d "$BODY"

echo ""
