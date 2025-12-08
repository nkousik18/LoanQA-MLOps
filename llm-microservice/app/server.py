import json
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.router import route
from app.chat import chat_route
from app.batch import batch_route
from app.backend import chat_stream
from app.chat_memory import chat_memory_route
from app.config import settings
from app.auth import verify_api_key, verify_signature, verify_timestamp

# Prometheus monitoring
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST


# ======================================================
# FastAPI App Configuration
# ======================================================

app = FastAPI(
    title="LoanDoc AI – vLLM Backend",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# PROMETHEUS INSTRUMENTATION (Correct Placement)
# ======================================================

# Attach Prometheus middleware BEFORE the app starts
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ======================================================
# Shared Authentication
# ======================================================

async def authenticate(request: Request,
                       api_key: str,
                       x_timestamp: str,
                       x_signature: str):

    verify_api_key(api_key)

    raw_body = (await request.body()).decode()
    print("RAW BODY SERVER RECEIVED:", repr(raw_body))

    verify_timestamp(x_timestamp)

    verify_signature(signature=x_signature, ts=x_timestamp, body=raw_body)


# ======================================================
# Endpoints
# ======================================================

@app.post("/query")
async def query(request: Request,
                api_key: str = Header(None),
                x_timestamp: str = Header(None),
                x_signature: str = Header(None)):

    await authenticate(request, api_key, x_timestamp, x_signature)
    payload = await request.json()
    return await route(payload)


@app.post("/chat/query")
async def chat_query(request: Request,
                     api_key: str = Header(None),
                     x_timestamp: str = Header(None),
                     x_signature: str = Header(None)):

    await authenticate(request, api_key, x_timestamp, x_signature)
    payload = await request.json()
    return await chat_memory_route(payload)


@app.post("/chat/stream")
async def chat_completions_stream(request: Request,
                                  api_key: str = Header(None),
                                  x_timestamp: str = Header(None),
                                  x_signature: str = Header(None)):

    await authenticate(request, api_key, x_timestamp, x_signature)
    payload = await request.json()
    messages = payload.get("messages", [])

    return StreamingResponse(
        chat_stream(messages),
        media_type="text/plain"
    )


@app.post("/batch_query")
async def batch_query(request: Request,
                      api_key: str = Header(None),
                      x_timestamp: str = Header(None),
                      x_signature: str = Header(None)):

    await authenticate(request, api_key, x_timestamp, x_signature)
    payload = await request.json()
    return await batch_route(payload)


@app.post("/chat/completions")
async def chat_completions(request: Request,
                           api_key: str = Header(None),
                           x_timestamp: str = Header(None),
                           x_signature: str = Header(None)):

    await authenticate(request, api_key, x_timestamp, x_signature)
    payload = await request.json()
    return await chat_route(payload)


# Debugging endpoint
@app.post("/debug/signature")
async def debug_signature(request: Request):
    body = await request.body()
    ts = request.headers.get("x-timestamp", "")
    from app.auth import generate_signature

    sig = generate_signature(ts, body.decode())
    return {
        "timestamp_used": ts,
        "raw_body": body.decode(),
        "signature_server_computed": sig
    }


# ======================================================
# Raw metrics endpoint (optional)
# ======================================================

@app.get("/metrics_raw")
def metrics_raw():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
