from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

api = FastAPI()

# Allow requests from Chrome extension or localhost for testing
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Pydantic model for request body
class TextRequest(BaseModel):
    text: str

# Dummy endpoints for testing
@api.post("/translate")
async def translate_endpoint(request: TextRequest):
    return {"result": f"Translated: {request.text}"}

@api.post("/summarize")
async def summarize_endpoint(request: TextRequest):
    return {"result": f"Summary: {request.text}"}

@api.post("/tts")
async def tts_endpoint(request: TextRequest):
    return {"result": f"TTS Playing: {request.text}"}

@api.post("/math_explain")
async def math_explain_endpoint(request: TextRequest):
    return {"result": f"Math explained: {request.text}"}
