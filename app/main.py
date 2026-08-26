import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from app.agent.support_agent import run_agent_turn
from app.memory.session_memory import clear_session

load_dotenv()

app = FastAPI(title="Aster & Row Support AI API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store the latest debug traces in memory for observability
_session_traces = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    sources: list
    handoff: bool
    session_id: str

class ResetRequest(BaseModel):
    session_id: str

# Handles customer chat messages by executing the support agent turn
# and caching the resulting trace for debug observability.
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        
    try:
        result = run_agent_turn(session_id, request.message)
        # Store trace for observability
        _session_traces[session_id] = result.get("trace", {})
        
        return ChatResponse(
            response=result["response"],
            sources=result["sources"],
            handoff=result["handoff"],
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Wipes the conversation memory and cached trace for a specific session.
@app.post("/api/reset")
async def reset_session(request: ResetRequest):
    clear_session(request.session_id)
    if request.session_id in _session_traces:
        _session_traces[request.session_id] = {}
    return {"status": "success", "message": "Session reset successful."}

# Returns the latest detailed step-by-step trace of the support agent execution for debugging.
@app.get("/api/debug")
async def get_debug_trace(session_id: str):
    if not session_id or session_id not in _session_traces:
        return {"error": "No trace found for this session ID."}
    return _session_traces[session_id]

# Mount static files for the frontend
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
