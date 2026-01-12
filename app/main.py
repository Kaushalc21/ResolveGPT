# app/main.py

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from app.rag_pipeline import RAGPipeline
from typing import List
from app.llm_client import generate_final_answer

# Initialize FastAPI app
app = FastAPI(
    title="AI Ticket Resolution System",
    description="RAG-based support ticket resolution using FAISS and embeddings",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")

# Initialize RAG pipeline (runs once at startup)
rag_pipeline = RAGPipeline(
    data_path=r"C:\Users\Kaushal\Desktop\AI-ticket-resolution\data\software_ticket_resolution_70k.csv"
)

# -------- Request & Response Models -------- #

class TicketQuery(BaseModel):
    query: str
    top_k: int = 5

class Match(BaseModel):
    index: int
    text: str
    resolution: str


class TicketResponse(BaseModel):
    final_answer: str
    matches: List[Match]
# -------- API Endpoints -------- #

# Serve HTML page
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# POST endpoint for API clientsfrom llm_client import generate_final_answer

@app.post("/resolve", response_model=TicketResponse)
def resolve_ticket(request: TicketQuery):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Call RAG pipeline (FAISS + BM25 + LLM happens inside)
    result = rag_pipeline.resolve_ticket(
        query=request.query,
        top_k=request.top_k
    )

    return result


    
# ✅ New GET endpoint for quick browser testing
@app.get("/query")
def query_ticket(q: str, top_k: int = 5):
    """
    Resolve a ticket query via GET (for browser testing)
    Example: /query?q=API%20returns%20500%20error&top_k=5
    """
    if not q.strip():
        return JSONResponse(status_code=400, content={"detail": "Query cannot be empty"})

    resolution = rag_pipeline.resolve_ticket(
      query=request.query,
      top_k=request.top_k
    )

    return {"resolution": resolution}
