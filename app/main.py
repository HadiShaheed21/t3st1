from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from .rag_engine import answer_question
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    """
    Root endpoint - API status and information.
    """
    return {
        "message": "Therapy Bot API is running!",
        "status": "healthy",
        "endpoints": {
            "POST /query": "Send a question to the therapy bot",
            "GET /docs": "API documentation",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "message": "Therapy Bot API is operational"}

@app.get("/debug")
async def debug_config():
    """
    Debug endpoint to check configuration (without exposing sensitive data).
    """
    from .config import settings
    return {
        "openai_key_present": bool(settings.OPENAI_API_KEY),
        "openai_key_length": len(settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else 0,
        "openai_key_prefix": settings.OPENAI_API_KEY[:10] + "..." if settings.OPENAI_API_KEY else "None",
        "pinecone_key_present": bool(settings.PINECONE_API_KEY),
        "pinecone_environment": settings.PINECONE_ENVIRONMENT,
        "pinecone_index": settings.PINECONE_INDEX
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: Request, body: QueryRequest):
    """
    Endpoint to handle user questions and return an LLM-generated answer.
    """
    try:
        logger.info(f"Received question: {body.question}")
        answer = await answer_question(body.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error in /query endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.") 