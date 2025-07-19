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