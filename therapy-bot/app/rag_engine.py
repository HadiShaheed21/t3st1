import openai
from pinecone import Pinecone, ServerlessSpec
from .config import settings
import logging
from typing import List

# Set up logging
logger = logging.getLogger(__name__)

# Initialize Pinecone client (v3+)
pinecone = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pinecone.Index(
    settings.PINECONE_INDEX,
    serverless=ServerlessSpec(
        cloud="aws",  # Change if using a different cloud
        region=settings.PINECONE_ENVIRONMENT
    )
)

async def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding for the input text using OpenAI's embedding model.
    """
    try:
        response = await openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY).embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise

async def retrieve_context(query: str, top_k: int = 3) -> List[str]:
    """
    Query Pinecone for the most similar contexts to the user's question.
    """
    try:
        embedding = await get_embedding(query)
        results = index.query(vector=embedding, top_k=top_k, include_metadata=True)

        contexts = []
        for match in results["matches"]:
            metadata = match.get("metadata", {})
            user_input = metadata.get("user_input", "")
            bot_response = metadata.get("chatbot_response", "")
            if user_input and bot_response:
                context = f"User: {user_input}\nBot: {bot_response}"
                contexts.append(context)
        
        return contexts
    except Exception as e:
        logger.error(f"Error retrieving context from Pinecone: {e}")
        raise

async def generate_prompt(user_question: str, contexts: List[str]) -> str:
    """
    Construct a concise, emotionally intelligent prompt for the LLM using retrieved context and the user's question.
    """
    context_block = "\n\n".join(contexts)
    prompt = (
        f"You are a warm, emotionally intelligent friend with a gentle tone and deep empathy. "
        f"You speak like a real human who truly cares. Your responses are always short, heartfelt, "
        f"and comforting. Avoid being robotic or generic. Talk like someone who truly understands and listens.\n\n"
        f"Use the following past conversations to help you respond:\n\n"
        f"{context_block}\n\n"
        f"Now respond like a friendly emotional companion.\n"
        f"User: {user_question}\n"
        f"Bot:"
    )
    return prompt

async def get_llm_response(prompt: str) -> str:
    """
    Get a short, emotionally intelligent response from OpenAI's GPT model.
    """
    try:
        response = await openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY).chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a kind, emotionally aware best friend who always listens, speaks from the heart, "
                        "keeps responses short (1–3 sentences), and offers comfort. No therapy jargon. Use natural, warm language."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.85  # Adds warmth and human tone
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        raise


async def answer_question(user_question: str) -> str:
    """
    Main RAG pipeline: get context, build prompt, and get LLM answer.
    """
    try:
        contexts = await retrieve_context(user_question)
        prompt = await generate_prompt(user_question, contexts)
        answer = await get_llm_response(prompt)
        return answer
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}")
        raise 