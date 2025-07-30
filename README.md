# Therapy-Bot: Emotional Support Chatbot Backend

This is a production-ready backend for an emotional support chatbot using Retrieval-Augmented Generation (RAG) with OpenAI (GPT-3.5/4) and Pinecone for vector search. Built with FastAPI, it is structured for scalability, security, and ease of deployment.

## Features
- **RAG Pipeline**: Combines OpenAI LLMs with Pinecone vector search for context-aware answers.
- **Async & Modular**: Clean, async code with modular structure for easy maintenance.
- **Secure Config**: Environment variables managed via `.env` and `config.py`.
- **Production-Ready**: Suitable for deployment on Render, Railway, etc.

## Project Structure
```
therapy-bot/
├── app/
│   ├── main.py
│   ├── rag_engine.py
│   └── config.py
├── .env
├── requirements.txt
├── run.sh
└── README.md
```

## Setup
1. **Clone the repo** and `cd therapy-bot`
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment**:
   - Copy `.env` and fill in your OpenAI and Pinecone keys and settings.
4. **Run the app**:
   ```bash
   bash run.sh
   ```

## API Usage
- **POST /query**
  - Request: `{ "question": "How can I manage stress?" }`
  - Response: `{ "answer": "..." }`

## Notes
- Make sure your Pinecone index is created and populated with context documents.
- For production, remove `--reload` from `run.sh` and use a process manager.

## License
MIT .
