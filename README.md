# Airline Policies Chatbot

A Retrieval-Augmented Generation (RAG) chatbot application that answers questions about airline travel policies for major airlines. This chatbot uses LangChain to process airline policy documents, stores vector embeddings in PostgreSQL with `pgvector`, and provides a conversational interface powered by **Groq's `openai/gpt-oss-120b`** model.

## Overview

The application features a self-contained RAG pipeline running completely with Docker:

- Loads airline policy documents (PDF, HTML, TXT) directly from the local `documents/` directory
- Generates vector embeddings locally using FastEmbed (`BAAI/bge-small-en-v1.5`), requiring no external embedding API keys
- Stores embeddings in PostgreSQL with the `pgvector` extension for semantic search
- Maintains conversation history using LangGraph with PostgreSQL checkpointing
- Answers questions using **Groq** ultra-fast inference with the open-weights **`openai/gpt-oss-120b`** model
- Provides a clean web-based chat interface

## Technology Stack

- **Backend**: FastAPI (Python 3.11)
- **RAG Framework**: LangChain, LangGraph
- **LLM**: Groq API (`openai/gpt-oss-120b`)
- **Embeddings**: FastEmbed (`BAAI/bge-small-en-v1.5` via ONNX Runtime, local)
- **Vector Database**: PostgreSQL 16 with `pgvector` extension (Docker container)
- **Conversation State**: PostgreSQL with LangGraph checkpointing (Docker container)
- **Containerization**: Docker & Docker Compose

## Quickstart with Docker Compose

The application stack consists of two services in **Docker Compose**:
- **`airline-rag-postgres`**: PostgreSQL 16 with `pgvector` on port 5432
- **`airline-rag-app`**: FastAPI application server on port 8000

### 1. Configure Environment

Copy `.env.example` to `.env` and add your Groq API key:

```bash
cp .env.example .env
```

Edit `.env` and set:
```dotenv
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```
*(Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys))*

### 2. Start the Stack

```bash
docker compose up -d --build
```

### 3. Initialize Databases & Index Documents

Once the containers are running:

1. **Initialize databases** (creates vector extensions, state tables, and schema):
   ```bash
   curl -X POST -H "x-api-key: admin-secret-key-123" http://localhost:8000/api/admin/initialize_databases
   ```

2. **Index documents** (processes local documents and generates vector embeddings):
   ```bash
   curl -X POST -H "x-api-key: admin-secret-key-123" http://localhost:8000/api/admin/index_documents
   ```

3. **Check status**:
   ```bash
   curl http://localhost:8000/api/health?services=true
   ```

### 4. Chat with the Application

Open your browser to:
[http://localhost:8000](http://localhost:8000)

## API Endpoints

- `GET /` - Web chat user interface
- `GET /api/health` - Health check endpoint (optional `?services=true` for full services check)
- `GET /api/health/groq` - Check Groq API connectivity and model status
- `POST /api/chat` - Chat endpoint for user queries
- `POST /api/admin/initialize_databases` - Initialize database schemas
- `POST /api/admin/index_documents` - Trigger background document indexing
- `GET /api/admin/index_documents/{job_id}` - Check indexing job status
- `GET /api/admin/list_documents` - List documents in local storage

## Configuration Reference

Key environment variables in `.env`:

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API Key | Required for chat |
| `LLM_MODEL` | Groq LLM model name | `openai/gpt-oss-120b` |
| `EMBEDDING_MODEL` | FastEmbed embedding model | `BAAI/bge-small-en-v1.5` |
| `VECTOR_DB_CONNECTION_STRING` | PostgreSQL connection for pgvector | `postgresql://postgres:postgrespassword@postgres:5432/vectordb` |
| `CONVERSATION_DB_CONNECTION_STRING` | PostgreSQL connection for LangGraph checkpoints | `postgresql://postgres:postgrespassword@postgres:5432/statedb` |
| `DOCUMENTS_DIR` | Directory containing policy documents | `/app/documents` |
| `ADMIN_API_KEY` | Admin API authentication key | `admin-secret-key-123` |
