# 🏛️ PolicyGPT Architecture & System Design

PolicyGPT is an enterprise-grade Retrieval-Augmented Generation (RAG) chatbot engine specialized in answering questions about airline contracts of carriage, fare rules, and travel policies with sub-second response times, verified source citations, and real-time safety guardrails.

---

## 📐 System Architecture Diagram

```
                                      ┌────────────────────────────────┐
                                      │   User Web Browser / Client   │
                                      │  (ChatGPT Clean Vanilla CSS)   │
                                      └──────────────┬─────────────────┘
                                                     │ HTTP / SSE Stream
                                                     ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FastAPI Backend (Port 8000)                             │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 🛡️ Input Guardrails Engine                                                       │  │
│  │  - Prompt Injection Defense       - Length & Profanity Filter                    │  │
│  │  - Rate Limiter (30 req/min)      - Airline Topic Relevance (Zero-Shot Judge)    │  │
│  └──────────────────────────────────────────┬───────────────────────────────────────┘  │
│                                             │ (Passed)                                 │
│                                             ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 🧠 LangGraph Conversation Engine                                                 │  │
│  │  - Persistent checkpointing to PostgreSQL (statedb)                              │  │
│  │  - Multi-turn contextual query rewriting & memory retention                      │  │
│  └───────────────────┬──────────────────────────────────────────────────────────────┘  │
│                      │                                                                 │
│                      ▼                                                                 │
│  ┌────────────────────────────────────────┐     ┌───────────────────────────────────┐  │
│  │ 🔍 FastEmbed ONNX Engine               │     │ ⚡ Groq Cloud Inference            │  │
│  │  - BAAI/bge-small-en-v1.5              │     │  - openai/gpt-oss-120b            │  │
│  │  - 384-dimensional dense vectors       │     │  - Streaming token output (SSE)   │  │
│  │  - 100% local, no API keys needed      │     │  - Temperature 0.1                │  │
│  └───────────────────┬────────────────────┘     └─────────────────▲─────────────────┘  │
│                      │                                            │                    │
│                      ▼ Vector Search                              │ Grounded Context   │
│  ┌────────────────────────────────────────┐                       │                    │
│  │ 🗄️ PostgreSQL 16 + pgvector (vectordb) ├───────────────────────┘                    │
│  │  - HNSW Index cosine distance          │                                            │
│  │  - Filter by airline metadata          │                                            │
│  └────────────────────────────────────────┘                                            │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 🛡️ Output Guardrails & Sanitizer                                                 │  │
│  │  - Automated PII Redaction ([REDACTED_CARD], [REDACTED_EMAIL], [REDACTED_PHONE]) │  │
│  │  - Hallucination / Overconfident Claim Detector                                  │  │
│  │  - Empty / Malformed Answer Guard                                                │  │
│  └──────────────────────────────────────────┬───────────────────────────────────────┘  │
│                                             │                                          │
│                                             ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 📊 RAG Evaluator (Parallel LLM-as-a-Judge)                                       │  │
│  │  - Context Relevance (0 - 100%)    - Answer Completeness (0 - 100%)              │  │
│  │  - Faithfulness (0 - 100%)         - Quality Tier (A+ Excellent / B / Review)   │  │
│  │  - Concurrent ThreadPoolExecutor execution (~350ms total)                         │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Core Subsystems

### 1. Vector Search & Storage (PGVector + FastEmbed)
- **Embedding Generation**: Uses `fastembed` with model `BAAI/bge-small-en-v1.5`. Runs locally via ONNX Runtime without relying on external embedding APIs.
- **Vector Database**: PostgreSQL with the `pgvector` extension. Chunks are indexed with cosine distance for fast semantic retrieval.
- **Metadata Filtering**: Enables targeted queries filtered by airline (e.g., `airline = 'United Airlines'`) or search across all indexed airlines.

### 2. Conversational Memory (LangGraph Checkpointing)
- Managed via `langgraph` state graphs with PostgreSQL checkpointer (`statedb`).
- Every conversation turn is committed with its `thread_id`, preserving contextual continuity across restarts and client reconnections.
- Contextual query rewriting resolves pronoun references (e.g., "what about for infants?" resolves against previous topic).

### 3. Multi-Layer Guardrails System
- **Layer 1: Input Length & Profanity**: Validates input length (3 to 4000 characters) and blocks profane expressions.
- **Layer 2: Prompt Injection Defense**: Regex matching for instruction override, developer mode jailbreaks, and system prompt leakage attempts.
- **Layer 3: Topic Relevance**: Checks airline domain keywords. If ambiguous, triggers a fast zero-shot LLM check to ensure passenger-related context.
- **Layer 4: Output PII Redaction**: Automatically regex-masks credit cards, email addresses, and phone numbers before user delivery.
- **Layer 5: Hallucination & Absolute Claims**: Flags overconfident assertions like *"100% guaranteed full refund"* to ensure passengers verify directly with the carrier.

### 4. Quality Evaluation (LLM-as-a-Judge)
- Runs asynchronously or in parallel with Server-Sent Events token generation.
- Three core metrics evaluated concurrently via `ThreadPoolExecutor`:
  1. **Context Relevance**: Measures whether retrieved context chunks directly contain the policy required.
  2. **Faithfulness**: Verifies whether every assertion in the response is supported by the context without hallucination.
  3. **Answer Completeness**: Determines whether the response directly and comprehensively answers the passenger query.
- Telemetry stats are aggregate-calculated and accessible via `/api/evaluations/stats`.

### 5. Frontend UI
- Modern, clean ChatGPT-inspired layout built in Vanilla CSS and JavaScript.
- Supports dark/light mode, real-time token streaming with cursor animation, interactive critique drawers, and modal dashboard.

### 6. LangSmith Observability & Tracing
- Native LangSmith instrumentation across LangChain and LangGraph runnables.
- Injects run metadata (`thread_id`, `airline`, `retrieved_count`) and system tags (`policygpt`, `stream`, `chat`).
- Tracks token counts, latency distributions, exact prompt outputs, and retrieval cosine similarity scores in the LangSmith dashboard.
