"""
Chat API endpoints for the LangChain RAG Chatbot.
Handles conversation management, real-time streaming, airline filtering, and document ingestion.
"""

import json
import logging
import os
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse

from app.models.schemas import ChatRequest, ChatResponse, AirlinesResponse
from app.core.memory import get_conversation_memory, ConversationMemory
from app.core.rag import get_rag_pipeline, RAGPipeline
from app.core.guardrails import get_input_guardrails, get_output_guardrails
from app.core.evaluator import get_evaluator
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


@router.get("/airlines", response_model=AirlinesResponse)
async def get_airlines(
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> AirlinesResponse:
    """
    Get all unique airlines indexed in the PGVector database.
    """
    try:
        stats = rag_pipeline.get_vector_store_stats()
        airlines = sorted(stats.get("unique_airlines", []))
        total_chunks = stats.get("total_chunks", 0)
        return AirlinesResponse(airlines=airlines, total_chunks=total_chunks)
    except Exception as e:
        logger.error(f"Failed to fetch airlines: {e}")
        return AirlinesResponse(airlines=[], total_chunks=0)


@router.get("/conversations")
async def list_conversations(
    conversation_memory: ConversationMemory = Depends(get_conversation_memory)
) -> Dict[str, Any]:
    """
    List all active conversation sessions from PostgreSQL state memory.
    """
    try:
        threads = conversation_memory.get_all_threads()
        return {"success": True, "threads": threads}
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        return {"success": False, "threads": [], "error": str(e)}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    conversation_memory: ConversationMemory = Depends(get_conversation_memory)
) -> ChatResponse:
    """
    Process a chat message with guardrails validation, airline filtering, and RAG evaluation.
    """
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        logger.info(f"Processing chat message (thread={thread_id}, airline={request.airline}): {request.message[:50]}...")
        
        # 1. Input Guardrails Check
        if request.enable_guardrails:
            input_guard = get_input_guardrails()
            guard_check = input_guard.check(request.message, thread_id)
            if not guard_check["passed"]:
                return ChatResponse(
                    response=f"🛡️ **Safety Guardrail Notification**\n\n{guard_check['message']}",
                    thread_id=thread_id,
                    sources=[],
                    guardrails={
                        "passed": False,
                        "code": guard_check["code"],
                        "reason": guard_check["message"],
                        "warnings": [guard_check["message"]],
                        "violations": [guard_check["code"]]
                    },
                    evaluation=None
                )
        
        # 2. Process message through LangGraph RAG memory
        result = conversation_memory.process_message(
            message=request.message,
            thread_id=thread_id,
            airline=request.airline
        )
        raw_response = result["response"]
        sources = result.get("sources", [])
        
        # 3. Output Guardrails Validation & Sanitization
        output_guard_report = None
        final_response = raw_response
        if request.enable_guardrails:
            output_guard = get_output_guardrails()
            output_guard_report = output_guard.validate(raw_response, sources)
            if output_guard_report.get("sanitized_text"):
                final_response = output_guard_report["sanitized_text"]
        
        # 4. RAG Quality Evaluation (LLM-as-a-Judge)
        eval_report = None
        if request.enable_eval:
            evaluator = get_evaluator()
            eval_record = evaluator.evaluate(
                query=request.message,
                response=final_response,
                sources=sources,
                thread_id=thread_id
            )
            eval_report = eval_record.to_dict()
        
        response = ChatResponse(
            response=final_response,
            thread_id=result["thread_id"],
            sources=sources,
            guardrails=output_guard_report,
            evaluation=eval_report
        )
        
        logger.info(f"Chat message processed successfully for thread {result['thread_id']}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to process chat message: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat message: {str(e)}"
        )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    conversation_memory: ConversationMemory = Depends(get_conversation_memory),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Stream token-by-token response via Server-Sent Events (SSE) with real-time Guardrails & Evaluation.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    
    async def event_generator():
        try:
            # 1. Input Guardrails Check
            if request.enable_guardrails:
                input_guard = get_input_guardrails()
                guard_check = input_guard.check(request.message, thread_id)
                if not guard_check["passed"]:
                    violation_payload = {
                        "type": "guardrail_violation",
                        "code": guard_check["code"],
                        "message": guard_check["message"]
                    }
                    yield f"data: {json.dumps(violation_payload)}\n\n"
                    
                    alert_content = f"🛡️ **Safety Guardrail Notification**\n\n{guard_check['message']}"
                    yield f"data: {json.dumps({'type': 'token', 'content': alert_content})}\n\n"
                    
                    done_payload = {
                        "type": "done",
                        "thread_id": thread_id,
                        "guardrails": {
                            "passed": False,
                            "violations": [guard_check["code"]],
                            "warnings": [guard_check["message"]]
                        },
                        "evaluation": None,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    yield f"data: {json.dumps(done_payload)}\n\n"
                    return
            
            # 2. Build contextual question with past conversation history
            contextual_question = conversation_memory.get_contextual_question(thread_id, request.message)
            
            # 3. Get LLM stream and retrieved sources
            token_stream, sources = rag_pipeline.stream_query(contextual_question, airline=request.airline)
            
            # 4. Stream sources first or as metadata
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            
            # 5. Stream tokens
            full_response_parts = []
            for chunk in token_stream:
                token_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if token_text:
                    full_response_parts.append(token_text)
                    yield f"data: {json.dumps({'type': 'token', 'content': token_text})}\n\n"
            
            full_response = "".join(full_response_parts)
            
            # 6. Output Guardrails Validation
            output_guard_report = None
            if request.enable_guardrails:
                output_guard = get_output_guardrails()
                output_guard_report = output_guard.validate(full_response, sources)
            
            # 7. RAG Quality Evaluation (LLM-as-a-Judge)
            eval_report = None
            if request.enable_eval:
                evaluator = get_evaluator()
                eval_record = evaluator.evaluate(
                    query=request.message,
                    response=full_response,
                    sources=sources,
                    thread_id=thread_id
                )
                eval_report = eval_record.to_dict()
                yield f"data: {json.dumps({'type': 'evaluation', 'evaluation': eval_report, 'guardrails': output_guard_report})}\n\n"
            
            # 8. Persist the turn to LangGraph PostgreSQL state memory
            conversation_memory.save_message_turn(thread_id, request.message, full_response)
            
            # 9. Send completion signal with evaluation and guardrails payload
            done_payload = {
                "type": "done",
                "thread_id": thread_id,
                "guardrails": output_guard_report,
                "evaluation": eval_report,
                "timestamp": datetime.utcnow().isoformat()
            }
            yield f"data: {json.dumps(done_payload)}\n\n"
            
        except Exception as e:
            logger.error(f"Error in chat stream: {e}", exc_info=True)
            error_payload = {
                "type": "error",
                "message": f"Error generating answer: {str(e)}"
            }
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    airline: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> Dict[str, Any]:
    """
    Upload an airline policy document (PDF, TXT, HTML) and index it into PGVector immediately.
    """
    try:
        filename = file.filename
        if not filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        
        # Save to local documents directory
        documents_dir = Path(settings.documents_dir)
        documents_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = documents_dir / filename
        content = await file.read()
        with open(target_path, "wb") as f:
            f.write(content)
        
        # Index document into PGVector
        logger.info(f"Indexing newly uploaded document: {filename} (Airline: {airline})")
        result = rag_pipeline.index_single_document(
            file_name=filename,
            airline=airline,
            description=description
        )
        
        return {
            "success": result.get("success", False),
            "filename": filename,
            "airline": airline,
            "chunks_created": result.get("chunks", 0),
            "message": result.get("message", "Uploaded and indexed successfully")
        }
        
    except Exception as e:
        logger.error(f"Failed to upload and index document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/conversation/{thread_id}")
async def get_conversation_history(
    thread_id: str,
    conversation_memory: ConversationMemory = Depends(get_conversation_memory)
) -> Dict[str, Any]:
    """
    Get conversation history for a specific thread.
    """
    try:
        logger.info(f"Retrieving conversation history for thread: {thread_id}")
        history = conversation_memory.get_conversation_history(thread_id)
        return history
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversation history: {str(e)}"
        )


@router.delete("/conversation/{thread_id}")
async def clear_conversation(
    thread_id: str,
    conversation_memory: ConversationMemory = Depends(get_conversation_memory)
) -> Dict[str, Any]:
    """
    Clear conversation history for a specific thread.
    """
    try:
        logger.info(f"Clearing conversation history for thread: {thread_id}")
        result = conversation_memory.clear_conversation(thread_id)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear conversation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear conversation: {str(e)}"
        )
