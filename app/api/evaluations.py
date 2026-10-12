"""
Evaluations & Guardrails API endpoints.

GET    /api/evaluations                  - All evaluations (latest N)
GET    /api/evaluations/stats            - Aggregate quality statistics & guardrails summary
GET    /api/evaluations/{thread_id}      - Evaluations for a specific conversation
POST   /api/evaluations/run              - On-demand evaluation test
DELETE /api/evaluations/{thread_id}      - Clear evaluations for a thread
DELETE /api/evaluations                  - Clear all evaluations
GET    /api/evaluations/guardrails/stats - Guardrail telemetry & active rules
POST   /api/evaluations/guardrails/check - Test a prompt against input guardrails
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from app.core.evaluator import get_evaluator
from app.core.guardrails import get_input_guardrails, get_guardrails_stats
from app.models.schemas import EvalRunRequest, GuardrailCheckRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/stats")
async def get_evaluation_stats() -> Dict[str, Any]:
    """Return aggregate evaluation and guardrails statistics."""
    evaluator = get_evaluator()
    eval_stats = evaluator.get_aggregate_stats()
    gr_stats = get_guardrails_stats()
    return {
        "success": True,
        "stats": eval_stats,
        "guardrails": gr_stats,
    }


@router.get("")
async def get_all_evaluations(limit: int = Query(100, ge=1, le=500)) -> Dict[str, Any]:
    """Return the most recent evaluations across all threads."""
    evaluator = get_evaluator()
    evals = evaluator.get_all_evals(limit=limit)
    return {"success": True, "total": len(evals), "evaluations": evals}


@router.get("/guardrails/stats")
async def get_guardrail_statistics() -> Dict[str, Any]:
    """Return guardrail enforcement telemetry and active rules."""
    stats = get_guardrails_stats()
    return {"success": True, "stats": stats}


@router.post("/guardrails/check")
async def check_guardrails(request: GuardrailCheckRequest) -> Dict[str, Any]:
    """Test a query against the input guardrails."""
    input_guard = get_input_guardrails()
    result = input_guard.check(request.query, request.thread_id or "test_thread")
    return {"success": True, "result": result}


@router.post("/run")
async def run_evaluation(request: EvalRunRequest) -> Dict[str, Any]:
    """Run an on-demand RAG evaluation for a query and response."""
    evaluator = get_evaluator()
    record = evaluator.evaluate(
        query=request.query,
        response=request.response,
        sources=request.sources or [],
        thread_id=request.thread_id or "test_thread"
    )
    return {"success": True, "evaluation": record.to_dict()}


@router.get("/{thread_id}")
async def get_thread_evaluations(thread_id: str) -> Dict[str, Any]:
    """Return all evaluations for a specific conversation thread."""
    evaluator = get_evaluator()
    evals = evaluator.get_evals_for_thread(thread_id)
    return {"success": True, "thread_id": thread_id, "total": len(evals), "evaluations": evals}


@router.delete("/{thread_id}")
async def clear_thread_evaluations(thread_id: str) -> Dict[str, Any]:
    """Clear evaluations for a specific conversation thread."""
    evaluator = get_evaluator()
    evaluator.clear_evals_for_thread(thread_id)
    return {"success": True, "message": f"Evaluations cleared for thread {thread_id}"}


@router.delete("")
async def clear_all_evaluations() -> Dict[str, Any]:
    """Clear all stored evaluations across all threads."""
    evaluator = get_evaluator()
    evaluator.clear_all_evals()
    return {"success": True, "message": "All evaluations cleared successfully"}
