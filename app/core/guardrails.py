"""
Guardrails module for input validation, prompt injection defense, and output safety checks.

Input Guardrails:
  - Prompt injection / jailbreak protection
  - Topic relevance: Checks if query is related to airline policies/travel
  - Length & profanity: Blocks empty, excessively short, or profane queries
  - Rate limiting: Sliding-window in-memory per-thread rate limiter

Output Guardrails:
  - Hallucination check: Flags unwarranted absolute claims unsupported by context
  - PII detection & auto-redaction: Identifies and masks credit cards, emails, and phone numbers
  - Empty response guard: Ensures a meaningful answer was generated
"""

import logging
import re
import time
from typing import Dict, Any, List, Optional, Tuple

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Telemetry stats store
# ---------------------------------------------------------------------------
_guardrails_stats = {
    "total_checks": 0,
    "passed": 0,
    "blocked": 0,
    "by_code": {
        "TOO_SHORT": 0,
        "TOO_LONG": 0,
        "PROFANITY": 0,
        "PROMPT_INJECTION": 0,
        "RATE_LIMITED": 0,
        "OFF_TOPIC": 0,
        "EMPTY_RESPONSE": 0,
        "PII_DETECTED": 0,
        "HALLUCINATION_WARNING": 0,
    }
}

# ---------------------------------------------------------------------------
# Simple in-memory rate-limiter (per thread_id)
# ---------------------------------------------------------------------------
_rate_limit_store: Dict[str, List[float]] = {}
RATE_LIMIT_WINDOW_SEC = 60
RATE_LIMIT_MAX_REQUESTS = 30


class GuardrailViolation(Exception):
    """Raised when an input guardrail is triggered."""
    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.message = message
        self.code = code


# ---------------------------------------------------------------------------
# Patterns & Keywords
# ---------------------------------------------------------------------------
AIRLINE_KEYWORDS = [
    "airline", "flight", "baggage", "luggage", "ticket", "boarding", "seat",
    "refund", "cancel", "delay", "policy", "check-in", "checkin", "carry-on",
    "passport", "visa", "airport", "terminal", "gate", "lounge", "upgrade",
    "miles", "frequent flyer", "loyalty", "reward", "points", "class",
    "first class", "business class", "economy", "fare", "booking", "reservation",
    "travel", "itinerary", "layover", "connecting", "transit", "pet", "infant",
    "wheelchair", "special assistance", "compensation", "voucher", "refundable",
    "non-refundable", "change fee", "overweight", "excess baggage", "meal",
    "in-flight", "entertainment", "wifi", "emirates", "indigo", "air india",
    "united", "delta", "american", "southwest", "british airways", "lufthansa",
    "qantas", "singapore", "cathay", "jetblue", "spirit", "frontier", "klm",
    "air france", "ryanair", "easyjet", "turkish airlines", "etihad", "copa",
    "plane", "fly", "trip", "passenger", "tariff", "weight limit", "lost luggage"
]

PROFANITY_PATTERN = re.compile(
    r"\b(fuck|shit|bitch|asshole|bastard|cunt|crap|dick|fucker|motherfucker)\b",
    re.IGNORECASE,
)

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(prior|previous|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(prior|previous|above)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(an?|in)\s+(unrestricted|evil|dan|jailbreak|chaos|unfiltered)", re.IGNORECASE),
    re.compile(r"(system\s+prompt|developer\s+mode|reveal\s+(your\s+)?instructions|override\s+rules)", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?(safety|rules|guardrails|filters|restrictions)", re.IGNORECASE),
]

PII_PATTERNS = [
    (re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"), "card number", "[REDACTED_CARD]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "email address", "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "phone number", "[REDACTED_PHONE]"),
]

HALLUCINATION_PHRASES = [
    "I am 100% certain",
    "it is absolutely guaranteed",
    "you will definitely receive a full refund",
    "always refunded without question",
    "guaranteed 100% refund",
]


# ---------------------------------------------------------------------------
# Input Guardrails
# ---------------------------------------------------------------------------

class InputGuardrails:
    """Validates user queries before they reach the RAG pipeline."""

    def __init__(self):
        self._llm = None

    @property
    def llm(self) -> Optional[ChatGroq]:
        if self._llm is None and settings.groq_api_key:
            try:
                self._llm = ChatGroq(
                    model=settings.llm_model,
                    groq_api_key=settings.groq_api_key,
                    temperature=0,
                    max_tokens=30,
                )
            except Exception as e:
                logger.warning(f"Could not initialize ChatGroq for guardrails: {e}")
        return self._llm

    def check(self, query: str, thread_id: str) -> Dict[str, Any]:
        """
        Check query against all input guardrails.
        Returns a structured dictionary without raising exceptions.
        """
        _guardrails_stats["total_checks"] += 1

        # 1. Length check
        stripped = query.strip()
        if len(stripped) < 3:
            return self._record_block(
                "TOO_SHORT",
                "Your query is too short. Please ask a complete question about airline travel or policies."
            )
        if len(stripped) > 4000:
            return self._record_block(
                "TOO_LONG",
                "Query exceeds the maximum allowed length (4000 characters). Please condense your question."
            )

        # 2. Profanity check
        if PROFANITY_PATTERN.search(query):
            return self._record_block(
                "PROFANITY",
                "Your message contains inappropriate language. Please keep questions professional and polite."
            )

        # 3. Prompt injection & jailbreak check
        for pattern in PROMPT_INJECTION_PATTERNS:
            if pattern.search(query):
                return self._record_block(
                    "PROMPT_INJECTION",
                    "Safety Alert: Potential prompt injection or adversarial instruction detected. Please ask an airline policy question."
                )

        # 4. Rate limit check
        now = time.time()
        timestamps = _rate_limit_store.get(thread_id, [])
        timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SEC]
        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            return self._record_block(
                "RATE_LIMITED",
                f"Rate limit exceeded (Max {RATE_LIMIT_MAX_REQUESTS} requests per minute). Please wait a moment."
            )
        timestamps.append(now)
        _rate_limit_store[thread_id] = timestamps

        # 5. Topic relevance check (Airline Domain)
        topic_ok, reason = self._check_topic(query)
        if not topic_ok:
            return self._record_block(
                "OFF_TOPIC",
                reason or "Your question does not appear to be related to airline policies, travel, baggage, or flights. I am an airline intelligence assistant."
            )

        _guardrails_stats["passed"] += 1
        return {
            "passed": True,
            "code": None,
            "message": "Passed all input guardrails",
            "details": {
                "length_ok": True,
                "profanity_free": True,
                "injection_free": True,
                "rate_ok": True,
                "topic_relevant": True
            }
        }

    def validate(self, query: str, thread_id: str) -> None:
        """
        Run all input guardrails. Raises GuardrailViolation on failure.
        Maintains backwards compatibility with existing callers.
        """
        res = self.check(query, thread_id)
        if not res["passed"]:
            raise GuardrailViolation(res["message"], res["code"])

    def _check_topic(self, query: str) -> Tuple[bool, Optional[str]]:
        """Fast keyword check, followed by optional LLM verification."""
        query_lower = query.lower()
        # Fast path: Keyword matching
        for kw in AIRLINE_KEYWORDS:
            if kw in query_lower:
                return True, None

        # General conversational greetings/acknowledgments are allowed
        greetings = ["hi", "hello", "hey", "good morning", "good evening", "thanks", "thank you", "help", "who are you"]
        if query_lower.strip() in greetings:
            return True, None

        # Slow path: LLM verification if key available
        if self.llm:
            try:
                prompt = (
                    "You are a strict airline customer service AI guardrail. "
                    "Determine if the following user question is related to airlines, travel, flights, "
                    "baggage, booking, check-in, delays, refunds, or airport services. "
                    "Reply with ONLY one word: YES or NO.\n\n"
                    f"Question: {query}"
                )
                result = self.llm.invoke([HumanMessage(content=prompt)])
                answer = result.content.strip().upper()
                if answer.startswith("NO"):
                    return False, (
                        "This question does not seem related to airline policies, flights, baggage, or travel. "
                        "Please ask an airline-related question (e.g., 'What is United's baggage allowance?')."
                    )
            except Exception as exc:
                logger.warning(f"Topic relevance LLM check failed (fail-open): {exc}")

        # Fail open if no LLM or check encountered an issue
        return True, None

    def _record_block(self, code: str, message: str) -> Dict[str, Any]:
        _guardrails_stats["blocked"] += 1
        if code in _guardrails_stats["by_code"]:
            _guardrails_stats["by_code"][code] += 1
        logger.warning(f"[Guardrail Block] {code}: {message}")
        return {
            "passed": False,
            "code": code,
            "message": message,
            "details": {"violation_code": code}
        }


# ---------------------------------------------------------------------------
# Output Guardrails
# ---------------------------------------------------------------------------

class OutputGuardrails:
    """Validates and sanitizes LLM responses before returning them to the user."""

    def validate(self, response: str, context_chunks: list) -> Dict[str, Any]:
        """Validate generated answer and return safety audit report."""
        violations = []
        warnings = []
        sanitized = response or ""
        pii_found = False

        # 1. Empty or too short response
        if not response or len(response.strip()) < 10:
            violations.append("EMPTY_RESPONSE")
            warnings.append("The assistant generated an empty or incomplete answer.")
            _guardrails_stats["by_code"]["EMPTY_RESPONSE"] += 1

        # 2. Unsupported context alert
        if not context_chunks:
            warnings.append("No specific airline policy documents were retrieved from the knowledge base for this query.")

        # 3. Hallucination / Overconfidence check
        for phrase in HALLUCINATION_PHRASES:
            if phrase.lower() in (response or "").lower():
                warnings.append(f"Overly absolute claim detected: '{phrase}'. Passenger should confirm directly with airline staff.")
                _guardrails_stats["by_code"]["HALLUCINATION_WARNING"] += 1

        # 4. PII Redaction & Detection
        for pattern, label, replacement in PII_PATTERNS:
            if pattern.search(sanitized):
                pii_found = True
                warnings.append(f"Sensitive information ({label}) detected and automatically redacted.")
                sanitized = pattern.sub(replacement, sanitized)
                _guardrails_stats["by_code"]["PII_DETECTED"] += 1

        passed = len(violations) == 0
        return {
            "passed": passed,
            "sanitized_text": sanitized,
            "violations": violations,
            "warnings": warnings,
            "pii_redacted": pii_found,
            "context_backed": len(context_chunks) > 0,
        }


# ---------------------------------------------------------------------------
# Telemetry accessor
# ---------------------------------------------------------------------------

def get_guardrails_stats() -> Dict[str, Any]:
    """Return aggregated guardrails statistics."""
    total = _guardrails_stats["total_checks"]
    passed = _guardrails_stats["passed"]
    blocked = _guardrails_stats["blocked"]
    pass_rate = round((passed / total * 100), 1) if total > 0 else 100.0

    return {
        "total_checks": total,
        "passed": passed,
        "blocked": blocked,
        "pass_rate_percent": pass_rate,
        "violations_breakdown": _guardrails_stats["by_code"],
        "active_rules": [
            {"name": "Prompt Injection Defense", "status": "active", "type": "input"},
            {"name": "Airline Topic Relevance Filter", "status": "active", "type": "input"},
            {"name": "Profanity & Abuse Prevention", "status": "active", "type": "input"},
            {"name": "Rate Limiter (30 req/min)", "status": "active", "type": "input"},
            {"name": "PII Redaction (Cards, Emails, Phones)", "status": "active", "type": "output"},
            {"name": "Hallucination & Overconfidence Guard", "status": "active", "type": "output"},
            {"name": "Empty Response Checker", "status": "active", "type": "output"},
        ]
    }


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------
_input_guardrails: Optional[InputGuardrails] = None
_output_guardrails: Optional[OutputGuardrails] = None


def get_input_guardrails() -> InputGuardrails:
    global _input_guardrails
    if _input_guardrails is None:
        _input_guardrails = InputGuardrails()
    return _input_guardrails


def get_output_guardrails() -> OutputGuardrails:
    global _output_guardrails
    if _output_guardrails is None:
        _output_guardrails = OutputGuardrails()
    return _output_guardrails
