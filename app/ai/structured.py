"""Structured extraction per document category.

After the analysis stage decides a document is an *invoice*, *contract*, etc., a
second schema-constrained call pulls the fields that matter for that category
(vendor, totals, parties, dates, governing law…). Results are stored as
``document_insights.kind = "extraction"`` with the schema name in ``options`` so a
UI can render a form and downstream systems (accounting, CLM) can consume typed
data instead of prose.

Every field carries provenance expectations: the prompt demands values verbatim
from the text and an overall ``confidence``; unknown fields are empty strings, never
guesses. Schemas are plain dicts (field → type hint string) so adding a category is
a data change, not a code change.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.ai.providers import LLMError, Message, Usage, get_llm, parse_json_response

# field → human-readable type hint shown to the model. A list value means "array of objects with these keys".
SCHEMAS: dict[str, dict[str, Any]] = {
    "invoice": {
        "vendor_name": "string",
        "customer_name": "string",
        "invoice_number": "string",
        "invoice_date": "string (as written)",
        "due_date": "string (as written)",
        "currency": "ISO code if determinable",
        "subtotal": "string (as written)",
        "tax": "string (as written)",
        "total_amount": "string (as written)",
        "payment_terms": "string",
        "line_items": [{"description": "string", "quantity": "string", "unit_price": "string", "amount": "string"}],
    },
    "contract": {
        "parties": [{"name": "string", "role": "string"}],
        "effective_date": "string (as written)",
        "term": "string (duration / end date)",
        "termination_clause": "string (<= 60 words)",
        "governing_law": "string",
        "payment_terms": "string",
        "key_obligations": [{"party": "string", "obligation": "string (<= 30 words)"}],
        "renewal": "string",
        "confidentiality": "string (<= 40 words)",
    },
    "legal": {
        "parties": [{"name": "string", "role": "string"}],
        "jurisdiction": "string",
        "key_dates": [{"label": "string", "date": "string"}],
        "obligations": [{"party": "string", "obligation": "string"}],
        "penalties": "string",
    },
    "financial": {
        "reporting_period": "string",
        "revenue": "string (as written)",
        "net_income": "string (as written)",
        "operating_margin": "string",
        "key_metrics": [{"metric": "string", "value": "string"}],
        "risks": [{"risk": "string (<= 25 words)"}],
        "outlook": "string (<= 60 words)",
    },
    "report": {
        "reporting_period": "string",
        "revenue": "string (as written)",
        "key_metrics": [{"metric": "string", "value": "string"}],
        "risks": [{"risk": "string (<= 25 words)"}],
        "outlook": "string (<= 60 words)",
        "people": [{"name": "string", "role": "string"}],
    },
    "policy": {
        "policy_name": "string",
        "applies_to": "string",
        "effective_date": "string",
        "rules": [{"topic": "string", "rule": "string (<= 30 words)"}],
        "exceptions": "string",
        "owner": "string (team or role)",
    },
    "research": {
        "title": "string",
        "authors": [{"name": "string", "affiliation": "string"}],
        "research_question": "string (<= 40 words)",
        "method": "string (<= 40 words)",
        "key_findings": [{"finding": "string (<= 30 words)"}],
        "limitations": "string (<= 40 words)",
    },
}

DEFAULT_SCHEMA = "generic"
SCHEMAS[DEFAULT_SCHEMA] = {
    "title": "string",
    "document_date": "string",
    "organizations": [{"name": "string", "role": "string"}],
    "people": [{"name": "string", "role": "string"}],
    "key_facts": [{"fact": "string (<= 30 words)"}],
    "action_items": [{"item": "string", "owner": "string", "due": "string"}],
}

# analysis categories → extraction schema
CATEGORY_TO_SCHEMA = {
    "invoice": "invoice",
    "contract": "contract",
    "legal": "legal",
    "financial": "financial",
    "report": "report",
    "policy": "policy",
    "research": "research",
}


def schema_for_category(category: str | None) -> str:
    return CATEGORY_TO_SCHEMA.get((category or "").lower(), DEFAULT_SCHEMA)


class StructuredExtraction(BaseModel):
    schema_name: str
    fields: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(0.0, ge=0, le=1)
    notes: str = ""

    @field_validator("fields", mode="before")
    @classmethod
    def _shape(cls, v):
        if not isinstance(v, dict):
            return {}
        out = {}
        for k, val in v.items():
            if isinstance(val, list):
                out[k] = [x for x in val if isinstance(x, dict | str)][:25]
            elif val is None:
                out[k] = ""
            else:
                out[k] = str(val)[:1000] if not isinstance(val, dict) else val
        return out


_SYSTEM = """You extract structured fields from a document.

Return ONE JSON object: {"fields": {...}, "confidence": number in [0,1], "notes": string (<= 30 words)}.
"fields" must contain exactly the keys of the EXTRACTION SCHEMA below, with values of the described type.
Rules:
- Copy values verbatim from the document (numbers, dates, names as written). Do not normalise or compute.
- If a field is not present in the document, use "" (or [] for arrays). Never guess.
- Arrays: one object per item found, at most 25.
- "confidence" reflects how completely and unambiguously the schema was filled.
Output JSON only."""


def extraction_messages(schema_name: str, document_text: str) -> list[Message]:
    schema = SCHEMAS[schema_name]
    return [
        Message("system", _SYSTEM),
        Message("user", f"EXTRACTION SCHEMA ({schema_name})\n{json.dumps(schema, indent=1)}\n\nDOCUMENT\n{document_text}"),
    ]


def _parse(schema_name: str, text: str) -> StructuredExtraction:
    try:
        data = parse_json_response(text)
        data.setdefault("schema_name", schema_name)
        data["schema_name"] = schema_name
        return StructuredExtraction.model_validate(data)
    except (ValidationError, ValueError) as e:
        raise LLMError(f"Extraction response failed validation: {e}") from e


def extract_structured(schema_name: str, excerpt: str) -> tuple[StructuredExtraction, Usage]:
    c = get_llm().complete(extraction_messages(schema_name, excerpt), temperature=0.0, max_tokens=1800, json_mode=True)
    return _parse(schema_name, c.text), c.usage


async def aextract_structured(schema_name: str, excerpt: str) -> tuple[StructuredExtraction, Usage]:
    c = await get_llm().acomplete(extraction_messages(schema_name, excerpt), temperature=0.0, max_tokens=1800, json_mode=True)
    return _parse(schema_name, c.text), c.usage
