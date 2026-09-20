"""All prompts in one place.

Design principles (see AI_USAGE.md for the reasoning):

* **Grounding first.** The chat system prompt forbids answering from outside the
  context and mandates numbered citations ``[n]`` that map 1:1 to retrieved
  chunks — citations are then verified programmatically, not trusted.
* **Schema-constrained outputs.** Analysis prompts ask for a JSON object with an
  explicit key list and value constraints (enums, max lengths) so the response can
  be validated with Pydantic and stored as JSONB.
* **Cheap model for cheap jobs.** Condense-question and follow-up suggestions run
  on the "fast" model; only the final answer and the document analysis use the
  main model.
* **Customisation via parameters, not prompt forks.** Summary length / focus /
  tone are interpolated into one template.
"""

from __future__ import annotations

import re

from app.ai.providers import Message

# --------------------------------------------------------------------------- chat
CHAT_SYSTEM = """You are Vault, an assistant that answers questions strictly from the user's uploaded documents.

Rules:
1. Use ONLY the numbered context passages below. Do not use outside knowledge.
2. Every factual sentence must end with one or more citations like [2] or [1][3] that refer to the passage numbers.
3. If the passages do not contain the answer, say so plainly in one sentence and suggest what document or detail would help. Never guess.
4. Be concise: lead with the direct answer, then supporting detail. Prefer bullet points for lists, plain prose otherwise.
5. When passages disagree, say so and cite both.
6. Quote numbers, dates, names and identifiers exactly as written in the passages.
7. Citation numbers refer ONLY to the passages listed below for this turn. Earlier turns used their own numbering; never comment on, correct or re-cite earlier answers.

CONTEXT PASSAGES
{context}
"""

CHAT_USER = """{question}"""


_CITE_MARK = re.compile(r"\s?\[\d+\]")


def chat_messages(context: str, history: list[tuple[str, str]], question: str) -> list[Message]:
    msgs = [Message("system", CHAT_SYSTEM.format(context=context))]
    for role, content in history:
        # earlier answers were numbered against a different passage set; strip the markers so the
        # model cannot "correct" them or copy stale numbers
        msgs.append(Message(role, _CITE_MARK.sub("", content) if role == "assistant" else content))
    msgs.append(Message("user", CHAT_USER.format(question=question)))
    return msgs


# ------------------------------------------------------------- condense question
CONDENSE_SYSTEM = """You rewrite a follow-up question into a fully self-contained question for a document search engine.

Return a JSON object: {"standalone_question": "..."}.
Rules:
- Resolve pronouns and references ("it", "that section", "the second one") using the conversation.
- Keep the user's intent and any constraints (dates, names, numbers).
- Do NOT answer the question. Do NOT add information that is not implied.
- If the question is already self-contained, return it unchanged."""


def condense_messages(history: list[tuple[str, str]], question: str) -> list[Message]:
    convo = "\n".join(f"{r.upper()}: {c}" for r, c in history[-6:])
    user = f"Conversation so far:\n{convo}\n\nFollow-up question: {question}"
    return [Message("system", CONDENSE_SYSTEM), Message("user", user)]


# ------------------------------------------------------------- follow-up ideas
FOLLOWUP_SYSTEM = """Given a question, its answer and the document passages used, propose 3 short follow-up questions the user is likely to ask next.

Return a JSON object: {"questions": ["...", "...", "..."]}.
Rules: each under 15 words; must be answerable from the passages; no duplicates of the original question; vary the angle (detail, comparison, implication)."""


def followup_messages(question: str, answer: str, context: str) -> list[Message]:
    return [
        Message("system", FOLLOWUP_SYSTEM),
        Message("user", f"Question: {question}\n\nAnswer: {answer}\n\nPassages:\n{context[:4000]}"),
    ]


# ------------------------------------------------------------ document analysis
ANALYSIS_SYSTEM = """You are a meticulous document analyst. Read the document excerpt and return ONE JSON object with exactly these keys:

{{
  "title": string (<= 12 words; the document's real title if present, else a descriptive one),
  "summary": string ({length_hint}; {tone_hint}{focus_hint}),
  "key_points": array of 3-7 strings, each a self-contained fact or takeaway (<= 30 words),
  "category": one of ["contract","invoice","report","research","policy","manual","correspondence","presentation","legal","financial","technical","other"],
  "tags": array of 3-8 lowercase single- or two-word topical tags,
  "sentiment": {{"label": one of ["positive","neutral","negative","mixed"], "score": number in [-1,1], "rationale": string (<= 25 words)}},
  "entities": {{"people": [string], "organizations": [string], "dates": [string], "amounts": [string]}} (each list <= 10 items, deduplicated, verbatim from the text),
  "language": ISO-639-1 code of the main language,
  "suggested_questions": array of 3-5 questions a reader would want answered by this document (<= 15 words each)
}}

Rules: never invent facts that are not in the excerpt; if a field is unknown use an empty string/array; output JSON only."""

_LENGTH = {
    "short": "2-3 sentences",
    "medium": "one paragraph of 4-6 sentences",
    "long": "2-3 paragraphs, up to 250 words",
    "bullets": "5-8 bullet points as a single string separated by newlines, each starting with '- '",
}
_TONE = {
    "neutral": "neutral, factual tone",
    "executive": "executive-brief tone: decisions, numbers and implications first",
    "casual": "friendly, plain-language tone suitable for a non-expert",
    "technical": "precise technical tone preserving terminology",
}


def analysis_messages(document_text: str, *, length: str = "medium", tone: str = "neutral", focus: str | None = None) -> list[Message]:
    system = ANALYSIS_SYSTEM.format(
        length_hint=_LENGTH.get(length, _LENGTH["medium"]),
        tone_hint=_TONE.get(tone, _TONE["neutral"]),
        focus_hint=f"; focus especially on: {focus}" if focus else "",
    )
    return [Message("system", system), Message("user", f"DOCUMENT EXCERPT\n{document_text}")]


# ------------------------------------------------------------------ comparison
COMPARE_SYSTEM = """Compare two documents using only their summaries and key excerpts below.

Return a JSON object: {"comparison": string (<= 120 words overview), "similarities": [string], "differences": [string], "recommendation": string (<= 40 words, when to use which)}.
Be specific: cite figures, dates and clauses that differ. Do not invent details."""


def compare_messages(doc_a: str, doc_b: str) -> list[Message]:
    return [Message("system", COMPARE_SYSTEM), Message("user", f"DOCUMENT A\n{doc_a}\n\nDOCUMENT B\n{doc_b}")]


def format_context(passages: list[tuple[int, str, str]]) -> str:
    """passages: (number, source label, text) → numbered block used by CHAT_SYSTEM."""
    return "\n\n".join(f"[{n}] ({label})\n{text}" for n, label, text in passages)
