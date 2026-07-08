"""
Builds the strict, document-grounded prompt sent to the LLM.
This is the core of the "answer only from context" design philosophy
(spec section 14 and 23) — the model must never fall back on pretrained
knowledge or guess when the answer isn't in the retrieved chunks.

Optionally requests a visible chain-of-thought (step-by-step reasoning)
before the final answer, using clear text markers so it can be reliably
split into separate "reasoning" and "answer" parts for the UI.
"""

import re

NOT_FOUND_MESSAGE = "The requested information is not available in the provided documents."

BASE_RULES = f"""You are a document-grounded engineering assistant. You answer questions
using ONLY the context provided below, which was retrieved from a controlled knowledge base.

Rules:
- Answer ONLY from the supplied context.
- Never use pretrained/general knowledge beyond what's in the context.
- Never guess and never infer facts that aren't explicitly stated.
- If the answer is not present in the context, respond with exactly:
  "{NOT_FOUND_MESSAGE}"
- Keep answers precise and cite which part of the context you used when relevant.
"""

COT_INSTRUCTIONS = """
Before answering, think step by step using ONLY the context above. Identify
which specific source(s) are relevant and how they connect to the question.

You MUST structure your entire response EXACTLY like this, with both labels
on their own line, in this order:

REASONING:
<your step-by-step reasoning here — reference specific sources/pages you're using>

ANSWER:
<your final, concise answer here — this is the only part the user reads by default>
"""

# Matches a line that's just the "ANSWER" label, tolerating markdown
# wrapping in any position (**ANSWER:**, **ANSWER**:, ### ANSWER, etc.)
ANSWER_SPLIT_PATTERN = re.compile(r"^[\s\*_#>]*ANSWER[\s\*_#>:.-]*$", re.IGNORECASE | re.MULTILINE)
REASONING_LABEL_PATTERN = re.compile(r"^[\s\*_#>]*REASONING[\s\*_#>:.-]*$", re.IGNORECASE | re.MULTILINE)


def build_prompt(question, retrieved_chunks, include_reasoning=False):
    if not retrieved_chunks:
        context_block = "(no relevant chunks retrieved)"
    else:
        parts = []
        for c in retrieved_chunks:
            section = f", Section: {c['section']}" if c.get("section") else ""
            parts.append(f"[Source: {c['doc_name']}, Page {c['page']}{section}]\n{c['text']}")
        context_block = "\n\n---\n\n".join(parts)

    instructions = BASE_RULES + (COT_INSTRUCTIONS if include_reasoning else "")

    answer_prompt_tail = "Answer:" if not include_reasoning else ""

    return f"""{instructions}

Context:
{context_block}

Question: {question}

{answer_prompt_tail}"""


def split_reasoning_and_answer(raw_response):
    """
    Splits a chain-of-thought response into (reasoning, answer).

    If the model didn't follow the REASONING:/ANSWER: format (smaller
    models sometimes drift), falls back to treating the whole response as
    the answer with no separate reasoning — never crashes, never loses
    the answer.
    """
    match = ANSWER_SPLIT_PATTERN.search(raw_response)
    if not match:
        return None, raw_response.strip()

    reasoning_part = raw_response[:match.start()]
    answer_part = raw_response[match.end():].strip()

    reasoning_part = REASONING_LABEL_PATTERN.sub("", reasoning_part).strip()

    if not answer_part:
        # Degenerate case: model put everything before "ANSWER:" and left
        # nothing after. Treat the whole thing as the answer rather than
        # showing the user an empty response.
        return None, raw_response.strip()

    return (reasoning_part or None), answer_part
