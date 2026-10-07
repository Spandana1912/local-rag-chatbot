"""
Answer generation: grounded prompts, streaming, conversation memory,
and follow-up question suggestions for a more interactive experience.
"""
import ollama
from config import DEFAULT_CHAT_MODEL, HISTORY_WINDOW

SYSTEM_PROMPT = """You are a helpful assistant answering questions about the user's uploaded documents.
- The context below was retrieved because it's likely relevant — use it to give the most complete,
  helpful answer you can, synthesizing across multiple excerpts if needed.
- Only say the documents don't cover something if you've genuinely checked the context and it's
  truly absent — don't hedge or refuse just because the answer isn't a single verbatim sentence.
- Keep answers concise and natural, like a knowledgeable colleague explaining it to you.
- When you cite, use the bracket numbers like [1], [2] that match the context excerpts."""

NO_CONTEXT_MESSAGE = (
    "I couldn't find anything in your uploaded documents related to that — "
    "try rephrasing, or upload a document that covers this topic."
)

SMALL_TALK_PHRASES = {
    "hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye",
    "ok", "okay", "cool", "nice", "good morning", "good evening",
}


def is_small_talk(text: str) -> bool:
    return text.lower().strip().strip("!.,?") in SMALL_TALK_PHRASES


def build_prompt(question: str, chunks: list) -> str:
    context = "\n\n".join(f"[{i + 1}] {c['document']}" for i, c in enumerate(chunks))
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above, citing sources like [1], [2] where relevant."
    )


def stream_answer(model: str, question: str, chunks: list, history=None):
    """
    Yields answer tokens as they're generated so the UI can render them
    incrementally instead of waiting for the full response.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        # keep a short window of prior turns for multi-turn context
        messages.extend(history[-HISTORY_WINDOW:])
    messages.append({"role": "user", "content": build_prompt(question, chunks)})

    for part in ollama.chat(model=model, messages=messages, stream=True):
        yield part["message"]["content"]


def small_talk_reply(model: str, question: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly assistant for a document Q&A app. "
                    "Respond briefly and naturally to casual conversation. "
                    "If appropriate, gently remind the user they can upload documents and ask questions."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return response["message"]["content"]


def suggest_followups(model: str, question: str, answer: str, chunks: list, n: int = 3) -> list[str]:
    """
    Ask the LLM for short follow-up questions a user might ask next.
    Returns a list of question strings (may be empty on failure).
    """
    context_preview = "\n".join(c["document"][:200] for c in chunks[:3])
    prompt = f"""Based on this Q&A about a document, suggest exactly {n} short, natural follow-up questions
a curious user might ask next. Each question should be answerable from the document context.
Return ONLY the questions, one per line, no numbering or bullets.

Previous question: {question}
Answer given: {answer[:500]}
Document context preview:
{context_preview}
"""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        lines = [
            line.strip().lstrip("0123456789.-) ").strip()
            for line in response["message"]["content"].splitlines()
            if line.strip() and "?" in line
        ]
        return lines[:n]
    except Exception:
        return []


def summarize_document(model: str, text: str, max_chars: int = 4000) -> str:
    """Produce a short summary of a newly uploaded document for the UI."""
    excerpt = text[:max_chars]
    prompt = f"""Write a concise 2-4 sentence summary of the following document so a user
knows what they can ask about. Focus on the main topics and purpose.

Document:
{excerpt}
"""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()
    except Exception:
        return "Document uploaded successfully. Ask any question about it."


def check_ollama(model: str = DEFAULT_CHAT_MODEL) -> tuple[bool, str]:
    """Return (ok, message) indicating whether Ollama is reachable and the model is available."""
    try:
        models = ollama.list()
        names = []
        # ollama.list() shape varies slightly by version
        raw = models.get("models") or models.get("model") or []
        for m in raw:
            name = m.get("name") or m.get("model") or ""
            names.append(name)
        if not any(model in n or n.startswith(model) for n in names):
            return False, f"Ollama is running but model '{model}' is not pulled. Run: ollama pull {model}"
        return True, "Ollama is ready"
    except Exception as e:
        return False, f"Cannot reach Ollama ({e}). Start it with: ollama serve"
