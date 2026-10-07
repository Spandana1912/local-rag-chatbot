"""
Lightweight, self-contained RAG evaluation harness (RAGAS-style), scored
entirely offline using your local Ollama model — no external API keys.

This is what turns "I built a RAG demo" into "I built and measured a RAG
pipeline, then improved it based on the numbers" — a much stronger resume
and interview story.

Metrics:
- Retrieval Hit Rate: did retrieval actually pull a chunk from the
  document that should have contained the answer?
- Faithfulness (0-1, LLM-as-judge): does the generated answer only state
  things supported by the retrieved context?
- Answer Relevance (0-1, LLM-as-judge): does the answer address the
  question that was asked?
"""
import time
import json
import ollama

JUDGE_MODEL = "llama3.2"

FAITHFULNESS_PROMPT = """You are grading whether an AI answer is faithful to the given context.
Context:
{context}

Answer:
{answer}

Does the answer contain ONLY claims supported by the context? Reply with a single number from 0
to 1 (1 = fully faithful, 0 = completely unsupported), followed by a one-sentence reason.
Format exactly:
SCORE: <number>
REASON: <reason>"""

RELEVANCE_PROMPT = """You are grading whether an AI answer actually addresses the question asked.
Question: {question}
Answer: {answer}

Reply with a single number from 0 to 1 (1 = fully relevant, 0 = not relevant at all), followed by
a one-sentence reason.
Format exactly:
SCORE: <number>
REASON: <reason>"""


def _judge(prompt, judge_model=JUDGE_MODEL):
    try:
        response = ollama.chat(model=judge_model, messages=[{"role": "user", "content": prompt}])
        text = response["message"]["content"]
        score = 0.0
        for line in text.splitlines():
            if line.strip().upper().startswith("SCORE"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
        return score, text
    except Exception as e:
        return 0.0, f"Judge error: {e}"


def evaluate_faithfulness(context, answer, judge_model=JUDGE_MODEL):
    return _judge(FAITHFULNESS_PROMPT.format(context=context, answer=answer), judge_model=judge_model)


def evaluate_relevance(question, answer, judge_model=JUDGE_MODEL):
    return _judge(RELEVANCE_PROMPT.format(question=question, answer=answer), judge_model=judge_model)


def retrieval_hit(expected_source, retrieved_chunks):
    return any(c["metadata"].get("source") == expected_source for c in retrieved_chunks)


def run_eval_suite(dataset_path, retriever, answer_fn, judge_model=JUDGE_MODEL, progress_cb=None):
    """
    dataset_path: JSON file of [{"question": ..., "expected_source": ...}, ...]
    retriever: a HybridRetriever instance
    answer_fn: function(question, chunks) -> answer string
    progress_cb: optional function(current, total, msg) for UI status updates
    """
    with open(dataset_path) as f:
        cases = json.load(f)

    results = []
    total = len(cases)
    for idx, case in enumerate(cases):
        if progress_cb:
            progress_cb(idx, total, f"Evaluating Q{idx+1}/{total}: {case['question'][:30]}...")

        t0 = time.time()
        chunks = retriever.search(case["question"], chat_model=judge_model)
        answer = answer_fn(case["question"], chunks)
        latency = round((time.time() - t0) * 1000, 2)
        context = "\n\n".join(c["document"] for c in chunks)

        hit = retrieval_hit(case["expected_source"], chunks)
        faith_score, faith_reason = evaluate_faithfulness(context, answer, judge_model=judge_model)
        rel_score, rel_reason = evaluate_relevance(case["question"], answer, judge_model=judge_model)

        results.append({
            "question": case["question"],
            "expected_source": case.get("expected_source", ""),
            "retrieval_hit": hit,
            "faithfulness": faith_score,
            "faithfulness_reason": faith_reason,
            "relevance": rel_score,
            "relevance_reason": rel_reason,
            "latency_ms": latency,
            "answer": answer,
        })

    if progress_cb:
        progress_cb(total, total, "Evaluation complete!")

    return results


def summarize(results):
    n = len(results) or 1
    return {
        "retrieval_hit_rate": sum(r["retrieval_hit"] for r in results) / n,
        "avg_faithfulness": sum(r["faithfulness"] for r in results) / n,
        "avg_relevance": sum(r["relevance"] for r in results) / n,
        "avg_latency_ms": sum(r.get("latency_ms", 0) for r in results) / n,
    }