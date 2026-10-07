"""
CLI: python run_eval.py

Runs the evaluation suite (eval_dataset.json) against whatever is currently
stored in ChromaDB and prints a summary report + per-question breakdown.
Requires Ollama running locally with the embedding and chat models pulled.
"""
import chromadb

from retrieval import HybridRetriever
from generation import stream_answer
from evaluation import run_eval_suite, summarize

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_documents")
retriever = HybridRetriever(collection)


def answer_fn(question, chunks):
    return "".join(stream_answer("llama3.2", question, chunks))


if __name__ == "__main__":
    results = run_eval_suite("eval_dataset.json", retriever, answer_fn)
    summary = summarize(results)

    print("\n=== RAG Evaluation Report ===")
    print(f"Retrieval Hit Rate:  {summary['retrieval_hit_rate']:.0%}")
    print(f"Avg Faithfulness:    {summary['avg_faithfulness']:.2f}")
    print(f"Avg Relevance:       {summary['avg_relevance']:.2f}\n")

    for r in results:
        print(f"- Q: {r['question']}")
        print(f"  hit={r['retrieval_hit']}  faithfulness={r['faithfulness']}  relevance={r['relevance']}\n")