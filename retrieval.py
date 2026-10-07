"""
Hybrid retrieval: combines dense vector search (ChromaDB embeddings) with
sparse keyword search (BM25), fused with Reciprocal Rank Fusion (RRF).

Why: vector search alone misses exact keyword/number/name matches (it
"understands" meaning but can blur specifics). BM25 alone misses semantic
paraphrases ("car" vs "vehicle"). Fusing both is a standard technique in
production RAG systems because it consistently beats either alone.
"""
import ollama
from rank_bm25 import BM25Okapi

from config import DEFAULT_EMBED_MODEL, DEFAULT_N_RESULTS, CANDIDATE_POOL, RRF_K


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridRetriever:
    def __init__(self, collection, embed_model: str = DEFAULT_EMBED_MODEL):
        self.collection = collection
        self.embed_model = embed_model
        self._bm25 = None
        self._bm25_ids: list = []
        self._bm25_docs: list = []
        self._bm25_meta: list = []

    def refresh_bm25_index(self):
        """Rebuild the BM25 index from whatever is currently in ChromaDB.
        Call this after adding or deleting documents."""
        data = self.collection.get(include=["documents", "metadatas"])
        self._bm25_ids = data.get("ids") or []
        self._bm25_docs = data.get("documents") or []
        self._bm25_meta = data.get("metadatas") or []
        if self._bm25_docs:
            tokenized = [_tokenize(d) for d in self._bm25_docs]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def _vector_search(self, query: str, n_results: int, where: dict | None = None):
        embedding = ollama.embeddings(model=self.embed_model, prompt=query)["embedding"]
        kwargs = {"query_embeddings": [embedding], "n_results": n_results}
        if where:
            kwargs["where"] = where
        results = self.collection.query(**kwargs)
        if not results["ids"][0]:
            return []
        return list(
            zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
        )

    def _bm25_search(self, query: str, n_results: int, allowed_sources: set | None = None):
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in ranked:
            meta = self._bm25_meta[i] or {}
            if allowed_sources and meta.get("source") not in allowed_sources:
                continue
            hits.append((self._bm25_ids[i], self._bm25_docs[i], meta))
            if len(hits) >= n_results:
                break
        return hits

def generate_query_variations(model: str, query: str, n_variations: int = 2) -> list[str]:
    """Generate alternative query formulations to improve recall across varied document phrasing."""
    prompt = f"""Generate {n_variations} alternative, short search queries with the same intent as: '{query}'.
Return ONLY the queries, one per line, no numbering or commentary."""
    try:
        res = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        text = res["message"]["content"]
        variations = [line.strip().lstrip("0123456789.-) ") for line in text.splitlines() if line.strip()]
        return [query] + variations[:n_variations]
    except Exception:
        return [query]


def generate_hyde_doc(model: str, query: str) -> str:
    """Generate a hypothetical document snippet to improve dense vector similarity matching."""
    prompt = f"""Write a short, realistic excerpt from a technical document or manual that answers this question: '{query}'.
Do not include conversational intros or preamble, output only the hypothetical passage."""
    try:
        res = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return res["message"]["content"].strip()
    except Exception:
        return query


def rewrite_query_with_history(model: str, query: str, history: list) -> str:
    """Rewrite follow-up queries using prior chat history into standalone, unambiguous queries."""
    if not history:
        return query
    recent = history[-4:]
    formatted_hist = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)
    prompt = f"""Given the following conversation history and a follow-up question, rewrite the follow-up question so it is a standalone query containing all context needed for document retrieval.

Conversation History:
{formatted_hist}

Follow-up question: {query}

Return ONLY the rewritten standalone question, nothing else."""
    try:
        res = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        rewritten = res["message"]["content"].strip()
        return rewritten if len(rewritten) > 3 else query
    except Exception:
        return query


class HybridRetriever:
    def __init__(self, collection, embed_model: str = DEFAULT_EMBED_MODEL):
        self.collection = collection
        self.embed_model = embed_model
        self._bm25 = None
        self._bm25_ids: list = []
        self._bm25_docs: list = []
        self._bm25_meta: list = []

    def refresh_bm25_index(self):
        """Rebuild the BM25 index from whatever is currently in ChromaDB."""
        data = self.collection.get(include=["documents", "metadatas"])
        self._bm25_ids = data.get("ids") or []
        self._bm25_docs = data.get("documents") or []
        self._bm25_meta = data.get("metadatas") or []
        if self._bm25_docs:
            tokenized = [_tokenize(d) for d in self._bm25_docs]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def _vector_search(self, query: str, n_results: int, where: dict | None = None):
        embedding = ollama.embeddings(model=self.embed_model, prompt=query)["embedding"]
        kwargs = {"query_embeddings": [embedding], "n_results": n_results}
        if where:
            kwargs["where"] = where
        results = self.collection.query(**kwargs)
        if not results["ids"] or not results["ids"][0]:
            return []
        return list(
            zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
        )

    def _bm25_search(self, query: str, n_results: int, allowed_sources: set | None = None):
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in ranked:
            meta = self._bm25_meta[i] or {}
            if allowed_sources and meta.get("source") not in allowed_sources:
                continue
            hits.append((self._bm25_ids[i], self._bm25_docs[i], meta))
            if len(hits) >= n_results:
                break
        return hits

    def search(
        self,
        query: str,
        n_results: int = DEFAULT_N_RESULTS,
        candidate_pool: int = CANDIDATE_POOL,
        rrf_k: int = RRF_K,
        sources: list[str] | None = None,
        multi_query: bool = False,
        hyde: bool = False,
        chat_model: str = "llama3.2",
    ):
        """
        Multi-stage hybrid retrieval with Reciprocal Rank Fusion (RRF).
        Supports Multi-query expansion and HyDE (Hypothetical Document Embeddings).
        """
        if self._bm25 is None:
            self.refresh_bm25_index()

        where = None
        allowed = None
        if sources:
            allowed = set(sources)
            if len(sources) == 1:
                where = {"source": sources[0]}
            else:
                where = {"source": {"$in": sources}}

        queries = [query]
        if multi_query:
            queries = generate_query_variations(chat_model, query, n_variations=2)

        scores: dict = {}
        payload: dict = {}

        for q in queries:
            search_query_vector = generate_hyde_doc(chat_model, q) if hyde else q
            vector_hits = self._vector_search(search_query_vector, candidate_pool, where=where)
            bm25_hits = self._bm25_search(q, candidate_pool, allowed_sources=allowed)

            for rank, (doc_id, doc, meta) in enumerate(vector_hits):
                scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
                payload[doc_id] = (doc, meta)
            for rank, (doc_id, doc, meta) in enumerate(bm25_hits):
                scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
                payload[doc_id] = (doc, meta)

        ranked_ids = sorted(scores, key=scores.get, reverse=True)[:n_results]
        return [
            {
                "id": doc_id,
                "document": payload[doc_id][0],
                "metadata": payload[doc_id][1],
                "score": scores[doc_id],
            }
            for doc_id in ranked_ids
        ]

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks belonging to a source filename. Returns count deleted."""
        data = self.collection.get(include=["metadatas"])
        ids = data.get("ids") or []
        metas = data.get("metadatas") or []
        to_delete = [i for i, m in zip(ids, metas) if m and m.get("source") == source]
        if to_delete:
            self.collection.delete(ids=to_delete)
            self.refresh_bm25_index()
        return len(to_delete)

    def get_all_chunks(self) -> list[dict]:
        """Fetch all indexed chunks with their metadata for vector inspection."""
        data = self.collection.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        return [
            {"id": ids[i], "document": docs[i], "metadata": metas[i]}
            for i in range(len(ids))
        ]

    def stats(self) -> dict:
        """Return basic collection stats for the UI."""
        data = self.collection.get(include=["metadatas"])
        ids = data.get("ids") or []
        metas = data.get("metadatas") or []
        sources: dict[str, int] = {}
        for m in metas:
            if m:
                src = m.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1
        return {"total_chunks": len(ids), "sources": sources}

