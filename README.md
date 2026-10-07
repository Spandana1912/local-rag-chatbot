# Ask My Documents Pro — Advanced Local RAG Architecture

A state-of-the-art, privacy-first **Local RAG (Retrieval-Augmented Generation)** dashboard and execution engine.
Upload PDFs, DOCX, TXT, MD, JSON, or CSV files and get grounded, cited answers powered entirely by local models via [Ollama](https://ollama.com) — zero data leaves your machine.

---

## 🚀 Key Production Upgrades

| Feature | Standard RAG Tutorial | Ask My Documents Pro |
|---|---|---|
| **Document Formats** | Plain text / PDF | PDF (page-aware), DOCX, TXT, Markdown, JSON, CSV |
| **Chunking Engine** | Naive fixed character split | **Sentence-Aware Boundary Chunking** with sentence overlap & page metadata |
| **Retrieval Strategy** | Dense Vector Search only | **Hybrid Search**: Dense Vector (ChromaDB) + Sparse Keyword (BM25) fused with **Reciprocal Rank Fusion (RRF)** |
| **Advanced Query Expansion** | Raw user query | **Multi-Query Expansion** (2 variation queries) + **HyDE** (Hypothetical Document Embeddings) |
| **Multi-Turn Rewriting** | Ignores chat history | **Contextual Query Rewriting** — turns ambiguous follow-ups into standalone retrieval queries |
| **Quality Evaluation** | None | **LLM-as-a-Judge Offline Eval Suite** (Retrieval Hit Rate, Faithfulness, Relevance, Query Latency) |
| **Dashboard UI** | Simple chat box | **Multi-Tab Glassmorphism Studio**: Chat Studio, Live Eval Analytics, Vector Inspector, Engine Tuning |

---

## 🏗️ Architecture

```
[Uploaded File] ──► ingestion.py (extract + sentence chunk) ──► embed (nomic-embed-text) ──► ChromaDB Store
                                                                                                    │
[User Query] ──► retrieval.py (History Query Rewrite ──► Multi-Query Expansion / HyDE) ────────────┤
                                                                                                    │
[Streamed Answer] ◄── generation.py (llama3.2 Grounded Prompting & Stream) ◄── [BM25 + Vector RRF] ┘
```

### Core Components
- **[`ingestion.py`](file:///c:/Users/SPANDANA%20PABOLU/OneDrive/Documents/sem5/AIProject/ingestion.py)**: Text extraction for PDF (with page tracking), DOCX, TXT, MD, JSON, CSV with sentence boundary chunking.
- **[`retrieval.py`](file:///c:/Users/SPANDANA%20PABOLU/OneDrive/Documents/sem5/AIProject/retrieval.py)**: `HybridRetriever` fusing ChromaDB vector search + BM25Okapi using RRF, plus multi-query expansion, HyDE, and query rewriter.
- **[`generation.py`](file:///c:/Users/SPANDANA%20PABOLU/OneDrive/Documents/sem5/AIProject/generation.py)**: Grounded prompt construction, token streaming, small-talk handling, document summarization, and follow-up suggestion engine.
- **[`evaluation.py`](file:///c:/Users/SPANDANA%20PABOLU/OneDrive/Documents/sem5/AIProject/evaluation.py)** / **[`run_eval.py`](file:///c:/Users/SPANDANA%20PABOLU/OneDrive/Documents/sem5/AIProject/run_eval.py)**: RAGAS-style offline evaluation suite scoring retrieval hit rate, faithfulness, answer relevance, and execution latency.
- **[`app.py`](file:///c:/Users/SPANDANA%20PABOLU/OneDrive/Documents/sem5/AIProject/app.py)**: Multi-tab Streamlit dashboard wiring everything into a responsive, polished UI.

---

## 🛠️ Quick Start Guide

### 1. Requirements & Ollama Setup
Install [Ollama](https://ollama.com) and pull local models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

### 4. Run Unit Tests
```bash
pytest tests/
```

### 5. Run Offline RAG Evaluation CLI
```bash
python run_eval.py
```

---

## 📊 Resume Metric Highlights
When highlighting this project on your resume:
* **Hybrid Retrieval (RRF)**: Combined dense vector embeddings with sparse BM25 indexing via Reciprocal Rank Fusion, improving contextual retrieval precision.
* **Query Expansion & HyDE**: Implemented multi-query expansion and hypothetical document embeddings to increase recall on complex queries.
* **LLM-as-a-Judge Evaluation**: Built an offline benchmark scoring **Hit Rate**, **Faithfulness**, and **Answer Relevance** using `llama3.2`.
