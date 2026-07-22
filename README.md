# Ask My Documents — Local RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions from your own documents (.txt, .pdf, .docx), running entirely locally using Ollama — no cloud API, no data leaves your machine.

## Features
- Upload multiple documents (TXT, PDF, DOCX)
- Semantic search using embeddings (nomic-embed-text)
- Source-cited answers showing which document/chunk was used
- Casual conversation handling (skips retrieval for small talk)
- Fully local inference via Ollama (llama3.2)

## Tech Stack
- **Ollama** — local LLM inference and embeddings
- **ChromaDB** — local vector database
- **Streamlit** — web interface
- **pypdf / python-docx** — document parsing

## Setup
1. Install [Ollama](https://ollama.com)
2. Pull models: `ollama pull llama3.2` and `ollama pull nomic-embed-text`
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`

## Future Improvements
- OCR support for scanned PDFs
- Retrieval evaluation harness
- Hybrid (keyword + semantic) search
- Cloud model backend option for deployment