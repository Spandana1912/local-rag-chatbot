"""
Ask My Documents Pro — Advanced Local RAG Dashboard
"""
import time
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import chromadb
import ollama

from config import (
    DB_PATH,
    COLLECTION_NAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_N_RESULTS,
    MIN_RELEVANCE_SCORE,
    PAGE_TITLE,
    PAGE_ICON,
    MAX_UPLOAD_MB,
    ENABLE_MULTI_QUERY,
    ENABLE_HYDE,
    ENABLE_QUERY_REWRITE,
)
from ingestion import extract_text, chunk_document
from retrieval import HybridRetriever, rewrite_query_with_history
from generation import (
    is_small_talk,
    small_talk_reply,
    stream_answer,
    suggest_followups,
    summarize_document,
    check_ollama,
    NO_CONTEXT_MESSAGE,
)
from evaluation import run_eval_suite, summarize

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# Modern UI Styling
st.markdown(
    """
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1.2rem; max-width: 1100px;}
    
    /* Header & Badge Styling */
    .pro-badge {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white; padding: 3px 10px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px;
        display: inline-block; margin-left: 8px; vertical-align: middle;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(226, 232, 240, 0.15);
        border-radius: 12px; padding: 16px; text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-val {
        font-size: 1.8rem; font-weight: 800; color: #4f46e5;
    }
    .metric-lbl {
        font-size: 0.82rem; color: #64748b; font-weight: 500; margin-top: 4px;
    }
    .doc-pill {
        display: inline-block; background: #e0e7ff; color: #3730a3;
        padding: 4px 12px; border-radius: 999px; font-size: 0.82rem;
        margin: 2px 4px 2px 0; font-weight: 500;
    }
    .source-tag {
        display: inline-block; background: #f1f5f9; color: #334155;
        padding: 3px 10px; border-radius: 999px; font-size: 0.78rem;
        margin: 2px 4px 2px 0; font-weight: 600;
    }
    .score-pill {
        display: inline-block; background: #ecfdf5; color: #047857;
        padding: 2px 8px; border-radius: 999px; font-size: 0.75rem;
        font-weight: 600; margin-left: 6px;
    }
    .meta-chip {
        display: inline-block; background: #f8fafc; color: #475569;
        border: 1px solid #e2e8f0; padding: 2px 8px; border-radius: 6px;
        font-size: 0.72rem; margin-right: 6px;
    }
    .status-ok { color: #10b981; font-weight: 600; }
    .status-bad { color: #ef4444; font-weight: 600; }
    
    /* Tabs customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        padding: 0 16px;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(name=COLLECTION_NAME)


collection = get_collection()

# --- Session State Initialization ---
if "retriever" not in st.session_state:
    st.session_state.retriever = HybridRetriever(collection)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "followups" not in st.session_state:
    st.session_state.followups = []
if "doc_summaries" not in st.session_state:
    st.session_state.doc_summaries = {}
if "chat_model" not in st.session_state:
    st.session_state.chat_model = DEFAULT_CHAT_MODEL
if "embed_model" not in st.session_state:
    st.session_state.embed_model = DEFAULT_EMBED_MODEL
if "n_results" not in st.session_state:
    st.session_state.n_results = DEFAULT_N_RESULTS
if "enable_multi_query" not in st.session_state:
    st.session_state.enable_multi_query = ENABLE_MULTI_QUERY
if "enable_hyde" not in st.session_state:
    st.session_state.enable_hyde = ENABLE_HYDE
if "enable_query_rewrite" not in st.session_state:
    st.session_state.enable_query_rewrite = ENABLE_QUERY_REWRITE
if "selected_sources" not in st.session_state:
    st.session_state.selected_sources = []
if "last_eval_results" not in st.session_state:
    st.session_state.last_eval_results = None


def known_sources() -> list[str]:
    data = collection.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    return sorted({m.get("source", "unknown") for m in metadatas if m})


def export_chat_markdown() -> str:
    lines = [f"# Ask My Documents Pro — Chat Export ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"]
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"### {role}\n{msg['content']}\n")
        if msg.get("sources"):
            lines.append(f"*Sources used: {', '.join(msg['sources'])}*\n")
    return "\n".join(lines)


# ========== SIDEBAR ==========
with st.sidebar:
    st.title("⚡ RAG Control Panel")
    st.caption("Privacy-first local intelligence engine")

    ok, status_msg = check_ollama(st.session_state.chat_model)
    if ok:
        st.markdown(f'<span class="status-ok">● {status_msg}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-bad">● {status_msg}</span>', unsafe_allow_html=True)

    st.divider()

    st.subheader("📁 Documents")
    existing = known_sources()
    stats = st.session_state.retriever.stats()

    if existing:
        st.markdown(
            "".join(f'<span class="doc-pill">{name}</span>' for name in existing),
            unsafe_allow_html=True,
        )
        st.caption(f"**{stats['total_chunks']}** indexed chunks across **{len(existing)}** document(s)")

        with st.expander("Manage & Delete Documents", expanded=False):
            for name in existing:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{name}** ({stats['sources'].get(name, 0)} chunks)")
                with col2:
                    if st.button("🗑", key=f"del_{name}", help=f"Delete {name}"):
                        n = st.session_state.retriever.delete_by_source(name)
                        st.session_state.doc_summaries.pop(name, None)
                        st.success(f"Removed {n} chunks")
                        st.rerun()

        st.session_state.selected_sources = st.multiselect(
            "Filter Search Scope",
            options=existing,
            default=existing,
            help="Select specific files to constrain retrieval scope.",
        )
    else:
        st.info("No documents uploaded yet.")

    st.subheader("➕ Upload Document")
    uploaded_file = st.file_uploader(
        "Choose file",
        type=["txt", "pdf", "docx", "md", "json", "csv"],
        label_visibility="collapsed",
        help=f"Max size ~{MAX_UPLOAD_MB} MB",
    )

    if uploaded_file is not None and st.button("Process & Index Document", use_container_width=True, type="primary"):
        progress = st.progress(0, text="Extracting text...")
        try:
            segments = extract_text(uploaded_file)
        except Exception as e:
            progress.empty()
            st.error(f"Failed to process file: {e}")
            segments = None

        if not segments:
            progress.empty()
            st.error("Could not extract readable text from document.")
        else:
            chunks = chunk_document(segments)
            total = len(chunks) or 1
            full_text = " ".join(c["text"] for c in chunks)

            for i, chunk in enumerate(chunks):
                embedding = ollama.embeddings(
                    model=st.session_state.embed_model, prompt=chunk["text"]
                )["embedding"]
                collection.add(
                    ids=[f"{uploaded_file.name}_chunk_{i}"],
                    embeddings=[embedding],
                    documents=[chunk["text"]],
                    metadatas=[
                        {
                            "source": uploaded_file.name,
                            "chunk_index": i,
                            "page": chunk["page"] if chunk["page"] is not None else -1,
                        }
                    ],
                )
                progress.progress((i + 1) / total, text=f"Embedding chunk {i+1}/{total}...")

            st.session_state.retriever.refresh_bm25_index()
            progress.progress(1.0, text="Generating document summary...")
            summary = summarize_document(st.session_state.chat_model, full_text)
            st.session_state.doc_summaries[uploaded_file.name] = summary
            progress.empty()
            st.success(f"Successfully indexed {uploaded_file.name} ({len(chunks)} chunks)")
            st.rerun()

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.followups = []
            st.rerun()
    with c2:
        if st.button("Reset KB", use_container_width=True):
            data = collection.get()
            ids = data.get("ids") or []
            if ids:
                collection.delete(ids=ids)
                st.session_state.retriever.refresh_bm25_index()
                st.session_state.doc_summaries = {}
            st.session_state.messages = []
            st.session_state.followups = []
            st.rerun()


# ========== MAIN APP LAYOUT ==========
st.title("📄 Ask My Documents Pro")
st.markdown("Advanced Local RAG Architecture • Hybrid BM25 + Vector Search • Multi-Query Expansion • LLM-as-a-Judge Eval")

tab_chat, tab_eval, tab_inspector, tab_settings = st.tabs(
    ["💬 Chat Studio", "📊 RAG Analytics & Eval", "🧠 Vector Store Inspector", "⚙️ Engine Tuning"]
)

# ---------------------------------------------------------
# TAB 1: CHAT STUDIO
# ---------------------------------------------------------
with tab_chat:
    # Render chat message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("sources"):
                    st.markdown(
                        "".join(f'<span class="source-tag">📄 {name}</span>' for name in msg["sources"]),
                        unsafe_allow_html=True,
                    )
                if msg.get("latency_ms"):
                    st.caption(f"⚡ Response generated in {msg['latency_ms']} ms | Retrieval k={st.session_state.n_results}")

    # Empty State Guide
    if not known_sources():
        st.info("👋 **Welcome to Ask My Documents Pro!** Upload a PDF, DOCX, TXT, MD, JSON, or CSV file in the sidebar to begin asking questions.")
    elif not st.session_state.messages:
        st.info("💡 **Ready to chat:** Your document knowledge base is loaded. Ask any question below or pick a suggested query.")

    # Follow-up suggestion buttons
    if st.session_state.followups:
        st.markdown("**Suggested Follow-ups:**")
        cols = st.columns(min(3, len(st.session_state.followups)))
        for i, q in enumerate(st.session_state.followups):
            with cols[i % len(cols)]:
                if st.button(q, key=f"fu_{i}", use_container_width=True):
                    st.session_state._pending_question = q
                    st.session_state.followups = []
                    st.rerun()

    # Input box
    question = st.chat_input("Ask anything about your documents...")
    if "_pending_question" in st.session_state:
        question = st.session_state.pop("_pending_question")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            if is_small_talk(question):
                with st.spinner("Responding..."):
                    answer = small_talk_reply(st.session_state.chat_model, question)
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.followups = []

            elif not known_sources():
                answer = "No documents loaded yet — please upload a document first!"
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            else:
                t0 = time.time()
                sources_filter = st.session_state.selected_sources or None

                # Query rewriting step
                search_query = question
                if st.session_state.enable_query_rewrite and len(st.session_state.messages) > 1:
                    with st.spinner("Rewriting query for history context..."):
                        search_query = rewrite_query_with_history(
                            st.session_state.chat_model, question, st.session_state.messages[:-1]
                        )

                with st.spinner("Performing hybrid retrieval (BM25 + Chroma Vector RRF)..."):
                    chunks = st.session_state.retriever.search(
                        search_query,
                        n_results=st.session_state.n_results,
                        sources=sources_filter,
                        multi_query=st.session_state.enable_multi_query,
                        hyde=st.session_state.enable_hyde,
                        chat_model=st.session_state.chat_model,
                    )

                best_score = max((c["score"] for c in chunks), default=0)

                if best_score < MIN_RELEVANCE_SCORE or not chunks:
                    answer = NO_CONTEXT_MESSAGE
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.followups = []
                else:
                    answer_placeholder = st.empty()
                    answer = ""
                    history_for_model = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    for token in stream_answer(
                        st.session_state.chat_model,
                        question,
                        chunks,
                        history=history_for_model,
                    ):
                        answer += token
                        answer_placeholder.write(answer)

                    latency_ms = round((time.time() - t0) * 1000, 2)
                    sources_used = sorted({c["metadata"].get("source", "unknown") for c in chunks})

                    st.markdown(
                        "".join(f'<span class="source-tag">📄 {name}</span>' for name in sources_used),
                        unsafe_allow_html=True,
                    )
                    st.caption(f"⚡ Generated in {latency_ms} ms")

                    with st.expander("🔍 View Retrieved Context Excerpts & RRF Scores", expanded=False):
                        for idx_c, c in enumerate(chunks):
                            meta = c["metadata"]
                            page = meta.get("page", -1)
                            page_label = f", Page {page}" if page and page != -1 else ""
                            st.markdown(
                                f"**[{idx_c+1}] {meta.get('source', 'unknown')}{page_label}**"
                                f' <span class="score-pill">RRF Score: {c["score"]:.4f}</span>',
                                unsafe_allow_html=True,
                            )
                            st.write(c["document"])
                            st.divider()

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources_used,
                            "latency_ms": latency_ms,
                        }
                    )

                    with st.spinner("Generating follow-up suggestions..."):
                        st.session_state.followups = suggest_followups(
                            st.session_state.chat_model, question, answer, chunks
                        )
                    if st.session_state.followups:
                        st.rerun()

    if st.session_state.messages:
        st.divider()
        md_data = export_chat_markdown()
        st.download_button(
            "📥 Export Chat Transcript (.md)",
            data=md_data,
            file_name=f"rag_chat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
        )


# ---------------------------------------------------------
# TAB 2: RAG ANALYTICS & EVALUATION
# ---------------------------------------------------------
with tab_eval:
    st.subheader("📊 Offline RAG Evaluation Suite (RAGAS-style)")
    st.caption("Measure your pipeline quality locally using LLM-as-a-Judge (`llama3.2`) — zero data sent external.")

    eval_path = Path("eval_dataset.json")
    eval_cases = []
    if eval_path.exists():
        with open(eval_path) as f:
            eval_cases = json.load(f)

    st.write(f"Loaded **{len(eval_cases)}** evaluation test cases from `eval_dataset.json`.")

    with st.expander("View / Edit Test Cases", expanded=False):
        st.json(eval_cases)

    if st.button("🚀 Run Full Evaluation Benchmark", type="primary"):
        if not known_sources():
            st.warning("Please upload documents first so retrieval can be evaluated!")
        elif not eval_cases:
            st.error("No evaluation cases found in eval_dataset.json.")
        else:
            progress_bar = st.progress(0, text="Initializing evaluation harness...")

            def update_progress(curr, total, msg):
                progress_bar.progress(curr / total, text=msg)

            def eval_answer_fn(q, chunks):
                return "".join(stream_answer(st.session_state.chat_model, q, chunks))

            results = run_eval_suite(
                "eval_dataset.json",
                st.session_state.retriever,
                eval_answer_fn,
                judge_model=st.session_state.chat_model,
                progress_cb=update_progress,
            )
            st.session_state.last_eval_results = results
            progress_bar.empty()
            st.success("Benchmark completed successfully!")

    if st.session_state.last_eval_results:
        res = st.session_state.last_eval_results
        summary = summarize(res)

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Retrieval Hit Rate", f"{summary['retrieval_hit_rate']:.0%}")
        with c2:
            st.metric("Avg Faithfulness", f"{summary['avg_faithfulness']:.2f} / 1.0")
        with c3:
            st.metric("Avg Answer Relevance", f"{summary['avg_relevance']:.2f} / 1.0")
        with c4:
            st.metric("Avg Query Latency", f"{summary['avg_latency_ms']:.0f} ms")

        st.subheader("Detailed Evaluation Breakdown")
        for idx, r in enumerate(res):
            with st.expander(f"Q{idx+1}: {r['question']}", expanded=(idx == 0)):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Retrieval Hit:** {'✅ Yes' if r['retrieval_hit'] else '❌ No'}")
                    st.write(f"**Expected Source:** `{r['expected_source']}`")
                    st.write(f"**Latency:** {r['latency_ms']} ms")
                with col_b:
                    st.write(f"**Faithfulness:** `{r['faithfulness']:.2f}`")
                    st.caption(f"Reason: {r['faithfulness_reason']}")
                    st.write(f"**Relevance:** `{r['relevance']:.2f}`")
                    st.caption(f"Reason: {r['relevance_reason']}")

                st.write("**Generated Output:**")
                st.info(r["answer"])


# ---------------------------------------------------------
# TAB 3: VECTOR STORE & DOCUMENT INSPECTOR
# ---------------------------------------------------------
with tab_inspector:
    st.subheader("🧠 Vector Store & Chunk Explorer")
    st.caption("Inspect indexed ChromaDB chunks, embeddings status, and test similarity search directly.")

    all_chunks = st.session_state.retriever.get_all_chunks()
    st.write(f"Total Chunks in ChromaDB Collection: **{len(all_chunks)}**")

    st.subheader("🔎 Search Test Lab")
    test_query = st.text_input("Test retrieval query:", placeholder="Type a search query to test RRF scoring...")
    if test_query:
        test_hits = st.session_state.retriever.search(
            test_query,
            n_results=st.session_state.n_results,
            multi_query=st.session_state.enable_multi_query,
            hyde=st.session_state.enable_hyde,
            chat_model=st.session_state.chat_model,
        )
        st.write(f"Top {len(test_hits)} Retrieved Chunks:")
        for idx_h, hit in enumerate(test_hits):
            st.markdown(
                f"**Rank #{idx_h+1} — {hit['metadata'].get('source')}** "
                f'<span class="score-pill">Score: {hit["score"]:.4f}</span>',
                unsafe_allow_html=True,
            )
            st.code(hit["document"])

    st.divider()
    st.subheader("📋 All Indexed Document Chunks")
    if all_chunks:
        for idx_c, chunk in enumerate(all_chunks[:30]):
            meta = chunk["metadata"] or {}
            with st.expander(f"Chunk #{idx_c+1} — {meta.get('source', 'Unknown')} (ID: {chunk['id']})"):
                st.json(meta)
                st.text_area("Chunk Content", chunk["document"], height=100, key=f"chunk_text_{idx_c}")
        if len(all_chunks) > 30:
            st.caption(f"Showing first 30 of {len(all_chunks)} chunks.")
    else:
        st.caption("No chunks currently in database.")


# ---------------------------------------------------------
# TAB 4: ENGINE TUNING & CONFIGURATION
# ---------------------------------------------------------
with tab_settings:
    st.subheader("⚙️ Advanced RAG Hyperparameter Tuning")
    st.caption("Adjust pipeline parameters to optimize retrieval precision and response generation.")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.chat_model = st.text_input(
            "Ollama LLM Model",
            value=st.session_state.chat_model,
            help="Chat generation model (e.g. llama3.2, mistral, llama3, phi3).",
        )
        st.session_state.embed_model = st.text_input(
            "Ollama Embedding Model",
            value=st.session_state.embed_model,
            help="Embedding model for vector index (e.g. nomic-embed-text, mxbai-embed-large).",
        )
        st.session_state.n_results = st.slider(
            "Retrieval Top-K Chunks",
            min_value=2,
            max_value=15,
            value=st.session_state.n_results,
            help="Number of chunks passed into context window.",
        )

    with col2:
        st.session_state.enable_multi_query = st.toggle(
            "Multi-Query Expansion",
            value=st.session_state.enable_multi_query,
            help="Generates 2 query variations and fuses RRF results to increase search recall.",
        )
        st.session_state.enable_hyde = st.toggle(
            "HyDE (Hypothetical Document Embeddings)",
            value=st.session_state.enable_hyde,
            help="Generates a hypothetical passage before vector search.",
        )
        st.session_state.enable_query_rewrite = st.toggle(
            "Contextual Query Rewriting",
            value=st.session_state.enable_query_rewrite,
            help="Rewrites multi-turn user follow-up questions using prior chat history.",
        )

    st.success("Settings apply immediately to new queries.")
