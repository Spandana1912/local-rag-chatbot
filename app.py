import streamlit as st
import ollama
import chromadb
from pypdf import PdfReader
from docx import Document
import io

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_documents")

st.title("📄 Ask My Documents")

# --- File upload section ---
st.sidebar.header("Upload a document")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt", "pdf", "docx"])

def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]

def extract_text(uploaded_file):
    """Reads a .txt, .pdf, or .docx file and returns its plain text content."""
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "txt":
        return uploaded_file.read().decode("utf-8")

    elif file_type == "pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    elif file_type == "docx":
        doc = Document(io.BytesIO(uploaded_file.read()))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    else:
        return None

if uploaded_file is not None:
    if st.sidebar.button("Process this document"):
        with st.sidebar.status("Processing document...", expanded=True) as status:
            text = extract_text(uploaded_file)

            if not text or not text.strip():
                st.error("Couldn't extract any text from this file.")
            else:
                chunks = chunk_text(text)
                st.write(f"Split into {len(chunks)} chunks")

                for i, chunk in enumerate(chunks):
                    embedding = ollama.embeddings(model='nomic-embed-text', prompt=chunk)['embedding']
                    collection.add(
                        ids=[f"{uploaded_file.name}_chunk_{i}"],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{"source": uploaded_file.name, "chunk_index": i}]
                    )
                    st.write(f"Stored chunk {i}")

                status.update(label="Done! Document added.", state="complete")

# --- Chat section ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

def is_small_talk(text):
    small_talk_phrases = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye", "ok", "okay", "cool", "nice"]
    cleaned = text.lower().strip().strip("!.,")
    return cleaned in small_talk_phrases

def get_source_name(meta):
    if meta and "source" in meta:
        return meta["source"]
    return "unknown document"

def get_chunk_index(meta):
    if meta and "chunk_index" in meta:
        return meta["chunk_index"]
    return "?"

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if is_small_talk(question):
            with st.spinner("..."):
                response = ollama.chat(model='llama3.2', messages=[
                    {'role': 'system', 'content': 'You are a friendly assistant. Respond briefly and naturally to casual conversation.'},
                    {'role': 'user', 'content': question}
                ])
            answer = response['message']['content']
            st.write(answer)

        else:
            with st.spinner("Thinking..."):
                question_embedding = ollama.embeddings(model='nomic-embed-text', prompt=question)['embedding']
                results = collection.query(query_embeddings=[question_embedding], n_results=3)

                retrieved_chunks = results['documents'][0]
                retrieved_metadata = results['metadatas'][0]
                context = "\n\n".join(retrieved_chunks)

                prompt = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}"""

                response = ollama.chat(model='llama3.2', messages=[
                    {'role': 'user', 'content': prompt}
                ])

            answer = response['message']['content']
            st.write(answer)

            sources_used = sorted(set(get_source_name(meta) for meta in retrieved_metadata))
            st.caption(f"📌 Sources: {', '.join(sources_used)}")

            with st.expander("See retrieved context"):
                for chunk, meta in zip(retrieved_chunks, retrieved_metadata):
                    st.markdown(f"**From `{get_source_name(meta)}`, chunk {get_chunk_index(meta)}:**")
                    st.write(chunk)
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})