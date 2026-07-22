import ollama
import chromadb

# Read the document
with open("documents/doc1.txt", "r", encoding="utf-8") as file:
    full_text = file.read()

def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]

chunks = chunk_text(full_text)
print(f"Split into {len(chunks)} chunks.")

# Set up Chroma - this creates a local folder to store the database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_documents")

# Generate an embedding for each chunk and store it
for i, chunk in enumerate(chunks):
    embedding_response = ollama.embeddings(model='nomic-embed-text', prompt=chunk)
    embedding = embedding_response['embedding']
    
    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[embedding],
        documents=[chunk]
    )
    print(f"Stored chunk {i} ({len(chunk)} chars)")

print("\nDone! All chunks embedded and stored in ./chroma_db")