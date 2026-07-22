import ollama
import chromadb

# Connect to the same Chroma database we built
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_documents")

print("Ask me anything about the document! Type 'quit' to exit.\n")

while True:
    question = input("You: ")
    
    if question.lower() == "quit":
        print("Goodbye!")
        break
    
    # Step 1: embed the question the same way we embedded chunks
    question_embedding = ollama.embeddings(model='nomic-embed-text', prompt=question)['embedding']
    
    # Step 2: find the most similar chunks in the database
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )
    
    retrieved_chunks = results['documents'][0]
    context = "\n\n".join(retrieved_chunks)
    
    # Step 3: build a prompt that includes the retrieved context
    prompt = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}"""
    
    # Step 4: send it to the chat model
    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'user', 'content': prompt}
    ])
    
    print(f"\nBot: {response['message']['content']}\n")