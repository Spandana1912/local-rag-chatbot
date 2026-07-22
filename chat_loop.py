import ollama

# This list holds the entire conversation history
messages = []

print("Chat started! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    
    # Add the user's message to history
    messages.append({'role': 'user', 'content': user_input})
    
    # Send the FULL history so far to the model
    response = ollama.chat(model='llama3.2', messages=messages)
    
    reply = response['message']['content']
    print(f"Bot: {reply}\n")
    
    # Add the model's reply to history too, so it remembers what IT said
    messages.append({'role': 'assistant', 'content': reply})