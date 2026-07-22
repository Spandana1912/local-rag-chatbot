import ollama

response = ollama.chat(model='llama3.2', messages=[
    {'role': 'user', 'content': 'Say hello and tell me one fun fact about AI.'}
])

print(response['message']['content'])