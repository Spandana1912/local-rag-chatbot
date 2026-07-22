with open("documents/doc1.txt", "r", encoding="utf-8") as file:
    full_text = file.read()

print(full_text)
print(f"\n--- Total characters: {len(full_text)} ---")