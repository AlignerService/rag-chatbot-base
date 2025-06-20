import json
import numpy as np
import faiss
import openai
import os
import tiktoken

# Indlæs OpenAI API-nøgle fra miljøvariabel
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load chunks og embeddings
with open("all_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss.index")

# Tokenizer til længdecheck
tokenizer = tiktoken.encoding_for_model("gpt-4")

def num_tokens(text):
    return len(tokenizer.encode(text))

def get_rag_answer(question, top_k=5):
    # Embed spørgsmålet
    response = openai.Embedding.create(
        input=[question],
        model="text-embedding-3-small"
    )
    question_embedding = np.array(response["data"][0]["embedding"], dtype=np.float32).reshape(1, -1)

    # Find de mest relevante chunks
    scores, indices = index.search(question_embedding, top_k)
    context_chunks = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            context_chunks.append(chunks[idx]["text"])

    # Filtrér tokens så prompten ikke bliver for lang
    max_tokens = 3000
    current_tokens = 0
    selected_chunks = []
    for chunk in context_chunks:
        chunk_tokens = num_tokens(chunk)
        if current_tokens + chunk_tokens < max_tokens:
            selected_chunks.append(chunk)
            current_tokens += chunk_tokens
        else:
            break

    # Skriv prompt
    prompt = (
        "You are an assistant for a dental company. Answer based only on the following context:

"
        + "

---

".join(selected_chunks)
        + f"

---

Question: {question}
Answer:"
    )

    # Send til GPT-4
    chat_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return chat_response["choices"][0]["message"]["content"]
