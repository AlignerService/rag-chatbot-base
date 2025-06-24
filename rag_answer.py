import os
import json
import numpy as np
import faiss
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

# --- Load environment and initialize OpenAI client ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Load chunks, embeddings, and index ---
with open("all_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss.index")

# --- Tokenizer ---
tokenizer = tiktoken.get_encoding("cl100k_base")  # or encoding_for_model

def num_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def get_rag_answer(question: str, top_k: int = 5) -> str:
    # 1) Get embedding for question
    resp = client.embeddings.create(input=[question], model="text-embedding-3-small")
    question_embedding = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)

    # 2) Search FAISS
    D, I = index.search(question_embedding, top_k)
    context_chunks = [chunks[idx]["text"] for idx in I[0] if 0 <= idx < len(chunks)]

    # 3) Trim to max tokens
    max_tokens = 3000
    current_tokens = 0
    selected = []
    for chunk in context_chunks:
        ctoks = num_tokens(chunk)
        if current_tokens + ctoks <= max_tokens:
            selected.append(chunk)
            current_tokens += ctoks
        else:
            break

    # 4) Build prompt
    context_text = "\n---\n".join(selected)
    prompt = (
        "You are an assistant for a dental company. Answer based only on the following context:\n"
        f"{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    # 5) Get chat completion
    chat_resp = client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return chat_resp.choices[0].message.content.strip()


if __name__ == "__main__":
    q = input("Enter your question: ")
    answer = get_rag_answer(q)
    print("Answer:\n", answer)
