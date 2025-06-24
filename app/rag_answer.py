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

# --- Load chunks and index ---
with open("all_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
# Note: embeddings.npy no longer needed here if index stores vectors
index = faiss.read_index("faiss.index")

# --- Tokenizer ---
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def get_rag_answer(question: str, top_k: int = 5) -> str:
    # 1) Embed question
    resp = client.embeddings.create(input=[question], model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"))
    q_emb = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)

    # 2) Search FAISS
    D, I = index.search(q_emb, top_k)
    context_chunks = [chunks[idx]["text"] for idx in I[0] if 0 <= idx < len(chunks)]

    # 3) Trim tokens
    max_tokens = 3000
    selected = []
    used = 0
    for c in context_chunks:
        ct = num_tokens(c)
        if used + ct <= max_tokens:
            selected.append(c)
            used += ct
        else:
            break

    # 4) Build prompt
    context_text = "\n---\n".join(selected)
    prompt = (
        "Du er tandlæge Helle Hatt fra AlignerService, en erfaren klinisk rådgiver.\n"
        "Svar baseret på følgende kontekst:\n"
        f"{context_text}\n\n"
        f"Spørgsmål: {question}\n"
        "Svar:"
    )

    # 5) Chat completion
    chat = client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4"),
        messages=[{"role":"user","content":prompt}],
        temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.2)),
        max_tokens=300
    )
    return chat.choices[0].message.content.strip()


if __name__ == "__main__":
    q = input("Indtast dit spørgsmål: ")
    print(get_rag_answer(q))
