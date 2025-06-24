import json
import faiss
import numpy as np
import os
from tqdm import tqdm
from openai import OpenAI

# === Setup OpenAI v1 client ===
import dotenv
dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Constants ===
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
INDEX_FILE      = "faiss.index"
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE   = "metadata.json"

# === Functions ===

def get_embedding(text: str, model: str = EMBEDDING_MODEL):
    try:
        resp = client.embeddings.create(input=[text], model=model)
        return resp.data[0].embedding
    except Exception as e:
        print("Fejl ved generering af embedding:", e)
        return None


def search(question: str, k: int = 5):
    # Indlæs FAISS-indeks og metadata
    index      = faiss.read_index(INDEX_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Beregn embedding for spørgsmålet
    query_vector = get_embedding(question)
    if query_vector is None:
        return []

    D, I = index.search(np.array([query_vector], dtype="float32"), k)
    result_chunks = []
    for idx in I[0]:
        if 0 <= idx < len(metadata):
            entry = metadata[idx]
            title = entry.get("title", "Uden titel")
            source = entry.get("source", "Ukendt kilde")
            text = entry.get("text", "").strip().replace("\n", " ").replace("\t", " ")
            result_chunks.append({"title": title, "source": source, "text": text})
    return result_chunks


def generate_answer(question: str, context_chunks: list):
    context = "\n".join([chunk["text"] for chunk in context_chunks])
    prompt = (
        "You are Karin from AlignerService. You are a helpful assistant with extensive experience in clear aligner support.\n"
        "Based on the following context, answer the user's question as informatively as possible. "
        "If the question requires clinical expertise beyond your scope, state that clearly and offer to involve a clinical advisor.\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4"),
            messages=[{"role": "user", "content": prompt}],
            temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.2)),
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Der opstod en fejl ved generering af svar: {e}"


# === CLI-interface ===
if __name__ == "__main__":
    question = input("Indtast dit spørgsmål: ")
    print(f"\n🔍 Søger: {question}\n")
    top_chunks = search(question)
    for i, chunk in enumerate(top_chunks, 1):
        print(f"{i}. {chunk['title']} [{chunk['source']}]")
    print("\n")
    answer = generate_answer(question, top_chunks)
    print("💬 Svar:\n", answer)
