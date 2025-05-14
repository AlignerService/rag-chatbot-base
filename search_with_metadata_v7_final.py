
import json
import faiss
import numpy as np
import openai
from tqdm import tqdm
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-ada-002"
INDEX_FILE = "faiss.index"
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.json"

def get_embedding(text, model=EMBEDDING_MODEL):
    try:
        response = openai.Embedding.create(input=[text], model=model)
        return response["data"][0]["embedding"]
    except Exception as e:
        print("Fejl ved generering af embedding:", e)
        return None

def search(question, k=5):
    # Indlæs FAISS-indeks, embeddings og metadata
    index = faiss.read_index(INDEX_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    # Beregn embedding for spørgsmålet
    query_vector = get_embedding(question)
    if query_vector is None:
        return []

    D, I = index.search(np.array([query_vector]), k)
    result_chunks = []
    for idx in I[0]:
        if idx < len(metadata):
            entry = metadata[idx]
            source = entry.get("source", "Ukendt kilde")
            title = entry.get("title", "Uden titel")
            text = entry.get("text", "").strip().replace("\n", " ").replace("\t", " ")
            result_chunks.append({"title": title, "source": source, "text": text})
    return result_chunks

def generate_answer(question, context_chunks):
    context = "\n".join([f"{chunk['text']}" for chunk in context_chunks])
    prompt = (
        "You are Karin from AlignerService. You are a helpful assistant with extensive experience in clear aligner support.\n"
        "Based on the following context, answer the user's question as informatively as possible. "
        "If the question requires clinical expertise beyond your scope, state that clearly and offer to involve a clinical advisor.\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Der opstod en fejl ved generering af svar: {e}"

if __name__ == "__main__":
    question = input("Indtast dit spørgsmål: ")
    print(f"\n🔍 Søger: {question}\n")
    top_chunks = search(question)
    for i, chunk in enumerate(top_chunks, 1):
        print(f"{i}. {chunk['title']} [{chunk['source']}]\n...")
    print("\n")
    answer = generate_answer(question, top_chunks)
    print("💬 Svar:\n", answer)
