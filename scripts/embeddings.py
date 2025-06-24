import os
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_FILE    = "faiss.index"
META_FILE     = "metadata.json"
EMBED_MODEL   = "text-embedding-ada-002"

# Load FAISS index & metadata
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

def make_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(input=text, model=EMBED_MODEL)
    return np.array(resp.data[0].embedding, dtype="float32")

def search(query: str, top_k: int = 5):
    vec = make_embedding(query)
    D, I = index.search(vec.reshape(1, -1), top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        chunk = metadata[idx]
        results.append({"score": float(dist), **chunk})
    return results
