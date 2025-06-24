# build_index.py

import os, json, csv, numpy as np, faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

METADATA_PATH   = "metadata.json"
INDEX_PATH      = "faiss.index"
MAPPING_CSV     = "id_mapping.csv"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

def main():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    vectors = []
    ids     = []

    for i, chunk in enumerate(metadata):
        text = chunk.get("text", "").strip()
        if not text:
            continue
        resp = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        emb  = resp.data[0].embedding
        vectors.append(emb)

        # Brug 'id' hvis den findes, ellers 'chunk_id', ellers fallback
        id_val = chunk.get("id") or chunk.get("chunk_id") or f"chunk_{i}"
        ids.append(id_val)

    vectors = np.array(vectors, dtype="float32")
    dim     = vectors.shape[1]

    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    int_ids = np.arange(len(ids), dtype="int64")
    index.add_with_ids(vectors, int_ids)

    faiss.write_index(index, INDEX_PATH)
    with open(MAPPING_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["faiss_id", "chunk_id"])
        for fid, cid in zip(int_ids, ids):
            writer.writerow([int(fid), cid])

    print(f"✅ Gemt {INDEX_PATH} og {MAPPING_CSV}")

if __name__ == "__main__":
    main()
