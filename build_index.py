# build_index.py

import os
import json
import csv
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# ===== Load API-nøgle =====
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== Stier & model =====
METADATA_PATH   = "metadata.json"
INDEX_PATH      = "faiss.index"
MAPPING_CSV     = "id_mapping.csv"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

def main():
    # 1) Indlæs metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 2) Beregn embeddings
    vectors, ids = [], []
    for chunk in metadata:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        resp = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        emb  = resp.data[0].embedding
        vectors.append(emb)
        ids.append(chunk["id"])

    vectors = np.array(vectors, dtype="float32")
    dim     = vectors.shape[1]

    # 3) Byg FAISS-index
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    int_ids = np.arange(len(ids), dtype="int64")
    index.add_with_ids(vectors, int_ids)

    # 4) Gem index & mapping
    faiss.write_index(index, INDEX_PATH)
    with open(MAPPING_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["faiss_id", "chunk_id"])
        for fid, cid in zip(int_ids, ids):
            writer.writerow([int(fid), cid])

    print(f"✅ Gemt {INDEX_PATH} og {MAPPING_CSV}")

if __name__ == "__main__":
    main()
