# build_index.py

import os
import json
import csv
import numpy as np
import faiss
import openai
from dotenv import load_dotenv

# ===== KONFIGURATION =====
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
METADATA_PATH   = "metadata.json"
INDEX_PATH      = "faiss.index"
MAPPING_CSV     = "id_mapping.csv"
EMBEDDING_MODEL = "text-embedding-ada-002"

# ===== 1) Load metadata =====
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ===== 2) Beregn embeddings =====
vectors = []
ids     = []
for chunk in metadata:
    text = chunk.get("text", "").strip()
    if not text:
        continue
    resp = openai.Embedding.create(input=text, model=EMBEDDING_MODEL)
    emb  = resp["data"][0]["embedding"]
    vectors.append(emb)
    ids.append(chunk["id"])

vectors = np.array(vectors, dtype="float32")
dim     = vectors.shape[1]

# ===== 3) Byg FAISS-index =====
index = faiss.IndexFlatL2(dim)
index = faiss.IndexIDMap(index)
int_ids = np.arange(len(ids), dtype="int64")
index.add_with_ids(vectors, int_ids)

# ===== 4) Gem index og mapping =====
faiss.write_index(index, INDEX_PATH)
with open(MAPPING_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["faiss_id", "chunk_id"])
    for fid, cid in zip(int_ids, ids):
        writer.writerow([int(fid), cid])

print(f"✅ Gemt {INDEX_PATH} og {MAPPING_CSV}")
