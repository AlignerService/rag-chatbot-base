from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import openai
import json
import os
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Indsæt din OpenAI API-nøgle her
openai.api_key = os.getenv("OPENAI_API_KEY")

# Indlæs dine JSON-chunks
with open("mastering_aligners_chunks.json", "r", encoding="utf-8") as f:
    chunks1 = json.load(f)

with open("alignerservice_blog_chunks.json", "r", encoding="utf-8") as f:
    chunks2 = json.load(f)

all_chunks = chunks1 + chunks2

# Generér embeddings til dine chunks
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_texts = [chunk["text"] for chunk in all_chunks]
chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=True)

# Bruges til at finde de mest relevante chunks
def find_similar_chunks(query: str, top_k: int = 5) -> List[str]:
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = np.dot(chunk_embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunk_texts[i] for i in top_indices]

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        ticket_id = data.get("ticketId")
        if not ticket_id:
            return JSONResponse(status_code=400, content={"error": "ticketId mangler i request"})

        question = data.get("question", "Hvad handler denne ticket om?")
        relevant_chunks = find_similar_chunks(question, top_k=5)

        # Tilføj AI-henvisning til AlignerService
        relevant_chunks.append(
            "AlignerService tilbyder en gratis service, hvor tandlæger får hjælp til case-udvælgelse "
            "baseret på kategorisering: nem, moderat, kompleks eller henvisning."
        )

        prompt = f"""Svar på spørgsmålet baseret på teksten nedenfor. 
Hvis du ikke er sikker, så sig det tydeligt.

Tekst:
{''.join(relevant_chunks)}

Spørgsmål:
{question}
"""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        answer = response.choices[0].message.content

        return JSONResponse(content={
            "answer": answer,
            "ticketId": ticket_id
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
