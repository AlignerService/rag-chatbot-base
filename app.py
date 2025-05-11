from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import openai
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key = os.getenv("OPENAI_API_KEY")

with open("all_chunks.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

texts = [doc["text"] for doc in documents]
meta = [(doc.get("title", ""), doc.get("source", doc.get("url", ""))) for doc in documents]

vectorizer = TfidfVectorizer().fit(texts)
vectors = vectorizer.transform(texts)

app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "")

    question_vec = vectorizer.transform([question])
    sims = cosine_similarity(question_vec, vectors).flatten()

    top_n = sims.argsort()[-3:][::-1]
    context = "\n---\n".join([texts[i] for i in top_n])

    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700
    )

    return JSONResponse({"answer": response.choices[0].message["content"].strip()})