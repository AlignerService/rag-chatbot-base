
import os
import sqlite3
import openai
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI()

# OpenAI klient til ny API syntaks
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TicketRequest(BaseModel):
    ticketId: str
    question: str

def query_database(question: str):
    conn = sqlite3.connect("alignerservice_chunks.db")
    cursor = conn.cursor()

    cursor.execute("SELECT content FROM documents WHERE content LIKE ?", ('%' + question[:20] + '%',))
    results = cursor.fetchall()
    conn.close()

    return [row[0] for row in results]

@app.post("/answer")
async def generate_answer(request: Request, payload: TicketRequest):
    try:
        context_chunks = query_database(payload.question)
        context = "\n\n".join(context_chunks[:5])  # max 5 relevante stykker

        messages = [
            {"role": "system", "content": "Du er en hjælpsom AI-assistent for AlignerService, der besvarer kundeservicehenvendelser professionelt og præcist."},
            {"role": "user", "content": f"Spørgsmål: {payload.question}\n\nRelevant kontekst:
{context}"}
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        final_answer = response.choices[0].message.content.strip()
        return {"reply": final_answer}

    except Exception as e:
        return JSONResponse(status_code=500, content={"reply": f"Fejl ved OpenAI-kald: {str(e)}"})
