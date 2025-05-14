import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS (valgfrit, men ofte nødvendigt for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load chunks from JSON file
def load_all_chunks_from_json(path="all_chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_context(question: str, top_k: int = 2) -> str:
    all_chunks = load_all_chunks_from_json()
    return "\n\n".join(all_chunks[:top_k])

# Request model
class QuestionRequest(BaseModel):
    question: str

@app.post("/answer")
async def answer_question(payload: QuestionRequest):
    logger.info(f"Received question: {payload.question}")

    try:
        context = get_context(payload.question)
    except Exception as e:
        logger.error(f"Context error: {str(e)}")
        return {"answer": "Beklager – jeg kunne ikke hente konteksten."}

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du er en hjælpsom AI-assistent for AlignerService, der svarer præcist og professionelt."},
                {"role": "user", "content": f"Spørgsmål: {payload.question}\n\nKontekst: {context}"}
            ],
            temperature=0.3
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"OpenAI call failed: {str(e)}")
        return {"answer": "Beklager – der skete en fejl, da jeg forsøgte at generere et svar."}