# app/rag_answer.py
import os
import json
import numpy as np
import faiss
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

# FastAPI-ting til endpointet
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hmac

# ---------------------------
# 0) Router + valgfri token
# ---------------------------
router = APIRouter()
bearer = HTTPBearer(auto_error=True)
RAG_BEARER_TOKEN = os.getenv("RAG_BEARER_TOKEN", "")

def _require_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    """
    Hvis RAG_BEARER_TOKEN er sat, kræver vi en Bearer <token> header.
    Hvis ikke, lader vi det passere. Simpelt, uden drama.
    """
    expected = (RAG_BEARER_TOKEN or "").strip()
    if not expected:
        return True  # ingen token sat -> ingen krav
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=403, detail="Invalid auth scheme")
    incoming = (credentials.credentials or "").strip()
    if not hmac.compare_digest(incoming, expected):
        raise HTTPException(status_code=403, detail="Invalid or missing token")
    return True

# ---------------------------
# 1) Dit eksisterende setup
# ---------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Load chunks and index ---
with open("all_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
index = faiss.read_index("faiss.index")

# --- Tokenizer ---
tokenizer = tiktoken.get_encoding("cl100k_base")
def num_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def get_rag_answer(question: str, top_k: int = 5) -> str:
    # 1) Embed question
    resp = client.embeddings.create(
        input=[question],
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
    )
    q_emb = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)

    # 2) Search FAISS
    D, I = index.search(q_emb, top_k)
    context_chunks = [chunks[idx]["text"] for idx in I[0] if 0 <= idx < len(chunks)]

    # 3) Trim tokens
    max_tokens = 3000
    selected, used = [], 0
    for c in context_chunks:
        ct = num_tokens(c)
        if used + ct <= max_tokens:
            selected.append(c)
            used += ct
        else:
            break

    # 4) Build prompt
    context_text = "\n---\n".join(selected)
    prompt = (
        "Du er tandlæge Helle Hatt fra AlignerService, en erfaren klinisk rådgiver.\n"
        "Svar baseret på følgende kontekst:\n"
        f"{context_text}\n\n"
        f"Spørgsmål: {question}\n"
        "Svar:"
    )

    # 5) Chat completion
    chat = client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4"),
        messages=[{"role": "user", "content": prompt}],
        temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.2)),
        max_tokens=300,
    )
    return chat.choices[0].message.content.strip()

# ---------------------------
# 2) Små hjælper-funktioner
#    (helt ufarlige “hooks”)
# ---------------------------
def _detect_language(text: str) -> str:
    if not text:
        return "en"
    t = text.lower()
    if any(ch in t for ch in ["æ", "ø", "å"]) or " ikke" in t or " tak" in t:
        return "da"
    if any(ch in t for ch in ["ä", "ö", "ü", "ß"]) or " bitte" in t:
        return "de"
    if any(ch in t for ch in ["à", "â", "ç", "é", "è", "ê", "ë"]):
        return "fr"
    return "en"

def _extract_brand_and_case(subject: str, body: str):
    """
    Ultra-simple version for nu. Finder bare et muligt sags-/case-id (tal på 6-10 cifre)
    og brand-navn hvis ordet står i teksten. Ingen garanti, men det lynes ikke i luften.
    """
    blob = f"{subject or ''}\n{body or ''}".lower()
    brand = None
    for name in ["invisalign", "spark", "angel", "clearcorrect", "suresmile", "trioclear", "clarity"]:
        if name in blob:
            brand = name
            break
    import re
    m = re.search(r"\b\d{6,10}\b", blob)
    case_id = m.group(0) if m else None
    return {"brand": brand, "caseId": case_id}

# ---------------------------
# 3) Selve endpointet
# ---------------------------
@router.post("/answer", dependencies=[Depends(_require_token)])
async def api_answer(request: Request):
    """
    Dette er det, din frontend kalder via fetch('/api/answer').
    Vi læser body, trækker 'question' (fallback til 'text'/'message'), kører din eksisterende RAG,
    og returnerer et pænt JSON-svar.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Læs inputfelter på en venlig måde
    question = ""
    if isinstance(body, dict):
        for key in ["question", "text", "message", "body", "plainText"]:
            if isinstance(body.get(key), str) and body.get(key).strip():
                question = body[key].strip()
                break

    if not question:
        # hvis folk har sendt noget gakket, så ekko det mindste
        question = json.dumps(body, ensure_ascii=False)

    # Bonus: træk fromEmail/subject (bruger vi lige nu kun til metadata)
    from_email = (body.get("fromEmail") or "").strip().lower() if isinstance(body, dict) else ""
    subject = (body.get("subject") or "").strip() if isinstance(body, dict) else ""
    brand_case = _extract_brand_and_case(subject, question)
    lang = _detect_language(question)

    # Kald din eksisterende funktion
    try:
        answer = get_rag_answer(question)
    except Exception as e:
        # Hvis noget går i stykker, så sig det rent ud
        raise HTTPException(status_code=500, detail=f"RAG failed: {e}")

    # Returnér i et format som UI’er plejer at kunne bruge
    return {
        "finalAnswer": answer,                  # det du vil vise i UI
        "finalAnswerMarkdown": answer,          # samme, hvis UI vil vise markdown
        "finalAnswerPlain": answer,             # plain tekst (du kører i forvejen ret plain)
        "language": lang,
        "brandCase": brand_case,
        "fromEmail": from_email,
        "subject": subject,
    }

# ---------------------------
# 4) Fortsat CLI-support
# ---------------------------
if __name__ == "__main__":
    q = input("Indtast dit spørgsmål: ")
    print(get_rag_answer(q))
