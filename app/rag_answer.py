# app/rag_answer.py
import os
import json
import re
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
    Hvis ikke, lader vi det passere.
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
    Ultra-simple version. Finder et muligt case-id (6-10 cifre)
    og brand-navn hvis ordet står i teksten.
    """
    blob = f"{subject or ''}\n{body or ''}".lower()
    brand = None
    for name in ["invisalign", "spark", "angel", "clearcorrect", "suresmile", "trioclear", "clarity"]:
        if name in blob:
            brand = name
            break
    m = re.search(r"\b\d{6,10}\b", blob)
    case_id = m.group(0) if m else None
    return {"brand": brand, "caseId": case_id}

# --- Autosvar-filter + tråd-udtræk (Zoho) ---
AUTOREPLY_MARKERS = [
    "Thank you for your message – we appreciate your trust in AlignerService.",
    "We’ve received your request, and the team has already begun processing it.",
    "you'll hear from us as soon as your setup is ready",
]
OUR_DOMAIN_HINTS = ["@alignerservice.com"]
OUR_SENDER_HINTS = ["tps@alignerservice.com", "support@alignerservice.com"]

def _looks_like_autoreply(txt: str) -> bool:
    t = (txt or "").strip().lower()
    if not t:
        return False
    return any(m.lower() in t for m in AUTOREPLY_MARKERS)

def _is_our_sender(val: str) -> bool:
    v = (val or "").strip().lower()
    if not v:
        return False
    if v in (s.lower() for s in OUR_SENDER_HINTS):
        return True
    return any(d in v for d in (h.lower() for h in OUR_DOMAIN_HINTS))

def _pick_inbound_from_payload(body: dict) -> str:
    """
    Find seneste INBOUND kundebesked i typiske Zoho-felter.
    Ignorer vores egne udsendte autosvar.
    """
    if not isinstance(body, dict):
        return ""

    list_keys = ["ticketThreadMessages", "messages", "thread", "comments", "conversation", "items"]

    for lk in list_keys:
        msgs = body.get(lk)
        if not isinstance(msgs, list) or not msgs:
            continue

        # seneste først
        for m in reversed(msgs):
            if not isinstance(m, dict):
                continue
            txt = None
            for tk in ["plainText", "plaintext", "text", "content", "body", "message", "message_text"]:
                if isinstance(m.get(tk), str) and m[tk].strip():
                    txt = m[tk].strip()
                    break
            if not txt:
                continue

            direction = (m.get("direction") or m.get("dir") or m.get("type") or "").strip().lower()
            sender = (m.get("from") or m.get("sender") or m.get("author") or "").strip().lower()

            if _is_our_sender(sender):
                continue
            if _looks_like_autoreply(txt):
                continue
            if direction and direction not in ("in", "inbound", "received", "customer", "incoming"):
                continue

            return txt  # god inbound-tekst

    # fallback i roden
    for k in ["customerMessage", "originalMessage", "ticketPlainText", "lastInbound"]:
        v = body.get(k)
        if isinstance(v, str) and v.strip() and not _looks_like_autoreply(v):
            return v.strip()

    return ""

# ---------------------------
# 3) Selve endpointet
# ---------------------------
@router.post("/answer", dependencies=[Depends(_require_token)])
async def api_answer(request: Request):
    """
    Frontend kalder via fetch('/api/answer').
    Vi læser body, trækker kundens seneste INBOUND-tekst hvis muligt,
    falder ellers tilbage til 'question'/'text'/... og kører din RAG.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Læs inputfelter på en venlig måde
    question = ""

    # 2A: Prøv først at finde seneste INBOUND fra kunden (Zoho tråd)
    inbound_txt = _pick_inbound_from_payload(body) if isinstance(body, dict) else ""
    if inbound_txt:
        question = inbound_txt

    # 2B: Hvis ikke, brug klassiske felter
    if not question and isinstance(body, dict):
        for key in ["question", "text", "message", "body", "plainText"]:
            if isinstance(body.get(key), str) and body.get(key).strip():
                question = body[key].strip()
                break

    # 2C: Hvis det ligner vores autosvar, forsøg igen at finde inbound
    if question and _looks_like_autoreply(question):
        inbound_again = _pick_inbound_from_payload(body) if isinstance(body, dict) else ""
        if inbound_again:
            question = inbound_again

    if not question:
        # sidste udvej: ekko hele body
        question = json.dumps(body, ensure_ascii=False)

    # Bonus: metadata til svaret
    from_email = (body.get("fromEmail") or "").strip().lower() if isinstance(body, dict) else ""
    subject = (body.get("subject") or "").strip() if isinstance(body, dict) else ""
    brand_case = _extract_brand_and_case(subject, question)
    lang = _detect_language(question)

    # Kald din eksisterende funktion
    try:
        answer = get_rag_answer(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG failed: {e}")

    # Returnér i et UI-venligt format
    return {
        "finalAnswer": answer,
        "finalAnswerMarkdown": answer,
        "finalAnswerPlain": answer,
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
