import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

ZOHO_ACCESS_TOKEN = "1000.f6ecbcfad9fe90b41550ae89ac4cc4b1.9a26d1dbc66f267b0fcb172551cfbe48"
ZOHO_PORTAL = "alignerservice"
RAG_API_URL = "https://alignerservice-rag.onrender.com/chat"

@app.get("/")
def health():
    return {"status": "Proxy server kører."}

@app.post("/chat")
async def chat_proxy(request: Request):
    try:
        raw = await request.body()
        print("RAW BODY MODTAGET:", raw)

        data = await request.json()
        ticket_id = data.get("ticketId")
        print("✅ Parsed JSON:", data)

        if not ticket_id:
            return JSONResponse(status_code=400, content={"error": "ticketId mangler"})

        # 1. Hent ticket-beskrivelse fra ZoHo Desk
        ticket_url = f"https://desk.zoho.eu/api/v1/tickets/{ticket_id}"
        headers = {
            "Authorization": f"Zoho-oauthtoken {ZOHO_ACCESS_TOKEN}"
        }
        r = requests.get(ticket_url, headers=headers)
        if r.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Kunne ikke hente ticket fra ZoHo", "details": r.text})
        
        description = r.json().get("description", "No ticket description provided.")

        # 2. Kald RAG API med beskrivelsen
        rag_payload = {
            "question": description
        }
        rag_headers = {
            "Content-Type": "application/json"
        }
        rag_response = requests.post(RAG_API_URL, json=rag_payload, headers=rag_headers)
        if rag_response.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Fejl fra RAG API", "details": rag_response.text})

        answer = rag_response.json().get("answer", "[Tomt svar]")

        # 3. Tilføj AI-svar som kommentar i ZoHo
        comment_url = f"https://desk.zoho.eu/api/v1/tickets/{ticket_id}/comments"
        comment_payload = {
            "is_public": False,
            "content": answer
        }
        comment_resp = requests.post(comment_url, json=comment_payload, headers=headers)

        if comment_resp.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Kunne ikke tilføje kommentar", "details": comment_resp.text})
        
        return JSONResponse(status_code=200, content={"status": "OK", "answer": answer})
    
    except Exception as e:
        print("❌ FEJL:", str(e))
        return JSONResponse(status_code=500, content={"error": "Fejl i proxy", "details": str(e)})