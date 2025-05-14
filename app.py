
from fastapi import FastAPI, Request
import subprocess

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "RAG API is live."}

@app.post("/update")
async def update_knowledge(request: Request):
    try:
        # Kør update_knowledge_db.py for at indsætte ny ticket i databasen
        result = subprocess.run(["python3", "update_knowledge_db.py"], capture_output=True, text=True)
        return {
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
