from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, validator
import json
import faiss
import numpy as np
from openai import OpenAI
import os
import sqlite3
import tiktoken
from datetime import datetime
import html
import logging
from contextlib import contextmanager
from typing import List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
class Config:
    INDEX_FILE = "faiss.index"
    EMBEDDINGS_FILE = "embeddings.npy"
    CHUNKS_FILE = "all_chunks.json"
    DB_PATH = os.getenv("KNOWLEDGE_DB", "knowledge.sqlite")
    MAX_TOKENS = 3000
    MAX_CHUNKS = 5
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-4"

# Global variables
tokenizer = tiktoken.get_encoding("cl100k_base")
chunks = []
index = None

def initialize_resources():
    """Initialize FAISS index and chunks safely"""
    global chunks, index
    
    try:
        # Load chunks
        if not os.path.exists(Config.CHUNKS_FILE):
            raise FileNotFoundError(f"Chunks file not found: {Config.CHUNKS_FILE}")
        
        with open(Config.CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        # Load FAISS index
        if not os.path.exists(Config.INDEX_FILE):
            raise FileNotFoundError(f"FAISS index not found: {Config.INDEX_FILE}")
        
        index = faiss.read_index(Config.INDEX_FILE)
        logger.info(f"Loaded {len(chunks)} chunks and FAISS index")
        
    except Exception as e:
        logger.error(f"Failed to initialize resources: {e}")
        raise

# Initialize on startup
initialize_resources()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def num_tokens(text: str) -> int:
    """Count tokens in text"""
    try:
        return len(tokenizer.encode(text))
    except Exception:
        # Fallback: approximate token count
        return len(text.split()) * 1.3

def get_top_chunks(question: str, k: int = Config.MAX_CHUNKS, max_tokens: int = Config.MAX_TOKENS) -> List[str]:
    """Get relevant chunks for the question"""
    try:
        # Generate embedding
        response = client.embeddings.create(
            input=[question], 
            model=Config.EMBEDDING_MODEL
        )
        vec = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
        
        # Search FAISS index
        scores, indices = index.search(vec, k)
        
        context_chunks = []
        total_tokens = 0
        
        for idx in indices[0]:
            # Skip invalid indices (FAISS can return -1)
            if idx < 0 or idx >= len(chunks):
                continue
                
            chunk_text = chunks[idx].get("text", "")
            if not chunk_text:
                continue
                
            chunk_tokens = num_tokens(chunk_text)
            if total_tokens + chunk_tokens > max_tokens:
                break
                
            context_chunks.append(chunk_text)
            total_tokens += chunk_tokens
        
        logger.info(f"Retrieved {len(context_chunks)} chunks with {total_tokens} tokens")
        return context_chunks
        
    except Exception as e:
        logger.error(f"Error in get_top_chunks: {e}")
        return []

def get_rag_answer(question: str) -> str:
    """Generate RAG answer"""
    if not question.strip():
        return "Please provide a valid question."
    
    context_chunks = get_top_chunks(question)
    if not context_chunks:
        return "⚠️ No relevant context found. Please try rephrasing your question."

    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "You are Karin from AlignerService. You are a helpful assistant with experience in clear aligner support.\n"
        "Answer the question based only on the context below. If the context doesn't contain enough information "
        "or if more clinical expertise is needed, clearly state this.\n\n"
        f"Context:\n{context}\n\n---\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    try:
        response = client.chat.completions.create(
            model=Config.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500  # Limit response length
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"⚠️ Unable to generate answer. Please try again later."

def insert_into_db(ticket_id: str, question: str, answer: str) -> bool:
    """Insert ticket data into database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT,
                    question TEXT,
                    answer TEXT,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticket_id, question)
                )
            ''')
            
            # Insert or update ticket
            cursor.execute('''
                INSERT OR REPLACE INTO tickets (ticket_id, question, answer, source)
                VALUES (?, ?, ?, ?)
            ''', (ticket_id, question, answer, "RAG Assistant"))
            
            conn.commit()
            return True
            
    except Exception as e:
        logger.error(f"Database insert error: {e}")
        return False

class AnswerRequest(BaseModel):
    ticketId: str
    question: str
    
    @validator('ticketId')
    def validate_ticket_id(cls, v):
        if not v or len(v.strip()) < 1:
            raise ValueError('Ticket ID cannot be empty')
        if len(v) > 100:
            raise ValueError('Ticket ID too long')
        return v.strip()
    
    @validator('question')
    def validate_question(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('Question must be at least 3 characters')
        if len(v) > 1000:
            raise ValueError('Question too long')
        return v.strip()

@app.post("/answer")
async def answer(req: AnswerRequest):
    """API endpoint for getting answers"""
    try:
        answer_text = get_rag_answer(req.question)
        success = insert_into_db(req.ticketId, req.question, answer_text)
        
        return {
            "answer": answer_text,
            "ticket_id": req.ticketId,
            "success": success
        }
    except Exception as e:
        logger.error(f"Error in answer endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/ui", response_class=HTMLResponse)
async def get_ui(ticketId: str = "", question: str = ""):
    """Web interface for asking questions"""
    # Escape HTML to prevent XSS
    safe_ticket_id = html.escape(ticketId)
    safe_question = html.escape(question)
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AlignerService AI Assistant</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            textarea {{ width: 100%; box-sizing: border-box; }}
            input[type="text"] {{ width: 100%; padding: 8px; margin: 5px 0; }}
            button {{ background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }}
        </style>
    </head>
    <body>
        <h2>AlignerService AI Assistant</h2>
        <form method="post" action="/ui">
            <label for="ticketId">Ticket ID:</label><br>
            <input type="text" name="ticketId" id="ticketId" value="{safe_ticket_id}" required /><br><br>
            
            <label for="question">Your question:</label><br>
            <textarea name="question" id="question" rows="4" placeholder="Ask your question here..." required>{safe_question}</textarea><br><br>
            
            <button type="submit">Get Answer</button>
        </form>
    </body>
    </html>
    """

@app.post("/ui", response_class=HTMLResponse)
async def post_ui(request: Request):
    """Handle form submission from web interface"""
    try:
        form = await request.form()
        ticket_id = str(form.get("ticketId", "")).strip()
        question = str(form.get("question", "")).strip()
        
        if not ticket_id or not question:
            raise HTTPException(status_code=400, detail="Both ticket ID and question are required")
        
        answer = get_rag_answer(question)
        insert_into_db(ticket_id, question, answer)
        
        # Escape HTML output
        safe_ticket_id = html.escape(ticket_id)
        safe_question = html.escape(question)
        safe_answer = html.escape(answer)
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AlignerService AI Assistant - Answer</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .answer {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #4CAF50; }}
                a {{ color: #4CAF50; text-decoration: none; }}
            </style>
        </head>
        <body>
            <h2>AlignerService AI Assistant</h2>
            <p><strong>Ticket ID:</strong> {safe_ticket_id}</p>
            <p><strong>Question:</strong> {safe_question}</p>
            <div class="answer">
                <strong>Answer:</strong><br>
                {safe_answer}
            </div>
            <p><a href="/ui?ticketId={safe_ticket_id}">Ask another question</a></p>
        </body>
        </html>
        """
        
    except Exception as e:
        logger.error(f"Error in UI form handler: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/update_ticket")
async def update_ticket(req: AnswerRequest):
    """Update existing ticket with new Q&A"""
    try:
        answer = get_rag_answer(req.question)
        success = insert_into_db(req.ticketId, req.question, answer)
        
        return {
            "ticket_id": req.ticketId,
            "answer": answer,
            "success": success
        }
    except Exception as e:
        logger.error(f"Error updating ticket: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "chunks_loaded": len(chunks),
        "timestamp": datetime.now().isoformat()
    }
