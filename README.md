# Simple RAG Proxy Server

This is a minimal FastAPI server to test webhook integration with Zoho Desk or similar platforms.

## Files
- `app.py`: Contains the FastAPI application with a single `/chat` endpoint.
- `requirements.txt`: Lists only the essential dependencies.
- `README.md`: This file.

## Render Settings

- **Start Command**: `uvicorn app:app --host 0.0.0.0 --port 10000`
- **Instance Type**: Starter ($7/month)
- **Root Directory**: leave blank
- **Auto Deploy**: On