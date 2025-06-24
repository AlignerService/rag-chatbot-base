# Simple RAG Proxy Server

This is a minimal FastAPI server to test webhook integration with Zoho Desk or similar platforms.

## Folder Structure

```bash
/app         ← your FastAPI application
/scripts     ← backup scripts and index build
/ui          ← frontend template
.github/...  ← CI configuration
render.yaml  ← Render deployment configuration
```

## Getting Started Locally

To start the server locally, run:

```bash
uvicorn app:app --reload
```

The server will be available at `http://127.0.0.1:8000` by default.

## Deployment on Render

Use the following settings for deployment on Render.com:

* **Start Command:** `uvicorn app:app --host 0.0.0.0 --port 10000`
* **Instance Type:** Starter (\$7/month)
* **Auto Deploy:** On
* **Root Directory:** (leave blank)
