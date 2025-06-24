# Simple RAG Proxy Server

This is a minimal FastAPI server to test webhook integration with Zoho Desk or similar platforms.

## Mappestruktur

```bash
/app         ← din FastAPI-app  
/scripts     ← backup-scripts og index-byg  
/ui          ← frontend-template  
.github/...  ← CI  
render.yaml  ← Render-konfig  
```

## Kom i gang lokalt

For at starte serveren lokalt skal du køre:

```bash
uvicorn app:app --reload
```

Serveren vil herefter være tilgængelig på `http://127.0.0.1:8000` som standard.

## Render Settings

Følgende indstillinger bruges til deployment på Render.com:

* **Start Command:** `uvicorn app:app --host 0.0.0.0 --port 10000`
* **Instance Type:** Starter (\$7/month)
* **Auto Deploy:** On
* **Root Directory:** (efterlades blank)
