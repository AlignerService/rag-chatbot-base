import os, asyncio, aiosqlite, glob

DB_PATH = os.getenv("SQLITE_PATH", "/Users/macpro/Dropbox/AlignerService/RAG:Database:aktiv/rag.sqlite3")
MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "migrations")

async def run_migrations():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TEXT NOT NULL);"
        )
        await db.commit()

        for path in sorted(glob.glob(os.path.join(MIGRATIONS_DIR, "*.sql"))):
            version = os.path.splitext(os.path.basename(path))[0]
            async with db.execute("SELECT 1 FROM schema_migrations WHERE version=?", (version,)) as cur:
                row = await cur.fetchone()
            if row:
                continue

            with open(path, "r", encoding="utf-8") as f:
                sql = f.read()
            await db.executescript(sql)
            await db.commit()
