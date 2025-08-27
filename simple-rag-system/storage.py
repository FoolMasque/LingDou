# storage.py
import aiosqlite
from pathlib import Path
from typing import Optional, Dict

DB_PATH = Path("./rag_meta.db")

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS images(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  business_id TEXT NOT NULL,
  original_url TEXT NOT NULL UNIQUE,
  local_path TEXT,
  public_url TEXT,
  sha256 TEXT,
  bytes INTEGER,
  status TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_images_biz ON images(business_id);
CREATE INDEX IF NOT EXISTS idx_images_sha ON images(sha256);
"""

_inited = False

async def init_db():
    """初始化数据库（只执行一次）"""
    global _inited
    if _inited:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(CREATE_SQL)
        await db.commit()
    _inited = True


async def get_image_by_url(original_url: str) -> Optional[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM images WHERE original_url=?", (original_url,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def get_image_by_hash(sha256: str) -> Optional[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM images WHERE sha256=?", (sha256,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def upsert_image(rec: Dict):
    keys = ["business_id","original_url","local_path","public_url","sha256","bytes","status"]
    vals = [rec.get(k) for k in keys]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
        INSERT INTO images(business_id,original_url,local_path,public_url,sha256,bytes,status)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(original_url) DO UPDATE SET
            local_path=excluded.local_path,
            public_url=excluded.public_url,
            sha256=excluded.sha256,
            bytes=excluded.bytes,
            status=excluded.status
        """, vals)
        await db.commit()
