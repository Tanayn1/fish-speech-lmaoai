import asyncpg
import os
from fish_speech.utils.schema import Voice

conn = None

async def connect():
    global conn
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))

async def disconnect():
    global conn
    await conn.close()

async def fetchVoice(id) -> Voice | None:
    global conn
    row = await conn.fetchrow('SELECT * FROM "Voices" WHERE id = $1', id)
    return Voice(**dict(row)) if row else None

