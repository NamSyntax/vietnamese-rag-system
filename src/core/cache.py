import json
import hashlib
import redis.asyncio as redis
from src.core.config import settings

# init Redis client (async)
redis_db = redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)


# set processing status
async def set_upload_status(session_id: str, status: str, expire_seconds: int = 3600):
    """save file processing status, auto delete after 1 hour"""
    await redis_db.set(f"status:{session_id}", status, ex=expire_seconds)


# get processing status
async def get_upload_status(session_id: str) -> str:
    status = await redis_db.get(f"status:{session_id}")
    return status or "Không tìm thấy phiên xử lý."


# clear session data
async def clear_session_data(session_id: str):
    await redis_db.delete(f"status:{session_id}")


def _hash_query(session_id: str, query: str) -> str:
    # hash query key
    clean_query = " ".join(query.lower().split())
    query_hash = hashlib.md5(clean_query.encode()).hexdigest()
    return f"cache:{session_id}:{query_hash}"


# get response from cache
async def get_cached_response(session_id: str, query: str):
    """check if query has been answered in session"""
    key = _hash_query(session_id, query)
    cached_data = await redis_db.get(key)
    if cached_data:
        return json.loads(cached_data)
    return None


# save response to cache
async def set_cached_response(
    session_id: str, query: str, response: str, sources: list, expire_seconds: int = 86400
):
    """save answer to cache (24h default)"""
    key = _hash_query(session_id, query)
    data = {"response": response, "sources": sources}
    await redis_db.set(key, json.dumps(data), ex=expire_seconds)
