# src/core/cache.py
import json 
import hashlib 
import redis.asyncio as redis
from src.core.config import settings

# init Redis client (async)
redis_db = redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)

async def set_upload_status(session_id: str, status: str, expire_seconds: int = 3600):
    """Lưu trạng thái xử lý file, tự động xóa sau 1 giờ để dọn rác"""
    await redis_db.set(f"status:{session_id}", status, ex=expire_seconds)

async def get_upload_status(session_id: str) -> str:
    status = await redis_db.get(f"status:{session_id}")
    return status or "Không tìm thấy phiên xử lý."

async def clear_session_data(session_id: str):
    await redis_db.delete(f"status:{session_id}")
    
def _hash_query(session_id: str, query: str) -> str:
    # hash querry - key
    clean_query = " ".join(query.lower().split())
    query_hash = hashlib.md5(clean_query.encode()).hexdigest()
    return f"cache:{session_id}:{query_hash}"

async def get_cached_response(session_id: str, query: str):
    """Kiểm tra xem câu hỏi này đã từng được trả lời trong session chưa"""
    key = _hash_query(session_id, query)
    cached_data = await redis_db.get(key)
    if cached_data:
        return json.loads(cached_data)
    return None

async def set_cached_response(session_id: str, query: str, response: str, sources: list, expire_seconds: int = 86400):
    """Lưu câu trả lời vào cache (Mặc định 24h)"""
    key = _hash_query(session_id, query)
    data = {"response": response, "sources": sources}
    await redis_db.set(key, json.dumps(data), ex=expire_seconds)