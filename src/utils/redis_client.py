# src/utils/redis_client.py
import redis.asyncio as redis
import os

# Khởi tạo kết nối Redis bất đồng bộ
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_db = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)

async def set_upload_status(session_id: str, status: str, expire_seconds: int = 3600):
    """Lưu trạng thái xử lý file, tự động xóa sau 1 giờ để dọn rác"""
    await redis_db.set(f"status:{session_id}", status, ex=expire_seconds)

async def get_upload_status(session_id: str) -> str:
    status = await redis_db.get(f"status:{session_id}")
    return status or "Không tìm thấy phiên xử lý."

async def clear_session_data(session_id: str):
    await redis_db.delete(f"status:{session_id}")
    # Sau này có thể dùng hàm này để xóa thêm lịch sử chat cache trong Redis