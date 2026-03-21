# src/api/routers/document.py
import asyncio
import logging
import os
import shutil
import tempfile
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form, Depends

# Import từ các file chúng ta đã quy hoạch lại
from src.core.cache import set_upload_status, get_upload_status, clear_session_data
from src.ingestion.vector_store import VectorStoreManager
from src.ingestion.pdf_loader import PDFIngestionPipeline
from src.api.dependencies import get_vector_store, get_pipeline

logger = logging.getLogger(__name__)

# khởi tạo Router FastAPI app
router = APIRouter(tags=["Documents"])

async def process_and_ingest(file_path: str, session_id: str, vector_store: VectorStoreManager, pipeline: PDFIngestionPipeline):
    """Tiến trình chạy ngầm để xử lý PDF và đưa vào Vector DB"""
    try:
        await set_upload_status(session_id, "Đang trích xuất và phân mảnh văn bản...")
        docs = await asyncio.to_thread(pipeline.process_pdf, file_path)
        
        if not docs:
            await set_upload_status(session_id, "Lỗi: Không tìm thấy nội dung văn bản.")
            return

        await set_upload_status(session_id, "Đang phân tích ngữ nghĩa và đưa vào bộ nhớ...")
        await vector_store.create_collection(session_id)
        await vector_store.upsert_documents(docs, collection_name=session_id)

        await set_upload_status(session_id, "Hoàn tất")
    except Exception as e:
        logger.error(f"Lỗi: {e}")
        await set_upload_status(session_id, f"Lỗi hệ thống: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    vector_store: VectorStoreManager = Depends(get_vector_store),
    pipeline: PDFIngestionPipeline = Depends(get_pipeline)
):
    """nhận file, lưu tạm, đẩy vào background task"""
    if not file.filename.endswith(".pdf"):
        return {"error": "Chỉ hỗ trợ file PDF."}
    
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"{session_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    await set_upload_status(session_id, "Đang tiếp nhận file...")
    background_tasks.add_task(process_and_ingest, file_path, session_id, vector_store, pipeline)
    return {"message": "Đã nhận file, bắt đầu xử lý.", "session_id": session_id}

@router.get("/status/{session_id}")
async def check_status(session_id: str):
    """gọi API liên tục để vẽ progres bar"""
    status = await get_upload_status(session_id)
    return {"status": status}

@router.delete("/session/{session_id}")
async def clear_session(session_id: str, vector_store: VectorStoreManager = Depends(get_vector_store)):
    """API để dọn dẹp collection khi user reset hoặc rời đi"""
    await vector_store.delete_collection(session_id)
    await clear_session_data(session_id)
    return {"message": f"Đã dọn dẹp dữ liệu của phiên {session_id}"}