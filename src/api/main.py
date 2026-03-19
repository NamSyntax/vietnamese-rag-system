# src/api/app.py
import asyncio
import json
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request, Depends, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from src.retrieval.search_engine import RAGRetriever
from src.generation.generator import RAGGenerator
from src.ingestion.vector_store import VectorStoreManager
from src.ingestion.pdf_loader import PDFIngestionPipeline
from src.utils.redis_client import set_upload_status, get_upload_status, clear_session_data

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# dictionary to track upload status in-memory
UPLOAD_STATUS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing system components...")
    try:
        app.state.retriever = RAGRetriever()
        app.state.generator = RAGGenerator()
        app.state.vector_store = VectorStoreManager()
        app.state.pipeline = PDFIngestionPipeline()
        logger.info("System components initialized successfully!")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize system components: {e}", exc_info=True)
        raise e
    finally:
        logger.info("Releasing system resources...")
        app.state.retriever = None
        app.state.generator = None
        app.state.vector_store = None
        app.state.pipeline = None

app = FastAPI(title="Vietnamese RAG Dynamic API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DEPENDENCIES ---
def get_retriever(request: Request) -> RAGRetriever: return request.app.state.retriever
def get_generator(request: Request) -> RAGGenerator: return request.app.state.generator
def get_vector_store(request: Request) -> VectorStoreManager: return request.app.state.vector_store
def get_pipeline(request: Request) -> PDFIngestionPipeline: return request.app.state.pipeline

# --- BACKGROUND TASK ---
async def process_and_ingest(file_path: str, session_id: str, vector_store: VectorStoreManager, pipeline: PDFIngestionPipeline):
    """pdf processing + vector store ingestion"""
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
        logger.error(f"Error: {e}")
        await set_upload_status(session_id, f"Lỗi hệ thống: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            
# --- ENDPOINTS ---
@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    vector_store: VectorStoreManager = Depends(get_vector_store),
    pipeline: PDFIngestionPipeline = Depends(get_pipeline)
):
    """receive file upload, save to temp, trigger background processing"""
    if not file.filename.endswith(".pdf"):
        return {"error": "Chỉ hỗ trợ file PDF."}
    
    # create a unique temp file
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"{session_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    await set_upload_status(session_id, "Đang nhận file...")
    
    # Đẩy task xử lý vào background để không block API
    background_tasks.add_task(process_and_ingest, file_path, session_id, vector_store, pipeline)
    
    return {"message": "Đã nhận file, bắt đầu xử lý.", "session_id": session_id}


@app.get("/status/{session_id}")
async def check_status(session_id: str):
    """Progress Bar UI"""
    status = await get_upload_status(session_id)
    return {"status": status}

@app.delete("/session/{session_id}")
async def clear_session(session_id: str, vector_store: VectorStoreManager = Depends(get_vector_store)):
    """API to clear session data when user starts a new upload or leaves"""
    await vector_store.delete_collection(session_id)
    await clear_session_data(session_id)
    return {"message": f"Đã dọn dẹp dữ liệu của phiên {session_id}"}

@app.get("/ask")
async def ask_rag(
    query: str = Query(...),
    session_id: str = Query(...), # Bắt buộc phải có session_id để biết tìm ở đâu
    retriever: RAGRetriever = Depends(get_retriever),
    generator: RAGGenerator = Depends(get_generator)
):
    async def stream_result():
        try:
            try:
                # Tìm đúng collection của người dùng này
                relevant_docs = await retriever.search(query, collection_name=session_id, top_k=8)
            except Exception as e:
                yield json.dumps({"type": "error", "message": "Phiên làm việc không tồn tại hoặc dữ liệu chưa sẵn sàng. Vui lòng tải file lại."}) + "\n"
                return

            if not relevant_docs:
                yield json.dumps({"type": "error", "message": "Dựa trên tài liệu bạn tải lên, tôi không tìm thấy thông tin phù hợp."}) + "\n"
                return

            sources = [{"page": d.get("page"), "chunk_index": d.get("chunk_index")} for d in relevant_docs]
            yield json.dumps({"type": "sources", "data": sources}) + "\n"

            try:
                async for chunk in generator.generate_stream(query, relevant_docs):
                    yield json.dumps({"type": "content", "data": chunk}) + "\n"
            except Exception as e:
                yield json.dumps({"type": "error", "message": "\n\n*(Lỗi: Mất kết nối tới LLM)*"}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "message": "Lỗi luồng hệ thống."}) + "\n"

    return StreamingResponse(stream_result(), media_type="application/x-ndjson")