import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# components
from src.core.config import settings
from src.retrieval.search_engine import RAGRetriever
from src.generation.generator import RAGGenerator
from src.ingestion.vector_store import VectorStoreManager
from src.ingestion.pdf_loader import PDFIngestionPipeline
from src.api.routers import document, chat

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# app lifecycle manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Đang khởi tạo toàn bộ AI Models (Singleton)...")
    try:
        # objects and states
        app.state.retriever = RAGRetriever()
        app.state.generator = RAGGenerator()
        app.state.vector_store = VectorStoreManager()
        app.state.pipeline = PDFIngestionPipeline()
        logger.info("Hệ thống khởi tạo thành công!")
        yield
    except Exception as e:
        logger.error(f"Khởi tạo thất bại: {e}", exc_info=True)
        raise e
    finally:
        logger.info("Đang giải phóng bộ nhớ...")
        app.state.retriever = None
        app.state.generator = None
        app.state.vector_store = None
        app.state.pipeline = None


# init app
app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routers
app.include_router(document.router)
app.include_router(chat.router)
