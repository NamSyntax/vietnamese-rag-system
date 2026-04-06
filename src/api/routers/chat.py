import json
import asyncio
from fastapi import APIRouter, Query, Depends
from fastapi.responses import StreamingResponse

from src.retrieval.search_engine import RAGRetriever
from src.generation.generator import RAGGenerator
from src.api.dependencies import get_retriever, get_generator
from src.core.cache import get_cached_response, set_cached_response  # add cache

router = APIRouter(tags=["Chat"])


# query rag system
@router.get("/ask")
async def ask_rag(
    query: str = Query(...),
    session_id: str = Query(...),
    retriever: RAGRetriever = Depends(get_retriever),
    generator: RAGGenerator = Depends(get_generator),
):
    # stream response generator
    async def stream_result():
        try:
            # cache check
            cached = await get_cached_response(session_id, query)
            if cached:
                # return sources
                yield json.dumps({"type": "sources", "data": cached["sources"]}) + "\n"
                # fake streaming for smooth ux
                words = cached["response"].split(" ")
                for word in words:
                    yield json.dumps({"type": "content", "data": word + " "}) + "\n"
                    await asyncio.sleep(0.02)
                return

            # run pipeline if no cache
            try:
                relevant_docs = await retriever.search(query, collection_name=session_id, top_k=8)
            except Exception:
                yield (
                    json.dumps(
                        {
                            "type": "error",
                            "message": "Phiên làm việc không tồn tại hoặc dữ liệu chưa sẵn sàng. Vui lòng tải file lại.",
                        }
                    )
                    + "\n"
                )
                return

            # handle empty results
            if not relevant_docs:
                yield (
                    json.dumps(
                        {
                            "type": "error",
                            "message": "Dựa trên tài liệu bạn tải lên, tôi không tìm thấy thông tin phù hợp.",
                        }
                    )
                    + "\n"
                )
                return

            sources = [
                {
                    "page": d.get("page"),
                    "chunk_index": d.get("chunk_index"),
                    "content": d.get("content"),
                }
                for d in relevant_docs
            ]
            yield json.dumps({"type": "sources", "data": sources}) + "\n"

            # cache response while streaming
            full_response_text = ""
            try:
                async for chunk in generator.generate_stream(query, relevant_docs):
                    full_response_text += chunk
                    yield json.dumps({"type": "content", "data": chunk}) + "\n"

                # save to redis
                await set_cached_response(session_id, query, full_response_text, sources)

            except Exception:
                yield (
                    json.dumps(
                        {
                            "type": "error",
                            "message": "\n\n*(Lỗi: Mất kết nối tới mô hình ngôn ngữ AI)*",
                        }
                    )
                    + "\n"
                )

        except Exception:
            yield json.dumps({"type": "error", "message": "Lỗi luồng hệ thống."}) + "\n"

    return StreamingResponse(stream_result(), media_type="application/x-ndjson")
