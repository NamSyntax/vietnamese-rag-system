# main_rag.py
import asyncio
from src.retrieval.search_engine import RAGRetriever
from src.generation.generator import RAGGenerator

async def run_rag_pipeline(user_query: str):
    retriever = RAGRetriever()
    generator = RAGGenerator()

    print(f"\n Đang tìm kiếm thông tin cho: '{user_query}'...")
    
    # retrieve - rerank
    relevant_docs = retriever.search(user_query, top_k=3)
    
    if not relevant_docs:
        print("Không tìm thấy thông tin liên quan trong database.")
        return

    # generate
    print("🤖 LLM đang suy luận...")
    answer = await generator.generate_answer(user_query, relevant_docs)

    print("\n" + "="*50)
    print(f"CÂU TRẢ LỜI:\n{answer}")
    print("="*50)
    
    # trích nguồn
    print("\nNGUỒN THAM KHẢO:")
    for doc in relevant_docs:
        source = doc['metadata'].get('source', 'Unknown')
        page = doc['metadata'].get('page', '?')
        print(f"- {source} (Trang {page+1})")

if __name__ == "__main__":
    query = input("Nhập câu hỏi của bạn: ")
    asyncio.run(run_rag_pipeline(query))