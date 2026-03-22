from src.ingestion.pdf_loader import PDFIngestionPipeline
from src.ingestion.vector_store import VectorStoreManager

def main():
    loader = PDFIngestionPipeline()
    chunks = loader.process_pdf("data/test.pdf")

    # indexing
    vdb = VectorStoreManager()
    vdb.create_collection()
    vdb.upsert_documents(chunks)

if __name__ == "__main__":
    main()