from src.ingestion.pdf_loader import PDFIngestionPipeline
from src.ingestion.vector_store import VectorStoreManager


def main():
    # init pipeline
    loader = PDFIngestionPipeline()
    # read and chunk pdf
    chunks = loader.process_pdf("data/test.pdf")

    # index chunks
    # init vector db
    vdb = VectorStoreManager()
    vdb.create_collection()
    vdb.upsert_documents(chunks)


if __name__ == "__main__":
    main()
