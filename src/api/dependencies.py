# src/api/dependencies.py
from fastapi import Request
from src.retrieval.search_engine import RAGRetriever
from src.generation.generator import RAGGenerator
from src.ingestion.vector_store import VectorStoreManager
from src.ingestion.pdf_loader import PDFIngestionPipeline

def get_retriever(request: Request) -> RAGRetriever: 
    return request.app.state.retriever

def get_generator(request: Request) -> RAGGenerator: 
    return request.app.state.generator

def get_vector_store(request: Request) -> VectorStoreManager: 
    return request.app.state.vector_store

def get_pipeline(request: Request) -> PDFIngestionPipeline: 
    return request.app.state.pipeline