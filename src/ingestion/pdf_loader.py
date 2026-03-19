# src/ingestion/pdf_loader.py
import os
import re
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from underthesea import sent_tokenize 
from src.utils.nlp_utils import clean_vietnamese_text, segment_vietnamese
import logging

logger = logging.getLogger(__name__)

class PDFIngestionPipeline:
    def __init__(self, max_chunk_length: int = 1000, chunk_overlap: int = 200):
        self.max_chunk_length = max_chunk_length
        self.chunk_overlap = chunk_overlap

    def _normalize_text(self, text: str) -> str:
        text = text.replace('\xa0', ' ')
        text = re.sub(r'^[\s]*[·•➢o]\s*', '- ', text, flags=re.MULTILINE)
        return text

    def _is_noise(self, block: str) -> bool:
        # loại nhanh header/footer của pdf
        if len(block) < 20:
            return True
        if re.match(r'^\s*\d+\s*$', block):  # page number
            return True
        return False

    # Core chunking 
    def _hybrid_structural_chunking(self, text: str) -> List[str]:
        text = self._normalize_text(text)

        # split theo structure
        blocks = re.split(r'\n\s*\n|\n(?=[-\*\+]\s*|\d+\.\s*)', text)

        chunks = []
        current_chunk_blocks = []
        current_length = 0

        for block in blocks:
            block = block.strip()
            if not block or self._is_noise(block):
                continue

            block_len = len(block)

            # -------- case 1: đoạn quá dài--------
            if block_len > self.max_chunk_length:
                if current_chunk_blocks:
                    chunks.append("\n\n".join(current_chunk_blocks))
                    current_chunk_blocks = []
                    current_length = 0

                sentences = sent_tokenize(block)
                temp_sentences = []
                temp_len = 0

                for sent in sentences:
                    if temp_len + len(sent) > self.max_chunk_length and temp_sentences:
                        chunks.append(" ".join(temp_sentences))

                        # overlap = 2 câu cuối
                        overlap = temp_sentences[-2:] if len(temp_sentences) >= 2 else temp_sentences
                        temp_sentences = overlap + [sent]
                        temp_len = sum(len(s) for s in temp_sentences) + len(temp_sentences)

                    else:
                        temp_sentences.append(sent)
                        temp_len += len(sent) + 1

                if temp_sentences:
                    current_chunk_blocks = [" ".join(temp_sentences)]
                    current_length = len(current_chunk_blocks[0])

                continue

            # -------- case 2: block bình thường --------
            if current_length + block_len > self.max_chunk_length and current_chunk_blocks:
                chunks.append("\n\n".join(current_chunk_blocks))

                # overlap block-level
                overlap_blocks = []
                overlap_length = 0

                for b in reversed(current_chunk_blocks):
                    if overlap_length + len(b) <= self.chunk_overlap:
                        overlap_blocks.insert(0, b)
                        overlap_length += len(b) + 2
                    else:
                        break

                current_chunk_blocks = overlap_blocks
                current_length = overlap_length

            current_chunk_blocks.append(block)
            current_length += block_len + 2

        if current_chunk_blocks:
            chunks.append("\n\n".join(current_chunk_blocks))

        return chunks

    # --------------------------- Main pipeline ---------------------------

    def process_pdf(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

        logger.info(f"loading PDF: {file_path}")
        loader = PyMuPDFLoader(file_path)
        raw_docs = loader.load()

        processed_chunks = []
        global_chunk_idx = 0

        for doc in raw_docs:
            cleaned_text = clean_vietnamese_text(doc.page_content)

            if not cleaned_text.strip():
                continue

            text_chunks = self._hybrid_structural_chunking(cleaned_text)

            for chunk_text in text_chunks:
                segmented_chunk = segment_vietnamese(chunk_text)

                enriched_metadata = doc.metadata.copy()
                enriched_metadata.update({
                    "chunk_index": global_chunk_idx,
                    "chunk_length": len(chunk_text),
                    "is_list": bool(re.search(r'^[-\*\+]\s', chunk_text)),
                })

                new_doc = Document(
                    page_content=segmented_chunk,  # dùng cho embedding
                    metadata={
                        **enriched_metadata,
                        "original_text": chunk_text  # dùng để trả lời
                    }
                )

                processed_chunks.append(new_doc)
                global_chunk_idx += 1

        logger.info(f"Done processing {len(processed_chunks)} chunks.")
        return processed_chunks