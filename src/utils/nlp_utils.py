# src/utils/nlp_utils.py
import re
from underthesea import word_tokenize

def clean_vietnamese_text(text: str) -> str:
    # remove trash characters, normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # kế hoach update: chuẩn hóa unicode
    return text

def segment_vietnamese(text: str) -> str:
    # chuẩn hóa segment
    return word_tokenize(text, format="text")