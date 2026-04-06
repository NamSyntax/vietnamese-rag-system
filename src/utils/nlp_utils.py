import re
from underthesea import word_tokenize


# text cleaning utility
def clean_vietnamese_text(text: str) -> str:
    # remove trash chars and normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# word segmentation utility
def segment_vietnamese(text: str) -> str:
    # normalize segment
    return word_tokenize(text, format="text")
