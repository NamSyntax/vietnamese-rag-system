from src.retrieval.search_engine import remove_accents, RAGRetriever
from unittest.mock import MagicMock


def test_remove_accents():
    text = "Cộng hòa xã hội chủ nghĩa Việt Nam"
    expected = "Cong hoa xa hoi chu nghia Viet Nam"
    assert remove_accents(text) == expected


def test_query_expansion_creates_variants(mocker):
    # Mocking init components out so it doesn't try to connect to qdrant or BGE local model
    mocker.patch("src.retrieval.search_engine.AsyncQdrantClient")
    mocker.patch(
        "src.retrieval.search_engine.ModelManager.get_embed_model", return_value=MagicMock()
    )
    mocker.patch("src.retrieval.search_engine.CrossEncoder", return_value=MagicMock())

    retriever = RAGRetriever()
    query = "Luật Dân Sự"  # segmented by underthesea internally in search, but testing expand here directly

    # testing isolated function
    variants = retriever._expand_query(query)

    assert "Luật Dân Sự" in variants
    assert "luật dân sự" in variants
    assert "Luat Dan Su" in variants
    # The output is a list derived from a set, length should be 3
    assert len(variants) == 3


def test_normalize_sparse(mocker):
    mocker.patch("src.retrieval.search_engine.AsyncQdrantClient")
    mocker.patch(
        "src.retrieval.search_engine.ModelManager.get_embed_model", return_value=MagicMock()
    )
    mocker.patch("src.retrieval.search_engine.CrossEncoder", return_value=MagicMock())

    retriever = RAGRetriever()

    # Sparse model outputs weights
    sparse_raw = {101: 5.0, 102: 10.0, 103: 2.5}
    expected = {101: 0.5, 102: 1.0, 103: 0.25}  # divided by max (10.0)

    normalized = retriever._normalize_sparse(sparse_raw)
    assert normalized == expected
