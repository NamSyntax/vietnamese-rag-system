import pytest
import os
from unittest.mock import MagicMock
from src.core.cache import _hash_query


# MOCK testing for fast CI
def test_cache_hashing():
    """Test Redis hashing logic doesn't throw errors"""
    h = _hash_query("session-123", "Câu hỏi test!")
    assert "cache:session-123" in h


@pytest.fixture
def mock_heavy_models(mocker):
    """
    Mock the heavy BGE-M3 model so it doesn't download 2.2GB on CI runs.
    Return a dummy embedding tensor matching the shape.
    """
    if os.environ.get("DISABLE_HEAVY_MODELS") == "true":
        # Mock BGEM3FlagModel
        mock_model = MagicMock()
        mock_model.encode.return_value = {
            "dense_vecs": MagicMock(),  # dummy arrays
            "lexical_weights": [{"123": 0.5}],
        }

        mocker.patch("src.core.model_manager.ModelManager._embed_model", mock_model)
        mocker.patch("src.core.model_manager.ModelManager.get_embed_model", return_value=mock_model)

        # Mock CrossEncoder
        mocker.patch("src.retrieval.search_engine.CrossEncoder", return_value=MagicMock())
        yield mock_model
    else:
        yield None


def test_heavy_models_mocked(mock_heavy_models):
    """
    Ensure the heavy models mockup works if CI environment variable is set.
    """
    if os.environ.get("DISABLE_HEAVY_MODELS") == "true":
        from src.core.model_manager import ModelManager

        model = ModelManager.get_embed_model()
        assert model is not None

        # Check that we bypass the real logic
        res = model.encode(["Test"])
        assert "dense_vecs" in res
