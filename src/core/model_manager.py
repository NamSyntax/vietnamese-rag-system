import logging
import torch
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)

# global model manager
class ModelManager:
    _embed_model = None

    @classmethod
    def get_embed_model(cls):
        """
        singleton pattern to load the embedding model only once.
        """
        if cls._embed_model is None:
            logger.info("loading BGE-M3 embedding model for the first time...")
            use_fp16 = torch.cuda.is_available()
            cls._embed_model = BGEM3FlagModel(
                'BAAI/bge-m3',
                use_fp16=use_fp16
            )
            logger.info("BGE-M3 embedding model loaded successfully!")
        return cls._embed_model