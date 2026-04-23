import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "face_embeddings")

# InsightFace configuration
FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "buffalo_l")
det_size = int(os.getenv("FACE_DET_SIZE", "640"))
FACE_DET_SIZE = (det_size, det_size)

# Search configuration
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "5"))
