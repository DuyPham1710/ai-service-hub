import logging
import uuid
from datetime import datetime, timezone

from qdrant_client import QdrantClient, models

from .config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, SIMILARITY_THRESHOLD, SEARCH_LIMIT

logger = logging.getLogger("ai-service-hub")


class FaceVectorStore:
    """Qdrant client wrapper: upsert, search, delete face embeddings."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "FaceVectorStore":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = COLLECTION_NAME
        self._ensure_collection()
        logger.info("Qdrant connection established")

    def _ensure_collection(self):
        """Tạo collection nếu chưa tồn tại."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            logger.info(f"Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=512,  # ArcFace embedding dimension
                    distance=models.Distance.COSINE,
                ),
            )
            # Index user_id để query/delete nhanh
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="user_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Collection '{self.collection_name}' created with payload index on 'user_id'")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")

    def upsert_face(self, embedding: list[float], user_id: str, image_url: str, source: str) -> str:
        """
        Lưu 1 face embedding + metadata vào Qdrant.

        Returns: point_id (uuid)
        """
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "user_id": user_id,
                        "image_url": image_url,
                        "source": source,  # "avatar" hoặc "selfie"
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            ],
        )
        logger.info(f"Upserted face embedding for user '{user_id}' (source: {source}, point: {point_id})")
        return point_id

    def upsert_registration_face(
        self, embedding: list[float], user_id: str, pose: str
    ) -> str:
        """
        Lưu 1 face registration embedding + pose metadata vào Qdrant.
        Không cần image_url vì ảnh gốc không được lưu trữ.

        Returns: point_id (uuid)
        """
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "user_id": user_id,
                        "source": "registration",
                        "pose": pose,  # "center", "up", "down", "left", "right"
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            ],
        )
        logger.info(f"Upserted registration face for user '{user_id}' (pose: {pose}, point: {point_id})")
        return point_id

    def search_similar(
        self,
        embedding: list[float],
        threshold: float = SIMILARITY_THRESHOLD,
        limit: int = SEARCH_LIMIT,
    ) -> list[dict]:
        """
        Tìm faces tương tự trong Qdrant.

        Returns: list of { user_id, confidence, image_url }
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=limit,
            score_threshold=threshold,
        )

        matches = []
        for result in results:
            matches.append({
                "user_id": result.payload.get("user_id"),
                "confidence": round(result.score, 4),
                "image_url": result.payload.get("image_url"),
                "source": result.payload.get("source"),
            })

        return matches

    def get_user_avatar_embedding(self, user_id: str) -> list[float] | None:
        """
        Lấy embedding avatar hiện tại của 1 user.
        Trả về vector 512-dim hoặc None nếu chưa enroll.
        """
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    ),
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value="avatar"),
                    ),
                ]
            ),
            with_vectors=True,
            limit=1,
        )

        points = results[0]  # scroll returns (points, next_offset)
        if len(points) == 0:
            return None

        return points[0].vector

    def get_user_registration_embedding(self, user_id: str) -> list[float] | None:
        """
        Lấy embedding registration (ground truth) của 1 user.
        Ưu tiên lấy pose "center" vì là góc mặt chính diện, chính xác nhất.
        Trả về vector 512-dim hoặc None nếu chưa register.
        """
        # Thử lấy center pose trước
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    ),
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value="registration"),
                    ),
                    models.FieldCondition(
                        key="pose",
                        match=models.MatchValue(value="center"),
                    ),
                ]
            ),
            with_vectors=True,
            limit=1,
        )

        points = results[0]
        if len(points) > 0:
            return points[0].vector

        # Fallback: lấy bất kỳ registration embedding nào
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    ),
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value="registration"),
                    ),
                ]
            ),
            with_vectors=True,
            limit=1,
        )

        points = results[0]
        if len(points) == 0:
            return None

        return points[0].vector

    def delete_by_user_and_source(self, user_id: str, source: str) -> None:
        """Xóa embeddings của 1 user theo source (avatar hoặc selfie)."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[  # bằng với toán tử AND
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(value=user_id),
                        ),
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source),
                        ),
                    ]
                )
            ),
        )
        logger.info(f"Deleted face embeddings for user '{user_id}' (source: {source})")

    def delete_all_by_user(self, user_id: str) -> None:
        """Xóa TẤT CẢ embeddings của 1 user."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(value=user_id),
                        ),
                    ]
                )
            ),
        )
        logger.info(f"Deleted ALL face embeddings for user '{user_id}'")

    def get_collection_info(self) -> dict:
        """Lấy thông tin collection (số lượng points, etc.)."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.value,
        }
