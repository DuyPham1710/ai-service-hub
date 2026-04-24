import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .face_detector import FaceDetector
from .vector_store import FaceVectorStore

logger = logging.getLogger("ai-service-hub")

router = APIRouter()

class EnrollRequest(BaseModel):
    user_id: str
    image_url: str
    source: str = "avatar"  # "avatar" hoặc "selfie"


class SearchRequest(BaseModel):
    image_urls: list[str]


class MatchedUser(BaseModel):
    user_id: str
    confidence: float


class ImageResult(BaseModel):
    image_index: int
    faces_detected: int
    matched_users: list[MatchedUser]
    unmatched_faces: int


class SearchResponse(BaseModel):
    results: list[ImageResult]


@router.get("/health")
async def health_check():
    """Health check cho face recognition service."""
    try:
        store = FaceVectorStore.get_instance()
        info = store.get_collection_info()
        return {"status": "ok", "model": "insightface_buffalo_l", "collection": info}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@router.post("/enroll")
async def enroll_face(request: EnrollRequest):
    """
    Enroll face từ avatar hoặc selfie.
    - Detect face trong ảnh
    - Nếu source=avatar → xóa embedding avatar cũ trước khi lưu mới
    - Nếu source=avatar → so sánh face mới với face cũ, chỉ enroll nếu cùng 1 người
    - Lưu embedding + userId vào Qdrant
    """
    detector = FaceDetector.get_instance()
    store = FaceVectorStore.get_instance()

    # Detect faces
    faces = detector.detect_faces_from_url(request.image_url)

    if len(faces) == 0:
        return {
            "success": False,
            "faces_detected": 0,
            "message": "No face detected in image",
        }

    # Lấy face lớn nhất (confidence cao nhất, đã sort)
    best_face = faces[0]
    new_embedding = best_face["embedding"]

    # ===== AVATAR: Kiểm tra Global Uniqueness =====
    if request.source == "avatar":
        # Search xem face mới này giống ai trong DB không
        similar_faces = store.search_similar(embedding=new_embedding, threshold=0.4)
        
        # Kiểm tra xem có match nào thuộc về user KHÁC không
        stolen_from = None
        for match in similar_faces:
            if match["user_id"] != request.user_id:
                stolen_from = match
                break
                
        if stolen_from is not None:
            logger.warning(f"Spoofing detected: User {request.user_id} tried to use face of {stolen_from['user_id']}")
            return {
                "success": False,
                "faces_detected": len(faces),
                "message": f"Khuôn mặt này đã thuộc về người khác. Cập nhật ảnh thành công nhưng không đăng ký nhận diện.",
                "similarity": stolen_from["confidence"],
            }

        # Nếu chưa ai sở hữu (hoặc chỉ match chính mình), cho phép replace
        # Xóa embedding avatar cũ
        store.delete_by_user_and_source(request.user_id, "avatar")

    # ===== SELFIE: so sánh face trong post với avatar của poster =====
    if request.source == "selfie":
        avatar_embedding = store.get_user_avatar_embedding(request.user_id)

        if avatar_embedding is None:
            # User chưa có avatar enrolled → không verify được → không enroll
            return {
                "success": False,
                "faces_detected": len(faces),
                "message": "Cannot verify selfie: user has no avatar face enrolled.",
            }

        import numpy as np
        avatar_vec = np.array(avatar_embedding)
        selfie_vec = np.array(new_embedding)
        similarity = float(np.dot(avatar_vec, selfie_vec) / (np.linalg.norm(avatar_vec) * np.linalg.norm(selfie_vec)))

        logger.info(f"Selfie verification for user {request.user_id}: similarity = {similarity:.4f}")

        if similarity < 0.4:
            # Face trong post khác face avatar → không phải selfie của user này
            return {
                "success": False,
                "faces_detected": len(faces),
                "message": f"Selfie face does not match avatar (similarity: {similarity:.4f}). Not enrolled.",
                "similarity": round(similarity, 4),
            }

    # Lưu embedding mới
    point_id = store.upsert_face(
        embedding=new_embedding,
        user_id=request.user_id,
        image_url=request.image_url,
        source=request.source,
    )

    return {
        "success": True,
        "faces_detected": len(faces),
        "message": "Face enrolled successfully",
        "point_id": point_id,
        "confidence": best_face["confidence"],
    }


@router.post("/search", response_model=SearchResponse)
async def search_faces(request: SearchRequest):
    """
    Detect ALL faces trong danh sách ảnh → search Qdrant → trả về matched userIds.
    """
    detector = FaceDetector.get_instance()
    store = FaceVectorStore.get_instance()

    results = []

    for idx, image_url in enumerate(request.image_urls):
        faces = detector.detect_faces_from_url(image_url)

        if len(faces) == 0:
            results.append(ImageResult(
                image_index=idx,
                faces_detected=0,
                matched_users=[],
                unmatched_faces=0,
            ))
            continue

        # Search Qdrant cho từng face
        all_matches: dict[str, float] = {}  # user_id → best confidence
        unmatched = 0

        for face in faces:
            matches = store.search_similar(face["embedding"])

            if len(matches) == 0:
                unmatched += 1
                continue

            # Deduplicate: giữ confidence cao nhất cho mỗi user_id
            for match in matches:
                uid = match["user_id"]
                conf = match["confidence"]
                if uid not in all_matches or conf > all_matches[uid]:
                    all_matches[uid] = conf     # Giữ confidence cao nhất cho mỗi user

        matched_users = [
            MatchedUser(user_id=uid, confidence=conf)
            for uid, conf in sorted(all_matches.items(), key=lambda x: x[1], reverse=True)
        ]

        results.append(ImageResult(
            image_index=idx,
            faces_detected=len(faces),
            matched_users=matched_users,
            unmatched_faces=unmatched,
        ))

    return SearchResponse(results=results)


@router.delete("/user/{user_id}")
async def delete_user_faces(user_id: str):
    """Xóa tất cả face embeddings của 1 user."""
    store = FaceVectorStore.get_instance()
    store.delete_all_by_user(user_id)
    return {"deleted": True, "user_id": user_id}
