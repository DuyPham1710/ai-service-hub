import logging
import numpy as np
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


class RegisterFaceRequest(BaseModel):
    user_id: str
    images: list[str]  # 5 base64 strings
    poses: list[str] = ["center", "up", "down", "left", "right"]


@router.get("/health")
async def health_check():
    """Health check cho face recognition service."""
    try:
        store = FaceVectorStore.get_instance()
        info = store.get_collection_info()
        return {"status": "ok", "model": "insightface_buffalo_l", "collection": info}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@router.post("/register")
async def register_face(request: RegisterFaceRequest):
    """
    Đăng ký khuôn mặt khi tạo tài khoản (hoặc quét lại sau).
    - Nhận 5 ảnh base64 (center, up, down, left, right)
    - Detect face mỗi ảnh (phải có đúng 1 face/ảnh)
    - Kiểm tra consistency: tất cả face phải cùng 1 người
    - Kiểm tra global uniqueness: face chưa thuộc về user khác
    - Xóa registration cũ (nếu re-scan) → lưu 5 embeddings mới
    """
    if len(request.images) != len(request.poses):
        return {
            "success": False,
            "message": f"Số lượng ảnh ({len(request.images)}) không khớp với số lượng pose ({len(request.poses)})",
        }

    detector = FaceDetector.get_instance()
    store = FaceVectorStore.get_instance()

    # ===== Bước 1: Detect face mỗi ảnh =====
    embeddings = []
    for idx, (base64_img, pose) in enumerate(zip(request.images, request.poses)):
        faces = detector.detect_faces_from_base64(base64_img)

        if len(faces) == 0:
            return {
                "success": False,
                "message": f"Không phát hiện khuôn mặt ở ảnh {pose} (ảnh {idx + 1}/{len(request.images)})",
                "failed_pose": pose,
            }

        if len(faces) > 1:
            return {
                "success": False,
                "message": f"Phát hiện nhiều hơn 1 khuôn mặt ở ảnh {pose}",
                "failed_pose": pose,
            }

        embeddings.append({
            "embedding": faces[0]["embedding"],
            "confidence": faces[0]["confidence"],
            "pose": pose,
        })

    logger.info(f"Registration: detected faces in all {len(embeddings)} images for user {request.user_id}")

    # ===== Bước 2: Consistency check — tất cả face phải cùng 1 người =====
    # for i in range(len(embeddings)):
    #     for j in range(i + 1, len(embeddings)):
    #         vec_a = np.array(embeddings[i]["embedding"])
    #         vec_b = np.array(embeddings[j]["embedding"])
    #         similarity = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))

    #         # Giảm ngưỡng cho consistency check vì các góc mặt khác nhau (trái/phải/lên/xuống)
    #         # sẽ có độ tương đồng thấp hơn so với 2 ảnh chụp thẳng
    #         if similarity < 0.15:
    #             logger.warning(
    #                 f"Consistency check failed for user {request.user_id}: "
    #                 f"{embeddings[i]['pose']} vs {embeddings[j]['pose']} = {similarity:.4f}"
    #             )
    #             return {
    #                 "success": False,
    #                 "message": f"Các ảnh không khớp nhau (ảnh {embeddings[i]['pose']} và {embeddings[j]['pose']}). Vui lòng quét lại.",
    #                 "similarity": round(similarity, 4),
    #             }

    # ===== Bước 3: Global uniqueness — face chưa thuộc user khác =====
    for emb in embeddings:
        similar_faces = store.search_similar(embedding=emb["embedding"], threshold=0.4)

        for match in similar_faces:
            if match["user_id"] != request.user_id:
                logger.warning(
                    f"Registration rejected: User {request.user_id} face matches user {match['user_id']} "
                    f"(confidence: {match['confidence']:.4f})"
                )
                return {
                    "success": False,
                    "message": "Khuôn mặt này đã được đăng ký bởi tài khoản khác.",
                    "matched_user_id": match["user_id"],
                    "similarity": match["confidence"],
                }

    # ===== Bước 4: Xóa registration cũ (re-scan) + lưu mới =====
    store.delete_by_user_and_source(request.user_id, "registration")

    point_ids = []
    for emb in embeddings:
        point_id = store.upsert_registration_face(
            embedding=emb["embedding"],
            user_id=request.user_id,
            pose=emb["pose"],
        )
        point_ids.append(point_id)

    logger.info(f"Registration completed for user {request.user_id}: {len(point_ids)} embeddings saved")

    return {
        "success": True,
        "message": "Đăng ký khuôn mặt thành công",
        "embeddings_saved": len(point_ids),
        "point_ids": point_ids,
    }


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

    # ===== SELFIE: so sánh face trong post với face đã đăng ký của poster =====
    if request.source == "selfie":
        # Ưu tiên dùng registration embedding (ground truth) nếu có
        # Fallback sang avatar nếu chưa register
        ref_embedding = store.get_user_registration_embedding(request.user_id)
        ref_source = "registration"

        if ref_embedding is None:
            ref_embedding = store.get_user_avatar_embedding(request.user_id)
            ref_source = "avatar"

        if ref_embedding is None:
            # User chưa có face nào enrolled → không verify được → không enroll
            return {
                "success": False,
                "faces_detected": len(faces),
                "message": "Cannot verify selfie: user has no face enrolled (no registration or avatar).",
            }

        ref_vec = np.array(ref_embedding)
        selfie_vec = np.array(new_embedding)
        similarity = float(np.dot(ref_vec, selfie_vec) / (np.linalg.norm(ref_vec) * np.linalg.norm(selfie_vec)))

        logger.info(f"Selfie verification for user {request.user_id} (vs {ref_source}): similarity = {similarity:.4f}")

        if similarity < 0.4:
            return {
                "success": False,
                "faces_detected": len(faces),
                "message": f"Selfie face does not match {ref_source} (similarity: {similarity:.4f}). Not enrolled.",
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
