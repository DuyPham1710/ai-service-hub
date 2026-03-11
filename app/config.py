from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-service-hub")


def create_app() -> FastAPI:
    """Khởi tạo FastAPI app với cấu hình chung."""
    app = FastAPI(
        title="AI Service Hub",
        description="Trung tâm các dịch vụ AI: kiểm duyệt hình ảnh, văn bản, nhận diện khuôn mặt,...",
        version="2.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {"message": "AI Service Hub is running"}

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return app
