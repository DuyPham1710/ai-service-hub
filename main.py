from app.config import create_app
from app.services.image_moderation.router import router as image_moderation_router
from app.services.video_moderation.router import router as video_moderation_router

app = create_app()

# Include service routers
app.include_router(
    image_moderation_router,
    prefix="/image-moderation",
    tags=["Image Moderation"],
)

app.include_router(
    video_moderation_router,
    prefix="/video-moderation",
    tags=["Video Moderation"],
)


if __name__ == "__main__":
    import uvicorn

    # Chuyển đổi tham số thành string 'main:app' và thêm reload=True để uvicorn tự khởi động lại khi code thay đổi
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

