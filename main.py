from app.config import create_app
from app.services.image_moderation.router import router as image_moderation_router

app = create_app()

# Include service routers
app.include_router(
    image_moderation_router,
    prefix="/image-moderation",
    tags=["Image Moderation"],
)

# Sau này thêm service mới, chỉ cần include router:
# from app.services.text_moderation.router import router as text_moderation_router
# app.include_router(text_moderation_router, prefix="/text-moderation", tags=["Text Moderation"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
