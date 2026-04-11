import os
import torch
import logging

logger = logging.getLogger("ai-service-hub")

# ───── Auto-detect GPU/CPU ─────
def get_device() -> str:
    """Tự động phát hiện GPU NVIDIA. Nếu có dùng GPU, không có thì dùng CPU."""
    if torch.cuda.is_available():
        device = "cuda:0"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🚀 GPU detected: {gpu_name} — sử dụng GPU để voice conversion")
    else:
        device = "cpu"
        logger.info("⚙️ Không tìm thấy GPU NVIDIA — sử dụng CPU (sẽ chậm hơn)")
    return device

DEVICE = get_device()

# ───── Đường dẫn model ─────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "voice_conversion")

# ───── Mapping preset → model file ─────
VOICE_PRESETS = {
    "con_gai": {
        "name": "Con gái",
        "model": "female.pth",
        "index": "female.index",
        "description": "Giọng nữ trẻ trung",
    },
    "tram": {
        "name": "Trầm",
        "model": "deep_male.pth",
        "index": "deep_male.index",
        "description": "Giọng nam trầm ấm",
    },
    "em_be": {
        "name": "Em bé",
        "model": "baby.pth",
        "index": "baby.index",
        "description": "Giọng em bé dễ thương",
    },
    "nguoi_ngoai_hanh_tinh": {
        "name": "Người ngoài hành tinh",
        "model": "alien.pth",
        "index": "alien.index",
        "description": "Giọng khác thường",
    },
    "robot": {
        "name": "Robot",
        "model": "robot.pth",
        "index": "robot.index",
        "description": "Giọng robot cơ khí",
    },
    "ac_quy": {
        "name": "Ác quỷ",
        "model": "demon.pth",
        "index": "demon.index",
        "description": "Giọng trầm rùng rợn",
    },
}

# ───── Loại file audio được hỗ trợ ─────
ALLOWED_AUDIO_TYPES = [
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/m4a",
    "audio/mp4",
    "audio/x-m4a",
    "audio/aac",
    "audio/ogg",
    "audio/webm",
    "application/octet-stream",  # Một số client gửi audio với type này
]

# ───── RVC inference settings ─────
RVC_SETTINGS = {
    "f0_method": "rmvpe",  # Phương pháp pitch detection tốt nhất
    "f0_up_key": 0,        # Không thay đổi cao độ mặc định (mỗi preset có thể override)
    "protect": 0.33,       # Bảo vệ giọng gốc (0-0.5)
    "filter_radius": 3,
    "resample_sr": 0,      # 0 = không resample
    "rms_mix_rate": 0.25,
    "index_rate": 0.75,    # Mức độ sử dụng index file
}
