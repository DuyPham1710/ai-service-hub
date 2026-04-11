import os
import uuid
import logging
import tempfile

from .config import DEVICE, MODELS_DIR, VOICE_PRESETS, RVC_SETTINGS

logger = logging.getLogger("ai-service-hub")

# Lazy-load RVC engine
_rvc_engine = None


def _get_rvc_engine():
    """Lazy-load RVCInference engine (chỉ khởi tạo lần đầu gọi)."""
    global _rvc_engine
    if _rvc_engine is None:
        try:
            from rvc_python.infer import RVCInference
            _rvc_engine = RVCInference(device=DEVICE)
            logger.info(f"✅ RVC engine initialized on device: {DEVICE}")
        except Exception as e:
            logger.error(f"❌ Không thể khởi tạo RVC engine: {e}")
            raise RuntimeError(f"RVC engine initialization failed: {e}")
    return _rvc_engine


def get_available_presets() -> list[dict]:
    """Trả về danh sách các preset giọng có sẵn (chỉ những preset có model file)."""
    presets = []
    for key, info in VOICE_PRESETS.items():
        model_path = os.path.join(MODELS_DIR, info["model"])
        presets.append({
            "key": key,
            "name": info["name"],
            "description": info["description"],
            "available": os.path.exists(model_path),
        })
    return presets


def convert_voice(input_path: str, preset_key: str) -> str:
    """
    Chuyển đổi giọng nói trong file audio theo preset.

    Args:
        input_path: Đường dẫn file audio gốc
        preset_key: Key của preset (ví dụ: "con_gai", "tram",...)

    Returns:
        Đường dẫn file audio đã chuyển giọng

    Raises:
        ValueError: Nếu preset không tồn tại hoặc model chưa được tải
        RuntimeError: Nếu quá trình convert thất bại
    """
    # Validate preset
    if preset_key not in VOICE_PRESETS:
        available = list(VOICE_PRESETS.keys())
        raise ValueError(
            f"Preset '{preset_key}' không tồn tại. Các preset có sẵn: {available}"
        )

    preset = VOICE_PRESETS[preset_key]
    model_path = os.path.join(MODELS_DIR, preset["model"])

    # Check model file exists
    if not os.path.exists(model_path):
        raise ValueError(
            f"Model file cho preset '{preset_key}' chưa được tải. "
            f"Vui lòng đặt file '{preset['model']}' vào thư mục models/voice_conversion/"
        )

    # Tạo output path
    output_dir = tempfile.gettempdir()
    output_filename = f"vc_{uuid.uuid4().hex}.wav"
    output_path = os.path.join(output_dir, output_filename)

    try:
        rvc = _get_rvc_engine()

        # Load model
        logger.info(f"🔄 Loading model: {preset['model']} cho preset '{preset['name']}'")
        rvc.load_model(model_path)

        # Check & load index file nếu có
        index_path = os.path.join(MODELS_DIR, preset.get("index", ""))
        if os.path.exists(index_path):
            logger.info(f"📂 Index file found: {preset.get('index', '')}")

        # Thực hiện convert
        logger.info(f"🎙️ Converting voice: {input_path} → {output_path}")
        rvc.infer_file(
            input_path=input_path,
            output_path=output_path,
        )

        logger.info(f"✅ Voice conversion thành công: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"❌ Voice conversion thất bại: {e}")
        # Cleanup output file nếu tạo dở
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Voice conversion failed: {e}")
