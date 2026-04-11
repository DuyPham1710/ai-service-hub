"""
Script tải các model RVC từ Hugging Face cho voice conversion.
Chạy script này trước khi start Docker hoặc trong Dockerfile.

Usage:
    python download_models.py
"""

import os
import urllib.request
import sys

# Thư mục lưu model
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models", "voice_conversion")

# Danh sách model cần tải
# Format: (filename, url, description)
# Các model RVC v2 từ Hugging Face community
MODELS = [
    (
        "female.pth",
        "https://huggingface.co/Politrees/RVC_resources/resolve/main/predictors/rmvpe.pt",
        "Base RMVPE model (pitch detection)",
    ),
]

# =============================================
# LƯU Ý QUAN TRỌNG:
# =============================================
# Hiện tại chưa có model RVC cố định cho từng preset giọng.
# Bạn cần tự tìm và tải model phù hợp từ:
#   - https://huggingface.co (tìm "RVC model")
#   - https://weights.gg
#   - https://voice-models.com
#
# Sau khi tải, đặt file .pth vào models/voice_conversion/ với tên:
#   female.pth, deep_male.pth, baby.pth, alien.pth, robot.pth, demon.pth
#
# Script này chỉ tải model RMVPE (pitch detection) cần thiết cho RVC.
# =============================================

# Model RMVPE (bắt buộc cho RVC inference)
RMVPE_URL = "https://huggingface.co/Politrees/RVC_resources/resolve/main/predictors/rmvpe.pt"
HUBERT_URL = "https://huggingface.co/Politrees/RVC_resources/resolve/main/embedders/contentvec_base.pt"


def download_file(url: str, dest_path: str, description: str = ""):
    """Tải file từ URL với progress bar."""
    if os.path.exists(dest_path):
        print(f"  ✅ Đã có: {os.path.basename(dest_path)} ({description})")
        return

    print(f"  ⬇️  Đang tải: {os.path.basename(dest_path)} ({description})...")

    try:
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(
                    f"\r      {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
                )
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print(f"\n  ✅ Hoàn tất: {os.path.basename(dest_path)}")
    except Exception as e:
        print(f"\n  ❌ Lỗi tải {os.path.basename(dest_path)}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)


def main():
    print("=" * 60)
    print("📦 Tải model cho Voice Conversion (RVC)")
    print("=" * 60)

    # Tạo thư mục models
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"\n📁 Thư mục model: {MODELS_DIR}")

    # Tải model RMVPE (pitch detection - bắt buộc)
    print("\n🔧 Tải model cơ sở (bắt buộc cho RVC):")
    rvc_base_dir = os.path.join(os.path.dirname(__file__), "models", "rvc_base")
    os.makedirs(rvc_base_dir, exist_ok=True)

    download_file(
        RMVPE_URL,
        os.path.join(rvc_base_dir, "rmvpe.pt"),
        "RMVPE pitch detection model",
    )
    download_file(
        HUBERT_URL,
        os.path.join(rvc_base_dir, "contentvec_base.pt"),
        "ContentVec base embedder",
    )

    # Kiểm tra voice models
    print("\n🎙️ Kiểm tra voice models (preset giọng):")
    presets = {
        "female.pth": "Con gái",
        "deep_male.pth": "Trầm",
        "baby.pth": "Em bé",
        "alien.pth": "Người ngoài hành tinh",
        "robot.pth": "Robot",
        "demon.pth": "Ác quỷ",
    }

    missing = []
    for filename, name in presets.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✅ {name}: {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {name}: {filename} — CHƯA CÓ")
            missing.append((filename, name))

    if missing:
        print("\n" + "=" * 60)
        print("⚠️  CÁC MODEL CHƯA CÓ:")
        print("=" * 60)
        print("Bạn cần tự tải model RVC (.pth) cho từng preset giọng.")
        print(f"Đặt file vào: {MODELS_DIR}")
        print()
        print("Tìm model tại:")
        print("  🔗 https://huggingface.co (tìm 'RVC model')")
        print("  🔗 https://weights.gg")
        print("  🔗 https://voice-models.com")
        print()
        for filename, name in missing:
            print(f"  - {filename} ← model cho giọng '{name}'")
    else:
        print("\n✅ Tất cả model đã sẵn sàng!")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
