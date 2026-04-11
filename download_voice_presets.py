import os
import urllib.request
import sys
import shutil

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models", "voice_conversion")

# Một model giọng nữ Nhật Bản (V2-AISO-HOWATTO) chất lượng cao để làm mẫu:
FEMALE_PTH_URL = "https://huggingface.co/aa444rt/RVC_V2_models_5_japanese_womens/resolve/main/V2-AISO-HOWATTO.pth"
FEMALE_INDEX_URL = "https://huggingface.co/aa444rt/RVC_V2_models_5_japanese_womens/resolve/main/Index-file-Please-Set-Index-Rate0/trained_IVF832_Flat_nprobe_1_v2.index"

# 6 preset chúng ta cần chuẩn bị:
PRESETS = [
    ("female.pth", "female.index"),
    ("deep_male.pth", "deep_male.index"),
    ("baby.pth", "baby.index"),
    ("alien.pth", "alien.index"),
    ("robot.pth", "robot.index"),
    ("demon.pth", "demon.index"),
]

def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path):
        return True
    
    print(f"  ⬇️  Đang tải model mẫu: {os.path.basename(dest_path)}...")
    try:
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 // total_size)
                sys.stdout.write(f"\r      {percent}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print(f"\n  ✅ Hoàn tất tải mẫu: {os.path.basename(dest_path)}")
        return True
    except Exception as e:
        print(f"\n  ❌ Lỗi tải {os.path.basename(dest_path)}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Tải 1 model mẫu (giọng nữ)
    base_pth = os.path.join(MODELS_DIR, "female.pth")
    base_idx = os.path.join(MODELS_DIR, "female.index")
    
    success_pth = download_file(FEMALE_PTH_URL, base_pth)
    success_idx = download_file(FEMALE_INDEX_URL, base_idx)
    
    if not success_pth:
        print("Tải model mẫu thất bại, vui lòng tải thủ công.")
        return

    # 2. Copy model mẫu cho các preset còn lại để app chạy được (không lỗi)
    print("\n  🔄 Đang nhân bản model mẫu làm placeholder cho các option còn lại...")
    for pth_file, idx_file in PRESETS[1:]: # Bỏ qua cái đầu tiên (female)
        target_pth = os.path.join(MODELS_DIR, pth_file)
        target_idx = os.path.join(MODELS_DIR, idx_file)
        
        if not os.path.exists(target_pth):
            shutil.copy2(base_pth, target_pth)
            print(f"      + Đã tạo placeholder: {pth_file}")
            
        if success_idx and not os.path.exists(target_idx):
            shutil.copy2(base_idx, target_idx)
            print(f"      + Đã tạo placeholder index: {idx_file}")

    print("\n✅ HOÀN TẤT SETUP MODEL MẪU!")
    print("=" * 60)
    print(" LƯU Ý QUAN TRỌNG: ")
    print(" Tôi đã dùng TẠM 1 giọng nữ (Female_1) làm mẫu chung cho CẢ 6 tuỳ chọn.")
    print(" Điều này giúp tính năng chạy mượt mà ngay bây giờ (bạn chọn 'Robot' hay 'Trầm'")
    print(" thì nó vẫn ra giọng nữ đó) để tránh văng lỗi app.")
    print(" Về sau, bạn tải file .pth của giọng Robot thật và ghi đè vào file robot.pth là xong!")
    print("=" * 60)

if __name__ == "__main__":
    main()
