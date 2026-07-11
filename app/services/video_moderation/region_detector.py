"""
Region Detector: Sử dụng CLIP GradCAM để xác định vùng vi phạm trong frame.

Thay vì blur toàn bộ frame, module này xác định chính xác VÙNG NÀO trong ảnh
khiến CLIP phân loại là vi phạm, rồi trả về bounding boxes để blur chỉ vùng đó.

Kỹ thuật: GradCAM (Gradient-weighted Class Activation Mapping)
- Chạy forward pass CLIP với ảnh + label vi phạm
- Lấy gradient của similarity score đối với feature map cuối cùng của Vision Transformer
- Gradient * activation = attention map → cho biết vùng nào "quan trọng" nhất cho label đó
"""
import logging
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from ..image_moderation.model import clip_model, clip_processor
from .config import GRADCAM_THRESHOLD, BBOX_PADDING_RATIO

logger = logging.getLogger("ai-service-hub")


def get_violation_regions(
    image: Image.Image,
    violation_label: str,
) -> list[dict]:
    """
    Xác định các vùng vi phạm trong ảnh bằng CLIP GradCAM.

    Args:
        image: PIL Image cần phân tích
        violation_label: Label vi phạm (vd: "nudity and sexual content")

    Returns:
        Danh sách bounding boxes: [{"x": int, "y": int, "w": int, "h": int}]
        Tọa độ tính theo kích thước ảnh gốc.
    """
    try:
        heatmap = _compute_gradcam(image, violation_label)
        if heatmap is None:
            return []

        bboxes = _heatmap_to_bboxes(heatmap, image.size)
        return bboxes

    except Exception as e:
        logger.warning(f"Không thể detect region cho '{violation_label}': {e}")
        return []


def _compute_gradcam(image: Image.Image, text_label: str) -> Optional[np.ndarray]:
    """
    Tính GradCAM attention map từ CLIP Vision Transformer.

    GradCAM cho ViT:
    1. Hook vào layer cuối cùng của vision encoder (encoder.layers[-1].layer_norm1)
    2. Forward pass: tính similarity(image, text)
    3. Backward pass: lấy gradient của similarity đối với activation
    4. attention_map = mean(gradient * activation, dim=channel)
    5. ReLU → chỉ giữ positive influence

    Returns:
        Numpy array 2D (spatial_h x spatial_w) đã normalize [0, 1], hoặc None nếu lỗi.
    """
    # Prepare inputs
    inputs = clip_processor(
        text=[text_label],
        images=image,
        return_tensors="pt",
        padding=True,
    )

    # Storage cho hook
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    # Hook vào layer cuối cùng của vision encoder
    # CLIP ViT structure: clip_model.vision_model.encoder.layers[-1].layer_norm1
    target_layer = clip_model.vision_model.encoder.layers[-1].layer_norm1

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Forward pass với gradient enabled
        outputs = clip_model(**inputs)

        # Similarity score giữa image và text label vi phạm
        similarity = outputs.logits_per_image[0, 0]

        # Backward pass
        clip_model.zero_grad()
        similarity.backward(retain_graph=False)

        if "value" not in activations or "value" not in gradients:
            logger.warning("GradCAM: Không capture được activations/gradients")
            return None

        # Lấy activation và gradient
        act = activations["value"].detach()  # (1, num_patches+1, hidden_dim)
        grad = gradients["value"].detach()   # (1, num_patches+1, hidden_dim)

        # Bỏ CLS token (patch đầu tiên), chỉ giữ spatial patches
        act = act[:, 1:, :]   # (1, num_patches, hidden_dim)
        grad = grad[:, 1:, :]  # (1, num_patches, hidden_dim)

        # GradCAM: mean(gradient, dim=-1) * mean(activation, dim=-1)
        # Hoặc đơn giản hơn: mean(gradient * activation, dim=-1)
        cam = (grad * act).mean(dim=-1)  # (1, num_patches)
        cam = torch.relu(cam)  # Chỉ giữ positive influence

        cam = cam.squeeze(0).cpu().numpy()  # (num_patches,)

        # Reshape thành 2D spatial map
        # CLIP ViT-B/32: input 224x224, patch_size 32 → 7x7 = 49 patches
        num_patches = cam.shape[0]
        spatial_size = int(np.sqrt(num_patches))

        if spatial_size * spatial_size != num_patches:
            logger.warning(
                f"GradCAM: num_patches={num_patches} không phải perfect square"
            )
            return None

        cam_2d = cam.reshape(spatial_size, spatial_size)

        # Normalize [0, 1]
        cam_max = cam_2d.max()
        if cam_max > 0:
            cam_2d = cam_2d / cam_max
        else:
            return None  # Không có activation nào → không detect được vùng

        return cam_2d

    finally:
        # Luôn gỡ hook để tránh memory leak
        fwd_handle.remove()
        bwd_handle.remove()


def _heatmap_to_bboxes(
    heatmap: np.ndarray,
    image_size: tuple[int, int],
) -> list[dict]:
    """
    Chuyển GradCAM heatmap thành danh sách bounding boxes.

    Args:
        heatmap: Numpy array 2D (spatial_h, spatial_w), đã normalize [0, 1]
        image_size: (width, height) của ảnh gốc

    Returns:
        Danh sách bounding boxes: [{"x": int, "y": int, "w": int, "h": int}]
    """
    img_w, img_h = image_size

    # Upscale heatmap về kích thước ảnh gốc (bilinear interpolation)
    heatmap_resized = cv2.resize(
        heatmap.astype(np.float32),
        (img_w, img_h),
        interpolation=cv2.INTER_LINEAR,
    )

    # Threshold: lấy các vùng có attention >= GRADCAM_THRESHOLD
    # Hạ ngưỡng nhẹ nếu cần để bắt được nhiều nét hơn, nhưng vẫn dựa trên config
    binary_mask = (heatmap_resized >= GRADCAM_THRESHOLD).astype(np.uint8) * 255

    # Sử dụng Canny edge detection kết hợp với mask để làm vùng nhận diện bám sát hơn (tùy chọn)
    # Nhưng cách ổn định nhất là dùng threshold Otsu trên vùng GradCAM đã lọc
    heatmap_norm = (heatmap_resized * 255).astype(np.uint8)
    _, otsu_mask = cv2.threshold(heatmap_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Kết hợp cả 2 mask: lấy giao hoặc hợp tùy chiến lược, ở đây lấy hợp để không bị sót
    combined_mask = cv2.bitwise_or(binary_mask, otsu_mask)

    # Morphological operations: dilate mạnh hơn để nối các mảng của cùng 1 object (ví dụ: tay và súng)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80)) # Đã tăng từ 50 lên 80 để vùng blur to ra
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    # Tìm contours
    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bboxes = []
    min_area = img_w * img_h * 0.02  # Tăng diện tích tối thiểu lên 2% để lọc bỏ nhiễu lốm đốm

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area < min_area:
            continue  # Bỏ qua nhiễu nhỏ

        # Thêm padding xung quanh bbox (tăng padding để chắc chắn che hết)
        # Vì GradCAM chỉ là attention (thường tập trung vào tâm vật thể), padding lớn một chút sẽ che được toàn bộ
        pad_w = int(w * (BBOX_PADDING_RATIO + 0.1)) # Tăng thêm 10% padding
        pad_h = int(h * (BBOX_PADDING_RATIO + 0.1))

        x = max(0, x - pad_w)
        y = max(0, y - pad_h)
        w = min(img_w - x, w + 2 * pad_w)
        h = min(img_h - y, h + 2 * pad_h)

        bboxes.append({"x": x, "y": y, "w": w, "h": h})

    # if bboxes:
    #     logger.debug(
    #         f"GradCAM detected {len(bboxes)} region(s): "
    #         f"{[f'{b[\"w\"]}x{b[\"h\"]} at ({b[\"x\"]},{b[\"y\"]})' for b in bboxes]}"
    #     )

    return bboxes
