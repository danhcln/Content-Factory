import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("app.services.subtitle_region")

PRESETS: Dict[str, Dict[str, float]] = {
    "BOTTOM": {
        "x": 0.05,
        "y": 0.72,
        "width": 0.90,
        "height": 0.20
    },
    "LOWER-MIDDLE": {
        "x": 0.05,
        "y": 0.60,
        "width": 0.90,
        "height": 0.20
    },
    "MIDDLE": {
        "x": 0.05,
        "y": 0.40,
        "width": 0.90,
        "height": 0.20
    }
}


def validate_coordinates(region: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate normalized coordinates (0.0 to 1.0).
    Clamps bounds safely and returns sanitized dictionary.
    """
    try:
        x = float(region.get("x", 0.05))
        y = float(region.get("y", 0.72))
        w = float(region.get("width", 0.90))
        h = float(region.get("height", 0.20))
    except (ValueError, TypeError):
        return {
            "valid": False,
            "error": "Tọa độ phải là các số thực hợp lệ (0.0 - 1.0)."
        }

    # Bounds checking
    if x < 0.0 or x > 0.95 or y < 0.0 or y > 0.95:
        return {
            "valid": False,
            "error": "Tọa độ x, y phải nằm trong khoảng từ 0.0 đến 0.95."
        }

    if w <= 0.05 or w > 1.0 or h <= 0.02 or h > 1.0:
        return {
            "valid": False,
            "error": "Chiều rộng và chiều cao phải lớn hơn 0 và không vượt quá 1.0."
        }

    # Clamp bounds so x + w <= 1.0 and y + h <= 1.0
    if x + w > 1.0:
        w = round(1.0 - x, 4)
    if y + h > 1.0:
        h = round(1.0 - y, 4)

    return {
        "valid": True,
        "region": {
            "x": round(x, 4),
            "y": round(y, 4),
            "width": round(w, 4),
            "height": round(h, 4)
        }
    }


def get_preset_region(preset_name: str) -> Dict[str, float]:
    """Retrieve normalized coordinates for a preset."""
    name = preset_name.upper().strip()
    return PRESETS.get(name, PRESETS["BOTTOM"]).copy()


def build_blur_filter(
    region: Dict[str, float],
    video_width: int,
    video_height: int,
    strength: int = 10
) -> str:
    """
    Construct an FFmpeg complex filter string to blur only the subtitle region,
    preserving the surrounding product and video content.
    """
    strength = max(3, min(strength, 35))
    rx = max(0.0, min(region.get("x", 0.05), 0.95))
    ry = max(0.0, min(region.get("y", 0.72), 0.95))
    rw = max(0.05, min(region.get("width", 0.90), 1.0 - rx))
    rh = max(0.02, min(region.get("height", 0.20), 1.0 - ry))

    # Convert to pixel coordinates (even numbers for encoder compatibility)
    crop_x = int(video_width * rx)
    crop_y = int(video_height * ry)
    crop_w = int(video_width * rw)
    crop_h = int(video_height * rh)

    # Ensure even dimensions
    if crop_w % 2 != 0:
        crop_w += 1
    if crop_h % 2 != 0:
        crop_h += 1
    if crop_x + crop_w > video_width:
        crop_w = video_width - crop_x
    if crop_y + crop_h > video_height:
        crop_h = video_height - crop_y

    return (
        f"[0:v]split[main][sub];"
        f"[sub]crop=w={crop_w}:h={crop_h}:x={crop_x}:y={crop_y},"
        f"boxblur=luma_radius={strength}:luma_power=2[blurred];"
        f"[main][blurred]overlay=x={crop_x}:y={crop_y}"
    )


def build_cover_filter(
    region: Dict[str, float],
    video_width: int,
    video_height: int,
    opacity: float = 0.85,
    color: str = "black"
) -> str:
    """
    Construct an FFmpeg filter string to cover the subtitle region with a semi-opaque box.
    """
    opacity = max(0.2, min(opacity, 1.0))
    rx = max(0.0, min(region.get("x", 0.05), 0.95))
    ry = max(0.0, min(region.get("y", 0.72), 0.95))
    rw = max(0.05, min(region.get("width", 0.90), 1.0 - rx))
    rh = max(0.02, min(region.get("height", 0.20), 1.0 - ry))

    crop_x = int(video_width * rx)
    crop_y = int(video_height * ry)
    crop_w = int(video_width * rw)
    crop_h = int(video_height * rh)

    return f"drawbox=x={crop_x}:y={crop_y}:w={crop_w}:h={crop_h}:color={color}@{opacity:.2f}:t=fill"


class SubtitleRegionDetector:
    """Best-effort local computer vision detection of burned-in Chinese subtitle regions."""

    @classmethod
    def auto_detect(
        cls,
        video_path: Path,
        duration: float = 10.0,
        sample_count: int = 5
    ) -> Dict[str, Any]:
        """
        Sample representative video frames and detect dense horizontal text regions
        in the lower half of the frame.
        If confidence is insufficient, sets SUBTITLE_REGION_REVIEW_REQUIRED and does NOT guess.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.warning("OpenCV not available for auto subtitle detection.")
            return {
                "detected": False,
                "status": "SUBTITLE_REGION_REVIEW_REQUIRED",
                "region": get_preset_region("BOTTOM"),
                "confidence": 0.0,
                "sampled_frames": 0,
                "reason": "OpenCV chưa được nạp. Sử dụng vùng mặc định BOTTOM."
            }

        video_path = Path(video_path)
        if not video_path.exists():
            return {
                "detected": False,
                "status": "SUBTITLE_REGION_REVIEW_REQUIRED",
                "region": get_preset_region("BOTTOM"),
                "confidence": 0.0,
                "sampled_frames": 0,
                "reason": "Không tìm thấy tệp video."
            }

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {
                "detected": False,
                "status": "SUBTITLE_REGION_REVIEW_REQUIRED",
                "region": get_preset_region("BOTTOM"),
                "confidence": 0.0,
                "sampled_frames": 0,
                "reason": "Không thể mở luồng video bằng OpenCV."
            }

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return {
                "detected": False,
                "status": "SUBTITLE_REGION_REVIEW_REQUIRED",
                "region": get_preset_region("BOTTOM"),
                "confidence": 0.0,
                "sampled_frames": 0,
                "reason": "Số lượng khung hình không xác định."
            }

        # Select representative timestamps across video duration
        # Focus on frames from 15% to 85% of duration
        step = max(1, total_frames // (sample_count + 1))
        sample_indices = [step * i for i in range(1, sample_count + 1) if step * i < total_frames]

        detected_bands: List[Tuple[float, float]] = []
        sampled_valid = 0

        for f_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            sampled_valid += 1
            h, w = frame.shape[:2]

            # Chinese subtitles in vertical video are primarily in lower half: y from 0.60 to 0.92
            roi_y_start = int(h * 0.60)
            roi_y_end = int(h * 0.92)
            roi = frame[roi_y_start:roi_y_end, :]

            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Sobel horizontal gradient to detect text character edges
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            abs_grad_x = cv2.convertScaleAbs(grad_x)

            # Morphological close to bridge character components
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            closed = cv2.morphologyEx(abs_grad_x, cv2.MORPH_CLOSE, kernel)

            # Threshold
            _, thresh = cv2.threshold(closed, 50, 255, cv2.THRESH_BINARY)

            # Project along horizontal axis (row density)
            row_densities = np.sum(thresh, axis=1) / (w * 255.0)

            # Find band with peak density
            max_val = np.max(row_densities) if len(row_densities) > 0 else 0.0
            if max_val > 0.15:
                # Find contiguous rows with density > 0.08
                active_rows = np.where(row_densities > 0.08)[0]
                if len(active_rows) > 10:
                    band_top = (roi_y_start + active_rows[0]) / float(h)
                    band_bottom = (roi_y_start + active_rows[-1]) / float(h)
                    detected_bands.append((band_top, band_bottom))

        cap.release()

        # If at least 3 sampled frames agree on subtitle location
        if len(detected_bands) >= max(2, sample_count // 2):
            avg_top = sum(b[0] for b in detected_bands) / len(detected_bands)
            avg_bottom = sum(b[1] for b in detected_bands) / len(detected_bands)

            # Add padding
            pad_y = 0.03
            final_y = max(0.50, avg_top - pad_y)
            final_h = min(0.35, (avg_bottom - avg_top) + (2 * pad_y))

            confidence = round(len(detected_bands) / float(sampled_valid), 2)

            # If confidence is good (>= 0.60)
            if confidence >= 0.60:
                logger.info(f"Auto-detected subtitle region: y={final_y:.2f}, h={final_h:.2f} (confidence: {confidence})")
                return {
                    "detected": True,
                    "status": "CONFIRMED",
                    "region": {
                        "x": 0.05,
                        "y": round(final_y, 4),
                        "width": 0.90,
                        "height": round(final_h, 4)
                    },
                    "confidence": confidence,
                    "sampled_frames": sampled_valid,
                    "supporting_frames": len(detected_bands),
                    "reason": f"Phát hiện vùng phụ đề với độ tin cậy {int(confidence * 100)}% ({len(detected_bands)}/{sampled_valid} khung hình)."
                }

        # Insufficient confidence -> DO NOT GUESS!
        logger.warning("Subtitle region detection confidence insufficient. Requires user confirmation.")
        return {
            "detected": False,
            "status": "SUBTITLE_REGION_REVIEW_REQUIRED",
            "region": get_preset_region("BOTTOM"),
            "confidence": 0.40,
            "sampled_frames": sampled_valid,
            "supporting_frames": len(detected_bands),
            "reason": "Độ tin cậy nhận diện tự động chưa đủ cao. Vui lòng xác nhận hoặc điều chỉnh vùng phụ đề."
        }
