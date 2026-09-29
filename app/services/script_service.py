import logging
import re
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.models import Video, Product, Voice
from app.services.ai import get_ai_manager, AIQuotaExceededError
from app.services.gemini_service import (
    GeminiService,
    get_api_key,
    get_model_name,
    clean_json_response,
    GeminiQuotaExceededError
)

logger = logging.getLogger("app.services.script")

PLACEHOLDER_KEYWORDS = [
    "[placeholder]", "kịch bản ở đây", "lời thoại ở đây", "chưa có kịch bản",
    "sample script", "lorem ipsum", "insert script here", "undefined", "null"
]


class ScriptService:
    def __init__(self):
        pass

    def validate_script(self, script_text: Optional[str]) -> Dict[str, Any]:
        """
        Deterministic Python validation for Vietnamese voice-over script:
        - Must not be empty or whitespace
        - Must have reasonable length (between 40 and 3000 characters)
        - Must not contain obvious placeholder markers
        - Must not be an unparsed JSON string or raw markdown error
        """
        if not script_text or not script_text.strip():
            return {
                "valid": False,
                "status": "SCRIPT_REVIEW_REQUIRED",
                "reason": "Kịch bản trống."
            }

        cleaned = script_text.strip()

        # Check length
        if len(cleaned) < 40:
            return {
                "valid": False,
                "status": "SCRIPT_REVIEW_REQUIRED",
                "reason": f"Kịch bản quá ngắn ({len(cleaned)} ký tự, tối thiểu 40 ký tự)."
            }

        if len(cleaned) > 3000:
            return {
                "valid": False,
                "status": "SCRIPT_REVIEW_REQUIRED",
                "reason": f"Kịch bản quá dài ({len(cleaned)} ký tự, tối đa 3000 ký tự)."
            }

        # Check placeholders
        lower_text = cleaned.lower()
        for kw in PLACEHOLDER_KEYWORDS:
            if kw in lower_text:
                return {
                    "valid": False,
                    "status": "SCRIPT_REVIEW_REQUIRED",
                    "reason": f"Kịch bản chứa từ khóa placeholder: '{kw}'."
                }

        # Check for unparsed raw JSON
        if (cleaned.startswith("{") and cleaned.endswith("}")) or (cleaned.startswith("[") and cleaned.endswith("]")):
            return {
                "valid": False,
                "status": "SCRIPT_ERROR",
                "reason": "Phản hồi chưa được xử lý (chuỗi JSON thô)."
            }

        # Check for error keywords
        if "error" in lower_text and ("traceback" in lower_text or "exception" in lower_text):
            return {
                "valid": False,
                "status": "SCRIPT_ERROR",
                "reason": "Phản hồi chứa thông điệp lỗi hệ thống."
            }

        return {
            "valid": True,
            "status": "SCRIPT_READY",
            "reason": "Kịch bản hợp lệ."
        }

    def generate_script_for_video(
        self,
        db: Session,
        video_id: str,
        regenerate: bool = False,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """Generate original Vietnamese voice-over script strictly tied to video_id."""
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            return {"success": False, "error": f"Không tìm thấy video {video_id}"}

        # Cache check: if valid script already exists and user did not request regeneration, reuse it!
        voice = db.query(Voice).filter(Voice.video_id == video_id).first()
        if not regenerate and voice and voice.script and voice.status == "SCRIPT_READY":
            logger.info(f"Video {video_id} already has a valid script (SCRIPT_READY). Reusing cached script.")
            return {
                "success": True,
                "video_id": video_id,
                "script": voice.script,
                "status": voice.status,
                "cached": True,
                "validation": {"valid": True, "status": "SCRIPT_READY", "reason": "Kịch bản đã có sẵn, tái sử dụng an toàn."}
            }

        # Fetch product details
        product = video.product
        prod_name = product.name_vietnamese if product else "Sản phẩm thông minh"
        content_angle = product.content_angle if product else ""
        hook = product.hook if product else ""

        api_key = get_api_key()
        if not api_key:
            return {
                "success": False,
                "status": "BLOCKED_EXTERNAL",
                "error": "Chưa cấu hình Gemini API Key. Vui lòng nhập key trong trang Settings."
            }

        model_name = get_model_name()
        prompt = f"""Bạn là một chuyên gia sáng tạo kịch bản video ngắn (TikTok, Reels, Shorts) triệu view.
Hãy viết một kịch bản LỜI ĐỌC (Voice-over) hoàn toàn NGUYÊN BẢN (ORIGINAL) bằng tiếng Việt cho video sau:
- Tên sản phẩm: {prod_name}
- Góc tiếp cận (Content Angle): {content_angle or 'Giải quyết vấn đề hàng ngày'}
- Câu giật tít mở đầu (Hook): {hook or 'Xem ngay món đồ tiện lợi này'}

YÊU CẦU QUAN TRỌNG:
1. Độ dài: Khoảng 80 - 150 từ (thời lượng đọc khoảng 30 - 50 giây).
2. Phong cách: Tự nhiên, cuốn hút, gần gũi, khơi gợi sự tò mò và nêu bật công dụng thực tế của sản phẩm.
3. CHỈ TRẢ VỀ DUY NHẤT LỜI ĐỌC NÓI RA THÀNH TIẾNG.
   KHÔNG thêm bất kỳ ghi chú đạo diễn nào như: [Nhạc nền], (Cảnh quay cận), [Hình ảnh], [Tiêu đề], Hook:, v.v.
   Để lời đọc có thể chuyển trực tiếp sang Text-to-Speech (TTS) đọc liền mạch.
"""
        raw_script = ""
        try:
            ai_manager = get_ai_manager()
            raw_script = ai_manager.generate(
                prompt,
                db=db,
                timeout=60.0,
                max_retries=max_retries,
                enable_fallback=True,
                allow_cross_provider_fallback=True
            )
        except (AIQuotaExceededError, GeminiQuotaExceededError) as qe:
            logger.error(f"Gemini daily quota exceeded during script generation for {video_id}: {qe}")
            video.status = "GEMINI_QUOTA_EXCEEDED"
            video.notes = "Gemini daily quota has been reached. Try again after quota reset or use a project with sufficient quota."
            db.commit()
            return {
                "success": False,
                "status": "GEMINI_QUOTA_EXCEEDED",
                "error": "Gemini daily quota has been reached. Try again after quota reset or use a project with sufficient quota."
            }
        except Exception as e:
            logger.error(f"Failed to generate script via Gemini: {e}")
            video.status = "SCRIPT_ERROR"
            video.notes = f"Lỗi Gemini: {str(e)}"
            db.commit()
            return {"success": False, "error": f"Lỗi gọi Gemini: {str(e)}", "status": "SCRIPT_ERROR"}

        cleaned = raw_script.strip()
        # Clean any accidental quotes
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()

        # Validate with deterministic Python code
        val_res = self.validate_script(cleaned)

        # Update or create Voice record strictly associated with video_id
        voice = db.query(Voice).filter(Voice.video_id == video_id).first()
        if not voice:
            voice = Voice(
                video_id=video_id,
                script=cleaned,
                tts_engine="VieNeu-TTS",
                status=val_res["status"]
            )
            db.add(voice)
        else:
            voice.script = cleaned
            voice.status = val_res["status"]

        # Update Video status
        video.status = val_res["status"]
        if not val_res["valid"]:
            video.notes = val_res["reason"]

        db.commit()
        db.refresh(voice)
        db.refresh(video)

        return {
            "success": val_res["valid"],
            "video_id": video_id,
            "script": cleaned,
            "status": val_res["status"],
            "validation": val_res
        }

    def edit_script(self, db: Session, video_id: str, new_script: str) -> Dict[str, Any]:
        """Allow manual editing of script and re-validate."""
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            return {"success": False, "error": f"Không tìm thấy video {video_id}"}

        voice = db.query(Voice).filter(Voice.video_id == video_id).first()
        cleaned = new_script.strip()
        val_res = self.validate_script(cleaned)

        if not voice:
            voice = Voice(
                video_id=video_id,
                script=cleaned,
                tts_engine="VieNeu-TTS",
                status=val_res["status"]
            )
            db.add(voice)
        else:
            voice.script = cleaned
            voice.status = val_res["status"]

        video.status = val_res["status"]
        if not val_res["valid"]:
            video.notes = val_res["reason"]
        else:
            video.notes = None

        db.commit()
        db.refresh(video)

        return {
            "success": val_res["valid"],
            "video_id": video_id,
            "script": cleaned,
            "status": val_res["status"],
            "validation": val_res
        }
