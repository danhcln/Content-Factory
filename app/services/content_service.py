import json
import logging
import re
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.models import Video, Voice, Content, Publishing
from app.services.gemini_service import (
    GeminiService,
    get_api_key,
    get_model_name,
    clean_json_response,
    GeminiQuotaExceededError
)

logger = logging.getLogger("app.services.content")

REQUIRED_PLATFORMS = [
    "facebook_personal",
    "facebook_page",
    "tiktok",
    "threads",
    "instagram",
    "shopee",
    "youtube"
]


def validate_content_json(json_data: Any) -> Dict[str, Any]:
    """
    Validate that the response JSON contains all 7 required platforms
    with their appropriate subfields.
    """
    if not isinstance(json_data, dict):
        return {"valid": False, "error": "Dữ liệu trả về không phải JSON object."}

    for plat in REQUIRED_PLATFORMS:
        if plat not in json_data or not isinstance(json_data[plat], dict):
            return {"valid": False, "error": f"Thiếu cấu trúc cho nền tảng: {plat}"}

    # Verify youtube has title, description, hashtags
    yt = json_data["youtube"]
    if "title" not in yt or "description" not in yt:
        return {"valid": False, "error": "Nền tảng YouTube thiếu 'title' hoặc 'description'."}

    # Verify other platforms have caption, hashtags
    for plat in ["facebook_personal", "facebook_page", "tiktok", "threads", "instagram", "shopee"]:
        item = json_data[plat]
        if "caption" not in item:
            return {"valid": False, "error": f"Nền tảng {plat} thiếu trường 'caption'."}

    return {"valid": True}


class ContentService:
    def __init__(self):
        pass

    def build_prompt(self, script_text: str, product_name: str = "") -> str:
        return f"""Bạn là một chuyên gia nội dung đa nền tảng cho video ngắn.
Hãy dựa DUY NHẤT vào kịch bản (Script) sau đây để tạo nội dung đăng bài cho 7 NỀN TẢNG KHÁC NHAU.
Sản phẩm: {product_name or 'Sản phẩm trong kịch bản'}

KỊCH BẢN GỐC (Nguồn duy nhất):
\"\"\"{script_text}\"\"\"

QUY TẮC BẮT BUỘC:
1. Tất cả caption phải NGẮN GỌN, súc tích, đi thẳng vào vấn đề.
2. NỘI DUNG CHỈ ĐƯỢC PHÉP RÚT RA TỪ KỊCH BẢN GỐC Ở TRÊN.
   TUYỆT ĐỐI KHÔNG tự ý bịa đặt: giá tiền, khuyến mãi, giảm giá sốc, bảo hành hay công dụng kỹ thuật nếu kịch bản không đề cập.
3. Phong cách từng nền tảng:
   - facebook_personal: Ngắn, giọng điệu chia sẻ cá nhân tự nhiên, đời thường + hashtags.
   - facebook_page: Ngắn, rõ ràng, tập trung vào giải pháp sản phẩm + hashtags.
   - tiktok: Cực ngắn, mở đầu lôi cuốn, hashtags chuẩn short-video.
   - threads: Ngắn, tự nhiên như lời tâm sự hoặc đặt câu hỏi, ít hashtags (tối đa 1-2).
   - instagram: Ngắn gọn, thẩm mỹ + bộ hashtag liên quan.
   - shopee: Cực ngắn, tập trung vào công năng sản phẩm, không bịa giá + hashtags.
   - youtube: "title" ngắn gọn giật tít chuẩn Shorts (dưới 70 ký tự), "description" cực ngắn, kèm "hashtags".

ĐỊNH DẠNG ĐẦU RA:
CHỈ TRẢ VỀ DUY NHẤT MỘT ĐỐI TƯỢNG JSON HỢP LỆ THEO ĐÚNG CẤU TRÚC SAU:
{{
  "facebook_personal": {{
    "caption": "...",
    "hashtags": "#..."
  }},
  "facebook_page": {{
    "caption": "...",
    "hashtags": "#..."
  }},
  "tiktok": {{
    "caption": "...",
    "hashtags": "#..."
  }},
  "threads": {{
    "caption": "...",
    "hashtags": "#..."
  }},
  "instagram": {{
    "caption": "...",
    "hashtags": "#..."
  }},
  "shopee": {{
    "caption": "...",
    "hashtags": "#..."
  }},
  "youtube": {{
    "title": "...",
    "description": "...",
    "hashtags": "#..."
  }}
}}
"""

    def generate_content_for_video(
        self,
        db: Session,
        video_id: str,
        regenerate: bool = False,
        max_retries: int = 2,
        mocked_response: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate captions and hashtags for all 7 platforms in ONE single Gemini call.
        Saves to CONTENT and PUBLISHING tables and transitions status to CONTENT_READY.
        Caches and avoids duplicate calls if already CONTENT_READY unless regenerate=True.
        """
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            return {"success": False, "error": f"Không tìm thấy video {video_id}"}

        # Cache check: if already CONTENT_READY and not requested to regenerate, reuse cached content!
        content_rec = db.query(Content).filter(Content.video_id == video_id).first()
        if not regenerate and content_rec and content_rec.status == "CONTENT_READY" and video.status == "CONTENT_READY":
            logger.info(f"Video {video_id} already has content (CONTENT_READY). Reusing cached content.")
            return {
                "success": True,
                "video_id": video_id,
                "status": "CONTENT_READY",
                "cached": True,
                "content": {
                    "facebook_personal": {"caption": content_rec.facebook_personal_caption or "", "hashtags": content_rec.facebook_personal_hashtags or ""},
                    "facebook_page": {"caption": content_rec.facebook_page_caption or "", "hashtags": content_rec.facebook_page_hashtags or ""},
                    "tiktok": {"caption": content_rec.tiktok_caption or "", "hashtags": content_rec.tiktok_hashtags or ""},
                    "threads": {"caption": content_rec.threads_caption or "", "hashtags": content_rec.threads_hashtags or ""},
                    "instagram": {"caption": content_rec.instagram_caption or "", "hashtags": content_rec.instagram_hashtags or ""},
                    "shopee": {"caption": content_rec.shopee_caption or "", "hashtags": content_rec.shopee_hashtags or ""},
                    "youtube": {"title": content_rec.youtube_title or "", "description": content_rec.youtube_description or "", "hashtags": content_rec.youtube_hashtags or ""}
                }
            }

        voice = db.query(Voice).filter(Voice.video_id == video_id).first()
        if not voice or not voice.script or not voice.script.strip():
            return {"success": False, "error": f"Video {video_id} chưa có kịch bản."}

        script_text = voice.script.strip()
        product_name = video.product.name_vietnamese if video.product else ""

        # If mocked response passed in (e.g. unit test or offline mode)
        if mocked_response:
            parsed_data = mocked_response
            val_res = validate_content_json(parsed_data)
            if not val_res["valid"]:
                return {"success": False, "error": val_res["error"]}
        else:
            api_key = get_api_key()
            if not api_key:
                video.status = "CONTENT_ERROR"
                video.notes = "Gemini API key chưa được cấu hình."
                db.commit()
                return {
                    "success": False,
                    "status": "BLOCKED_EXTERNAL",
                    "error": "Gemini API key chưa được cấu hình. Vui lòng nhập key trong Settings."
                }

            model_name = get_model_name()
            prompt = self.build_prompt(script_text, product_name)

            parsed_data = None
            last_error = ""

            gemini = GeminiService()
            for attempt in range(max_retries + 1):
                try:
                    raw_text = gemini.call_gemini(prompt, db=db, timeout=60.0, max_retries=max_retries)
                    cleaned = clean_json_response(raw_text)
                    candidate = json.loads(cleaned)
                    val_res = validate_content_json(candidate)
                    if val_res["valid"]:
                        parsed_data = candidate
                        break
                    else:
                        last_error = val_res["error"]
                        logger.warning(f"Attempt {attempt + 1} validation failed: {last_error}")
                except GeminiQuotaExceededError as qe:
                    logger.error(f"Gemini daily quota exceeded during content generation for {video_id}: {qe}")
                    video.status = "GEMINI_QUOTA_EXCEEDED"
                    video.notes = "Gemini daily quota has been reached. Try again after quota reset or use a project with sufficient quota."
                    db.commit()
                    return {
                        "success": False,
                        "status": "GEMINI_QUOTA_EXCEEDED",
                        "error": "Gemini daily quota has been reached. Try again after quota reset or use a project with sufficient quota."
                    }
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Attempt {attempt + 1} Gemini call/parse failed: {last_error}")

            if not parsed_data:
                video.status = "CONTENT_ERROR"
                video.notes = f"Không thể tạo nội dung sau {max_retries + 1} lần thử: {last_error}"
                db.commit()
                return {
                    "success": False,
                    "status": "CONTENT_ERROR",
                    "error": f"Lỗi tạo nội dung: {last_error}"
                }

        # Save to CONTENT table
        content_rec = db.query(Content).filter(Content.video_id == video_id).first()
        if not content_rec:
            content_rec = Content(video_id=video_id)
            db.add(content_rec)

        content_rec.facebook_personal_caption = parsed_data["facebook_personal"].get("caption", "")
        content_rec.facebook_personal_hashtags = parsed_data["facebook_personal"].get("hashtags", "")

        content_rec.facebook_page_caption = parsed_data["facebook_page"].get("caption", "")
        content_rec.facebook_page_hashtags = parsed_data["facebook_page"].get("hashtags", "")

        content_rec.tiktok_caption = parsed_data["tiktok"].get("caption", "")
        content_rec.tiktok_hashtags = parsed_data["tiktok"].get("hashtags", "")

        content_rec.threads_caption = parsed_data["threads"].get("caption", "")
        content_rec.threads_hashtags = parsed_data["threads"].get("hashtags", "")

        content_rec.instagram_caption = parsed_data["instagram"].get("caption", "")
        content_rec.instagram_hashtags = parsed_data["instagram"].get("hashtags", "")

        content_rec.shopee_caption = parsed_data["shopee"].get("caption", "")
        content_rec.shopee_hashtags = parsed_data["shopee"].get("hashtags", "")

        content_rec.youtube_title = parsed_data["youtube"].get("title", "")
        content_rec.youtube_description = parsed_data["youtube"].get("description", "")
        content_rec.youtube_hashtags = parsed_data["youtube"].get("hashtags", "")

        content_rec.status = "CONTENT_READY"

        # Initialize Publishing record if not exists
        pub_rec = db.query(Publishing).filter(Publishing.video_id == video_id).first()
        if not pub_rec:
            pub_rec = Publishing(
                video_id=video_id,
                published_count=0,
                status="NOT PUBLISHED"
            )
            db.add(pub_rec)

        video.status = "CONTENT_READY"
        video.notes = None
        db.commit()
        db.refresh(content_rec)
        db.refresh(video)

        logger.info(f"7-platform content saved for {video_id}. Status: CONTENT_READY")
        return {
            "success": True,
            "video_id": video_id,
            "status": "CONTENT_READY",
            "content": parsed_data
        }
