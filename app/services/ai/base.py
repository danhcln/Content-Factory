"""
Core Provider Abstraction & Neutral Contracts for Multi-AI Architecture (Phase 2)

This module defines provider-neutral interfaces, options, execution metadata,
and exception hierarchies for both Direct BYOK and Gateway providers.
It contains ZERO provider-specific networking details.
"""
import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session


# ==============================================================================
# PROVIDER-NEUTRAL EXCEPTIONS
# ==============================================================================

class AIProviderError(RuntimeError):
    """Base exception for all AI provider operations, fully compatible with RuntimeError."""
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        upstream_message: str = "",
        error_type: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.provider = provider
        self.model = model
        self.upstream_message = upstream_message
        self.error_type = error_type or "unknown"


class AIAuthenticationError(AIProviderError):
    """Raised when authentication fails (HTTP 401 / invalid API key)."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_type", "authentication")
        kwargs.setdefault("status_code", 401)
        super().__init__(message, **kwargs)


class AIPermissionError(AIProviderError):
    """Raised when permission is denied / resource forbidden (HTTP 403)."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_type", "permission")
        kwargs.setdefault("status_code", 403)
        super().__init__(message, **kwargs)


class AIModelNotFoundError(AIProviderError):
    """Raised when requested model does not exist or is inaccessible (HTTP 404)."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_type", "model_not_found")
        kwargs.setdefault("status_code", 404)
        super().__init__(message, **kwargs)


class AIQuotaExceededError(AIProviderError):
    """Raised when daily or billing quota is exhausted. Must NOT be retried."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_type", "quota")
        kwargs.setdefault("status_code", 429)
        super().__init__(message, **kwargs)


class AIRateLimitError(AIProviderError):
    """Raised when temporary rate limit (RPM/TPM) is hit. Retryable."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_type", "rate_limit")
        kwargs.setdefault("status_code", 429)
        super().__init__(message, **kwargs)


class AIBadRequestError(AIProviderError):
    """Raised when request format is invalid (HTTP 400)."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_type", "bad_request")
        kwargs.setdefault("status_code", 400)
        super().__init__(message, **kwargs)


class AIServiceUnavailableError(AIProviderError):
    """Raised when upstream server is overloaded or temporarily unavailable (HTTP 500/502/503/504)."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_type", "service_unavailable")
        kwargs.setdefault("status_code", 503)
        super().__init__(message, **kwargs)


class AITimeoutError(AIProviderError):
    """Raised when request times out."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_type", "timeout")
        super().__init__(message, **kwargs)


class AINetworkError(AIProviderError):
    """Raised on network connection / DNS failure."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_type", "network")
        super().__init__(message, **kwargs)


class AIInvalidResponseError(AIProviderError):
    """Raised when response content is malformed, empty, or unparseable."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_type", "invalid_response")
        super().__init__(message, **kwargs)


class AIFallbackExhaustedError(AIProviderError):
    """Raised when all configured AI fallback providers have failed."""
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_type", "fallback_exhausted")
        kwargs.setdefault("status_code", 503)
        super().__init__(message, **kwargs)


# ==============================================================================
# PROVIDER-NEUTRAL OPTIONS & METADATA
# ==============================================================================

@dataclass
class AIGenerationOptions:
    """Provider-neutral configuration options for content generation."""
    timeout: float = 60.0
    max_retries: int = 2
    enable_fallback: bool = True
    allow_cross_provider_fallback: bool = False
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    system_instruction: Optional[str] = None
    extra_params: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionMetadata:
    """Provider-neutral execution metadata for audits and monitoring. Never contains secrets."""
    provider: str
    provider_type: str = "direct"
    configured_model: str = ""
    actual_provider_used: str = ""
    actual_model_used: str = ""
    fallback_used: bool = False
    fallback_attempts: int = 0
    fallback_chain: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "SUCCESS"
    error_type: Optional[str] = None
    attempts: int = 1
    duration_seconds: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "provider_type": self.provider_type,
            "configured_model": self.configured_model,
            "actual_provider_used": self.actual_provider_used or self.provider,
            "actual_model_used": self.actual_model_used,
            "fallback_used": self.fallback_used,
            "fallback_attempts": self.fallback_attempts,
            "fallback_chain": self.fallback_chain,
            "status": self.status,
            "error_type": self.error_type,
            "attempts": self.attempts,
            "duration_seconds": self.duration_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens
        }


# ==============================================================================
# TOKEN BUDGET & JSON HELPERS
# ==============================================================================

def get_research_max_output_tokens(count: int) -> int:
    """
    Deterministic token budget mapping for Research batch generation:
    - <= 5 items:  800 tokens
    - <= 10 items: 1500 tokens
    - <= 20 items: 2800 tokens
    - <= 30 items: 4000 tokens
    - > 30 items:  6000 tokens
    """
    if count <= 5:
        return 800
    elif count <= 10:
        return 1500
    elif count <= 20:
        return 2800
    elif count <= 30:
        return 4000
    else:
        return 6000


def clean_json_text(text: str) -> str:
    """Extract clean JSON substring from an LLM response containing markdown code blocks or conversational text."""
    if not text:
        return ""
    text = text.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        if m:
            text = m.group(1).strip()
        else:
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text).strip()

    if not (text.startswith("{") or text.startswith("[")):
        start_obj = text.find("{")
        start_arr = text.find("[")
        if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
            end_obj = text.rfind("}")
            if end_obj > start_obj:
                text = text[start_obj:end_obj + 1]
        elif start_arr != -1:
            end_arr = text.rfind("]")
            if end_arr > start_arr:
                text = text[start_arr:end_arr + 1]

    return text.strip()


# ==============================================================================
# AI PROVIDER ABSTRACT CONTRACT
# ==============================================================================

class AIProvider(ABC):
    """
    Abstract Base Class defining the contract for all AI Providers.
    Concrete providers implement this interface (Gemini, OpenAI, Anthropic, Groq, OpenRouter).
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique provider identifier (e.g., 'gemini', 'openai', 'anthropic', 'groq', 'openrouter')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable provider name (e.g., 'Google Gemini', 'OpenAI', etc.)."""
        pass

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Provider role classification: 'direct' or 'gateway'."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        db: Optional[Session] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        enable_fallback: bool = True,
        options: Optional[AIGenerationOptions] = None,
        **kwargs
    ) -> str:
        """Generate text response from the provider."""
        pass

    @abstractmethod
    def test_connection(self, db: Optional[Session] = None) -> Dict[str, Any]:
        """
        Test connection/credentials with the provider.
        Returns normalized dictionary:
        {
            "provider": str,
            "provider_type": str,
            "configured": bool,
            "connected": bool,
            "configured_model": str,
            "message": str,
            "error_type": Optional[str]
        }
        """
        pass

    @abstractmethod
    def list_models(self, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        """
        List available models supported/accessible for this provider.
        Returns normalized list of model dicts:
        [
            {
                "id": str,
                "name": str,
                "provider": str,
                "provider_type": str,
                "available": bool
            }
        ]
        """
        pass

    @abstractmethod
    def supports_model(self, model_name: str) -> bool:
        """Check whether the model identifier belongs to or is supported by this provider."""
        pass

    @abstractmethod
    def get_last_execution_metadata(self) -> Dict[str, Any]:
        """Retrieve execution metadata from the most recent operation. Guaranteed secret-free."""
        pass

    # ==========================================================================
    # DOMAIN HELPERS (Default Provider-Neutral Implementations)
    # ==========================================================================

    def generate_products(
        self,
        niche: str,
        count: int = 10,
        db: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch Product Research: Generate N products in exactly ONE AI request.
        Strict Low-Consumption Mode: max_retries=0, enable_fallback=False.
        Validates output locally in Python (zero AI calls).
        """
        prompt = f"""You are an expert e-commerce and viral short-video researcher for Douyin (TikTok China).
Target Niche: {niche}
Generate a JSON list of exactly {count} trending, problem-solving, or viral products for this niche.

CRITICAL REQUIREMENTS:
1. "name_vietnamese": Clear, commercial Vietnamese product name.
2. "name_chinese": Natural commercial Chinese product name used by Chinese suppliers.
3. "douyin_keywords": Natural Chinese search phrases specifically used on Douyin to discover real product showcase videos. Do NOT make literal word-by-word translations. Use authentic Douyin short-video terms (e.g., '居家好物', '神器', '开箱', '测评', '好物推荐', specific feature descriptors).
4. "content_angle": Short compelling angle for short video (in Vietnamese).
5. "hook": Short 1-sentence opening hook to grab attention in first 3 seconds (in Vietnamese).

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects. No intro text, no conversational text, no markdown other than standard JSON.
Example structure:
[
  {{
    "name_vietnamese": "Nồi cơm điện mini đa năng",
    "name_chinese": "多功能迷你电饭煲",
    "douyin_keywords": "宿舍迷你电饭煲 独居一人食好物 煮饭神器",
    "content_angle": "Giải pháp nấu ăn tiện lợi nhanh gọn cho người sống một mình",
    "hook": "Đừng mua nồi cơm to nữa nếu bạn sống một mình hoặc ở trọ!"
  }}
]
"""
        output_budget = get_research_max_output_tokens(count)
        raw_content = self.generate(
            prompt,
            db=db,
            timeout=60.0,
            max_retries=0,
            enable_fallback=False,
            options=AIGenerationOptions(
                timeout=60.0,
                max_retries=0,
                enable_fallback=False,
                allow_cross_provider_fallback=False,
                max_output_tokens=output_budget
            )
        )

        cleaned = clean_json_text(raw_content)
        try:
            items = json.loads(cleaned)
            if not isinstance(items, list):
                raise ValueError("Response is not a JSON list.")
        except Exception as e:
            raise ValueError(f"Malformed AI response format: {str(e)}")

        valid_items = []
        for it in items:
            if not isinstance(it, dict):
                continue
            name_vi = str(it.get("name_vietnamese", "")).strip()
            if not name_vi:
                continue
            valid_items.append({
                "name_vietnamese": name_vi,
                "name_chinese": str(it.get("name_chinese", "")).strip(),
                "douyin_keywords": str(it.get("douyin_keywords", "")).strip(),
                "content_angle": str(it.get("content_angle", "")).strip(),
                "hook": str(it.get("hook", "")).strip()
            })

        if not valid_items:
            raise ValueError("No valid products could be parsed from AI response.")

        return valid_items

    def generate_timed_script(
        self,
        video_duration: float,
        segments: List[Dict[str, Any]],
        product_info: Optional[Dict[str, Any]] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Generate timed voiceover script matching video timeline.
        Validates output locally in Python.
        """
        if not segments:
            return {"valid": False, "error": "Danh sách phân đoạn timeline trống."}

        prod_name = (product_info or {}).get("name_vietnamese", "Sản phẩm")
        content_angle = (product_info or {}).get("content_angle", "Tính năng nổi bật")
        hook = (product_info or {}).get("hook", "Món đồ tiện ích không thể bỏ lỡ")

        segment_lines = []
        for s in segments:
            sid = s.get("segment_id")
            st = float(s.get("start_time", s.get("start", 0.0)))
            dur = float(s.get("duration", 0.0))
            et = float(s.get("end_time", s.get("end", round(st + dur, 2))))
            if dur <= 0.0:
                dur = round(et - st, 2)
            desc = s.get("description", "")
            desc_part = f" - Diễn biến cảnh: {desc}" if desc else ""
            segment_lines.append(f"- Phân đoạn {sid}: từ {st:.1f}s đến {et:.1f}s (Thời lượng: {dur:.1f}s){desc_part}")

        timeline_str = "\n".join(segment_lines)

        prompt = f"""Bạn là một chuyên gia biên kịch video ngắn triệu view (TikTok, Reels, Shorts).
Nhiệm vụ: Viết lời đọc thuyết minh (Voice-over) bằng TIẾNG VIỆT cho TOÀN BỘ video sau.

THÔNG TIN SẢN PHẨM:
- Tên sản phẩm: {prod_name}
- Góc tiếp cận: {content_angle}
- Câu mở đầu (Hook): {hook}
- Tổng thời lượng video: {video_duration:.1f} giây

TIMELINE CÁC PHÂN ĐOẠN (Toàn bộ kịch bản phải khớp với từng mốc thời gian):
{timeline_str}

QUY TẮC BẮT BUỘC:
1. KHÔNG được viết quá dài. Tốc độ nói tiếng Việt tự nhiên là khoảng 2.5 đến 3.2 từ mỗi giây.
   Mỗi phân đoạn PHẢI có số lượng từ vừa vặn với thời lượng của phân đoạn đó.
2. Lời thoại tự nhiên, liền mạch giữa các phân đoạn, giọng văn bán hàng thu hút.
3. CHỈ TRẢ VỀ DUY NHẤT một đối tượng JSON hợp lệ theo đúng cấu trúc sau:
{{
  "segments": [
    {{
      "segment_id": 1,
      "vietnamese_text": "..."
    }}
  ]
}}
"""
        raw_text = self.generate(
            prompt,
            db=db,
            timeout=60.0,
            enable_fallback=True,
            options=AIGenerationOptions(
                timeout=60.0,
                max_retries=2,
                enable_fallback=True,
                allow_cross_provider_fallback=True
            )
        )
        cleaned = clean_json_text(raw_text)

        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            raise ValueError(f"AI returned malformed JSON: {e}")

        from app.services.gemini_service import validate_timed_script
        val_result = validate_timed_script(segments, parsed)
        if not val_result.get("valid"):
            raise ValueError(val_result.get("error", "Invalid script structure"))

        return val_result

    @staticmethod
    def build_rewrite_prompt(
        failed_segments: List[Dict[str, Any]],
        round_num: int = 1,
        max_rounds: int = 2,
        product_name: str = ""
    ) -> str:
        """Construct the prompt for batch rewriting failed segments."""
        segment_lines = []
        for s in failed_segments:
            sid = s.get("segment_id")
            target_dur = s.get("target_duration", 0.0)
            actual_dur = s.get("actual_duration", 0.0)
            curr_text = s.get("current_text", s.get("vietnamese_text", ""))
            direction = s.get("direction", "shorten" if actual_dur > target_dur else "lengthen")

            action = "RÚT NGẮN lại lời đọc" if direction == "shorten" else "KÉO DÀI thêm lời đọc"
            segment_lines.append(
                f"- Phân đoạn {sid}: Thời lượng mục tiêu: {target_dur:.1f}s | "
                f"Thời lượng audio thực tế: {actual_dur:.1f}s. "
                f"Yêu cầu: {action}.\n  Lời hiện tại: \"{curr_text}\""
            )

        failed_str = "\n".join(segment_lines)
        prod_context = f"\nSản phẩm: {product_name}" if product_name else ""

        return f"""Sau khi đo đạc âm thanh thực tế, các phân đoạn sau đây KHÔNG khớp với thời lượng video.{prod_context}
Hãy viết lại LỜI ĐỌC TIẾNG VIỆT cho TẤT CẢ các phân đoạn bị lệch dưới đây trong MỘT LẦN DUY NHẤT.
Vòng viết lại: {round_num}/{max_rounds}

DANH SÁCH PHÂN ĐOẠN CẦN CHỈNH SỬA:
{failed_str}

QUY TẮC:
1. Nếu cần rút ngắn: Giảm bớt số từ, dùng từ cô đọng, giữ trọn ý chính.
2. Nếu cần kéo dài: Thêm mô tả nhẹ nhàng, tự nhiên.
3. CHỈ TRẢ VỀ DUY NHẤT MỘT ĐỐI TƯỢNG JSON với cấu trúc:
{{
  "rewritten_segments": [
    {{
      "segment_id": 1,
      "vietnamese_text": "..."
    }}
  ]
}}
"""

    def rewrite_failed_segments(
        self,
        failed_segments: List[Dict[str, Any]],
        round_num: int = 1,
        max_rounds: int = 2,
        db: Optional[Session] = None,
        product_name: str = ""
    ) -> Dict[str, Any]:
        """
        Duration Rewrite: Sends ALL failed segments in ONE single AI request.
        Enforces maximum rewrite rounds (max 2 rounds, no infinite retries).
        Validates rewritten segments locally with Python.
        """
        if round_num > max_rounds:
            return {
                "success": False,
                "error": f"Đã vượt quá số lần viết lại tối đa ({max_rounds} vòng). Cần can thiệp thủ công.",
                "max_rounds_exceeded": True,
                "round": round_num
            }

        if not failed_segments:
            return {
                "success": True,
                "round": round_num,
                "rewritten_segments": []
            }

        prompt = self.build_rewrite_prompt(
            failed_segments,
            round_num=round_num,
            max_rounds=max_rounds,
            product_name=product_name
        )
        raw_text = self.generate(
            prompt,
            db=db,
            timeout=60.0,
            enable_fallback=True,
            options=AIGenerationOptions(
                timeout=60.0,
                max_retries=1,
                enable_fallback=True,
                allow_cross_provider_fallback=True
            )
        )
        cleaned = clean_json_text(raw_text)

        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            raise ValueError(f"AI returned malformed JSON: {e}")

        from app.services.gemini_service import validate_rewritten_segments
        val_result = validate_rewritten_segments(failed_segments, parsed)
        if not val_result.get("valid"):
            raise ValueError(val_result.get("error", "Invalid rewritten segments structure"))

        return {
            "success": True,
            "round": round_num,
            "rewritten_segments": val_result.get("rewritten_segments", [])
        }

