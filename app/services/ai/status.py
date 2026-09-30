"""
Single Provider-Aware Status Helper for Multi-AI Architecture

Provides a unified, non-sensitive, local-only snapshot of the active AI provider
for Dashboard, Settings, Header Badge, and Research UI.
Guarantees ZERO external network calls and ZERO token consumption during status queries.
"""
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.config import (
    get_active_ai_provider,
    get_active_ai_model,
    get_key_hint,
    get_gemini_api_key,
    get_openai_api_key,
    get_anthropic_api_key,
    get_groq_api_key,
    get_openrouter_api_key,
    get_mwapi_api_key,
)
from app.services.ai.manager import get_provider_key_configured
from app.services.gemini_status import GeminiStatusTracker
from app.services.usage_tracker import GeminiUsageTracker


PROVIDER_METADATA: Dict[str, Dict[str, Any]] = {
    "gemini": {
        "display_name": "Gemini",
        "short_name": "Gemini",
        "is_gateway": False,
        "key_getter": get_gemini_api_key,
    },
    "openai": {
        "display_name": "OpenAI",
        "short_name": "OpenAI",
        "is_gateway": False,
        "key_getter": get_openai_api_key,
    },
    "anthropic": {
        "display_name": "Anthropic Claude",
        "short_name": "Anthropic",
        "is_gateway": False,
        "key_getter": get_anthropic_api_key,
    },
    "groq": {
        "display_name": "Groq",
        "short_name": "Groq",
        "is_gateway": False,
        "key_getter": get_groq_api_key,
    },
    "openrouter": {
        "display_name": "OpenRouter",
        "short_name": "OpenRouter",
        "is_gateway": True,
        "key_getter": get_openrouter_api_key,
    },
    "mwapi": {
        "display_name": "MWAPI Gateway",
        "short_name": "MWAPI",
        "is_gateway": True,
        "key_getter": get_mwapi_api_key,
    },
}


def get_active_provider_status(db: Optional[Session] = None) -> Dict[str, Any]:
    """
    Construct a single provider-aware status object for the active AI provider.
    Local SQLite and environment lookups only. ZERO network requests.
    """
    pid = get_active_ai_provider(db)
    if pid not in PROVIDER_METADATA:
        pid = "gemini"

    meta = PROVIDER_METADATA.get(pid, PROVIDER_METADATA["gemini"])
    display_name = meta["display_name"]
    short_name = meta["short_name"]
    is_gateway = meta["is_gateway"]
    key_getter = meta["key_getter"]

    raw_key = key_getter(db)
    configured = bool(raw_key)
    key_hint = get_key_hint(raw_key)
    model = get_active_ai_model(db)

    # Local Gemini status for fallback/Gemini-specific diagnostic tracking
    gemini_status = GeminiStatusTracker.get_status(db=db)
    gemini_usage = GeminiUsageTracker.get_usage(db=db)

    if not configured:
        status_code = "AUTH_ERROR"
        status_msg = f"Chưa cấu hình API Key cho {display_name}"
        status_hint = f"Vui lòng nhập API Key cho {display_name} trong Cài đặt."
        badge_label = "Chưa có Key"
        badge_color = "#f59e0b"  # amber
    else:
        if pid == "gemini":
            status_code = gemini_status.get("status", "READY")
            if status_code == "QUOTA_EXCEEDED":
                status_msg = "Đã hết hạn mức Gemini hôm nay"
                status_hint = gemini_status.get("hint", "Chờ hạn mức được làm mới hoặc dùng API Key khác.")
                badge_label = "Hết hạn mức"
                badge_color = "#f97316"  # orange
            elif status_code == "BUSY":
                status_msg = "Gemini đang quá tải"
                status_hint = gemini_status.get("hint", "Google đang có lượng truy cập cao.")
                badge_label = "Quá tải"
                badge_color = "#f59e0b"
            elif status_code == "RATE_LIMITED":
                status_msg = "Gemini đang giới hạn tạm thời"
                status_hint = gemini_status.get("hint", "Đợi vài giây rồi thử lại.")
                badge_label = "Giới hạn"
                badge_color = "#f59e0b"
            elif status_code == "AUTH_ERROR":
                status_msg = "Lỗi Gemini API Key"
                status_hint = "Kiểm tra lại API Key trong Cài đặt."
                badge_label = "Lỗi Key"
                badge_color = "#ef4444"
            else:
                status_code = "READY"
                status_msg = f"{display_name} đã cấu hình và sẵn sàng"
                status_hint = f"Mô hình: {model}. Mỗi lượt chạy gửi đúng 1 yêu cầu duy nhất tới {display_name}."
                badge_label = "Sẵn sàng"
                badge_color = "#22c55e"  # green
        else:
            # For non-Gemini providers, status is READY if configured
            status_code = "READY"
            status_msg = f"{display_name} đã cấu hình và sẵn sàng"
            status_hint = f"Mô hình: {model}. Mỗi lượt chạy gửi đúng 1 yêu cầu duy nhất tới {display_name}."
            badge_label = "Sẵn sàng"
            badge_color = "#22c55e"

    diagnostic_action = f"KIỂM TRA {display_name.upper()}"

    return {
        "provider_id": pid,
        "display_name": display_name,
        "short_name": short_name,
        "model": model,
        "configured": configured,
        "key_hint": key_hint,
        "status": status_code,
        "status_msg": status_msg,
        "status_hint": status_hint,
        "badge_label": badge_label,
        "badge_color": badge_color,
        "diagnostic_action": diagnostic_action,
        "is_gateway": is_gateway,
        "gemini_status": gemini_status,
        "gemini_usage": gemini_usage,
    }
