"""
Gemini API Status Tracker — stores lightweight status in SQLite settings table.
Does NOT call Gemini. Only records state from actual Gemini requests.
"""
import datetime
import logging
from typing import Dict, Any, Optional
try:
    from sqlalchemy.orm import Session
    from app.database import SessionLocal
    from app.models import Setting
except ImportError:
    Session = Any
    SessionLocal = None
    Setting = None

logger = logging.getLogger("app.services.gemini_status")

# Keys stored in 'settings' table (no API keys, no request content)
STATUS_KEY = "gemini_last_status"           # READY/BUSY/QUOTA_EXCEEDED/AUTH_ERROR/etc.
STATUS_AT_KEY = "gemini_last_status_at"     # ISO timestamp of last status update
HTTP_CODE_KEY = "gemini_last_http_code"     # e.g. "200", "429", "503"
SUCCESS_AT_KEY = "gemini_last_success_at"   # ISO timestamp of last successful request
MESSAGE_KEY = "gemini_last_message"         # short human-friendly Vietnamese message


# --- Status → human-readable Vietnamese message map ---
STATUS_MESSAGES: Dict[str, str] = {
    "READY":          "Gemini sẵn sàng",
    "CHECKING":       "Đang kiểm tra Gemini...",
    "BUSY":           "Gemini đang quá tải — sẽ thử lại",
    "RATE_LIMITED":   "Gemini đang giới hạn tạm thời",
    "QUOTA_EXCEEDED": "Đã hết hạn mức Gemini hôm nay",
    "AUTH_ERROR":     "Lỗi Gemini API Key",
    "NETWORK_ERROR":  "Không kết nối được Gemini",
    "ERROR":          "Gemini đang gặp lỗi",
    "UNKNOWN":        "Chưa kiểm tra Gemini",
}

STATUS_HINT: Dict[str, str] = {
    "READY":          "Lần gọi gần nhất thành công.",
    "CHECKING":       "",
    "BUSY":           "Google đang có lượng truy cập cao. Hệ thống đã thử lại có giới hạn.",
    "RATE_LIMITED":   "Gemini tạm thời giới hạn số yêu cầu mỗi phút. Hãy đợi vài giây rồi thử lại.",
    "QUOTA_EXCEEDED": "Chờ hạn mức được làm mới (thường sau 0:00 UTC) hoặc sử dụng project/tier có hạn mức phù hợp.",
    "AUTH_ERROR":     "Kiểm tra lại API Key trong Cài đặt.",
    "NETWORK_ERROR":  "Kiểm tra kết nối mạng và thử lại.",
    "ERROR":          "Xem chi tiết trong logs/app.log.",
    "UNKNOWN":        "Nhấn KIỂM TRA GEMINI để xác minh.",
}


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class GeminiStatusTracker:
    """
    Stores and retrieves Gemini API status in SQLite settings.
    ZERO Gemini quota usage — all methods are purely local.
    """

    @staticmethod
    def _set(key: str, value: str, db: Session) -> None:
        rec = db.query(Setting).filter(Setting.key == key).first()
        if rec:
            rec.value = value
        else:
            db.add(Setting(key=key, value=value, description=f"Gemini status tracker: {key}"))

    @classmethod
    def update_status(
        cls,
        status: str,
        db: Optional[Session] = None,
        http_code: Optional[int] = None,
        custom_message: Optional[str] = None
    ) -> None:
        """
        Record a new Gemini status. Called by gemini_service after any real request.
        Does NOT make any network calls.
        """
        owns_session = False
        if db is None:
            db = SessionLocal()
            owns_session = True
        try:
            now = _now_iso()
            msg = custom_message or STATUS_MESSAGES.get(status, status)
            cls._set(STATUS_KEY, status, db)
            cls._set(STATUS_AT_KEY, now, db)
            cls._set(MESSAGE_KEY, msg, db)
            if http_code is not None:
                cls._set(HTTP_CODE_KEY, str(http_code), db)
            if status == "READY":
                cls._set(SUCCESS_AT_KEY, now, db)
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to update Gemini status: {e}")
            if db:
                try:
                    db.rollback()
                except Exception:
                    pass
        finally:
            if owns_session:
                db.close()

    @classmethod
    def get_status(cls, db: Optional[Session] = None) -> Dict[str, Any]:
        """
        Return current Gemini status dict. Never calls Gemini API.
        """
        owns_session = False
        if db is None:
            db = SessionLocal()
            owns_session = True
        try:
            def _get(key: str, default: str = "") -> str:
                rec = db.query(Setting).filter(Setting.key == key).first()
                return rec.value if rec and rec.value else default

            status = _get(STATUS_KEY, "UNKNOWN")
            message = _get(MESSAGE_KEY, STATUS_MESSAGES.get(status, status))
            http_code_str = _get(HTTP_CODE_KEY, "")
            http_code = int(http_code_str) if http_code_str.isdigit() else None

            from app.config import get_gemini_model
            model = get_gemini_model()

            return {
                "model": model,
                "status": status,
                "message": message,
                "hint": STATUS_HINT.get(status, ""),
                "last_http_code": http_code,
                "last_status_at": _get(STATUS_AT_KEY) or None,
                "last_success_at": _get(SUCCESS_AT_KEY) or None,
            }
        except Exception as e:
            logger.warning(f"Failed to read Gemini status: {e}")
            return {
                "model": "gemini-3.8-flash",
                "status": "UNKNOWN",
                "message": STATUS_MESSAGES["UNKNOWN"],
                "hint": STATUS_HINT["UNKNOWN"],
                "last_http_code": None,
                "last_status_at": None,
                "last_success_at": None,
            }
        finally:
            if owns_session:
                db.close()
