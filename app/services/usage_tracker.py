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

logger = logging.getLogger("app.services.usage_tracker")

USAGE_DATE_KEY = "gemini_usage_date"
REQUESTS_TODAY_KEY = "gemini_requests_today"
SUCCESSFUL_REQUESTS_KEY = "gemini_successful_requests"
FAILED_REQUESTS_KEY = "gemini_failed_requests"
QUOTA_ERRORS_KEY = "gemini_quota_errors"

DISCLAIMER_TEXT = "Lưu ý: Số liệu thống kê cục bộ mang tính tham khảo, không đại diện cho bộ đếm hạn ngạch chính thức của Google."


class GeminiUsageTracker:
    """
    Lightweight, thread-safe local usage tracker for Gemini API requests.
    Persisted in SQLite 'settings' table.
    Resets 'requests_today' when the local date rolls over.
    """

    @staticmethod
    def _get_today_str() -> str:
        return datetime.date.today().isoformat()

    @classmethod
    def _ensure_daily_reset(cls, db: Session) -> None:
        today_str = cls._get_today_str()
        date_rec = db.query(Setting).filter(Setting.key == USAGE_DATE_KEY).first()

        if not date_rec:
            # Initialize for the first time
            db.add(Setting(key=USAGE_DATE_KEY, value=today_str, description="Gemini usage tracking date"))
            db.add(Setting(key=REQUESTS_TODAY_KEY, value="0", description="Gemini total requests today"))
            db.add(Setting(key=SUCCESSFUL_REQUESTS_KEY, value="0", description="Gemini successful requests today"))
            db.add(Setting(key=FAILED_REQUESTS_KEY, value="0", description="Gemini failed requests today"))
            db.add(Setting(key=QUOTA_ERRORS_KEY, value="0", description="Gemini quota errors today"))
            db.commit()
            return

        if date_rec.value != today_str:
            # Date has rolled over -> Reset daily counters
            date_rec.value = today_str
            for key_name in [REQUESTS_TODAY_KEY, SUCCESSFUL_REQUESTS_KEY, FAILED_REQUESTS_KEY, QUOTA_ERRORS_KEY]:
                rec = db.query(Setting).filter(Setting.key == key_name).first()
                if rec:
                    rec.value = "0"
                else:
                    db.add(Setting(key=key_name, value="0", description="Gemini counter"))
            db.commit()

    @classmethod
    def get_usage(cls, db: Optional[Session] = None) -> Dict[str, Any]:
        """Fetch current daily stats."""
        owns_session = False
        if db is None:
            db = SessionLocal()
            owns_session = True

        try:
            cls._ensure_daily_reset(db)
            today_str = cls._get_today_str()

            def get_val(key_name: str) -> int:
                rec = db.query(Setting).filter(Setting.key == key_name).first()
                if rec and rec.value:
                    try:
                        return int(rec.value)
                    except ValueError:
                        return 0
                return 0

            return {
                "date": today_str,
                "requests_today": get_val(REQUESTS_TODAY_KEY),
                "successful_requests": get_val(SUCCESSFUL_REQUESTS_KEY),
                "failed_requests": get_val(FAILED_REQUESTS_KEY),
                "quota_errors": get_val(QUOTA_ERRORS_KEY),
                "disclaimer": DISCLAIMER_TEXT
            }
        finally:
            if owns_session:
                db.close()

    @classmethod
    def _increment(cls, key_name: str, db: Optional[Session] = None) -> None:
        owns_session = False
        if db is None:
            db = SessionLocal()
            owns_session = True

        try:
            cls._ensure_daily_reset(db)
            rec = db.query(Setting).filter(Setting.key == key_name).first()
            if not rec:
                rec = Setting(key=key_name, value="1", description="Gemini counter")
                db.add(rec)
            else:
                try:
                    val = int(rec.value or "0")
                except ValueError:
                    val = 0
                rec.value = str(val + 1)
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to increment {key_name}: {e}")
            if db:
                db.rollback()
        finally:
            if owns_session:
                db.close()

    @classmethod
    def record_request_start(cls, db: Optional[Session] = None) -> None:
        """Call immediately before attempting a Gemini API request."""
        cls._increment(REQUESTS_TODAY_KEY, db=db)

    @classmethod
    def record_success(cls, db: Optional[Session] = None) -> None:
        """Call when a Gemini request completes with HTTP 200 and valid output."""
        cls._increment(SUCCESSFUL_REQUESTS_KEY, db=db)

    @classmethod
    def record_failure(cls, db: Optional[Session] = None, is_quota: bool = False) -> None:
        """Call when a Gemini request fails. If is_quota is True, also increments quota_errors."""
        cls._increment(FAILED_REQUESTS_KEY, db=db)
        if is_quota:
            cls._increment(QUOTA_ERRORS_KEY, db=db)
