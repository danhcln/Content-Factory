import os
import logging
from pathlib import Path
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path=None, override=False):
        p = Path(dotenv_path) if dotenv_path else Path(".env")
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if override or k not in os.environ:
                        os.environ[k] = v

logger = logging.getLogger("app.config")

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

# Production default model: gemini-3.8-flash
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


def get_gemini_model(db=None) -> str:
    """
    Single source of truth for active Gemini Model.
    Priority:
    1. SQLite settings table ('gemini_model' or 'GEMINI_MODEL') if present and non-empty.
    2. Environment variable GEMINI_MODEL in .env.
    3. Production default: DEFAULT_GEMINI_MODEL ('gemini-3.8-flash').
    """
    # 1. Check SQLite settings table
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["gemini_model", "GEMINI_MODEL"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read model from db settings: {e}")

    # 2. Check environment variable
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_model = os.getenv("GEMINI_MODEL", "").strip()
    if env_model:
        return env_model

    # 3. Production default
    return DEFAULT_GEMINI_MODEL


def get_gemini_api_key(db=None) -> str:
    """
    Single source of truth for Gemini API Key.
    Priority:
    1. SQLite settings table ('gemini_api_key' or 'GEMINI_API_KEY') if present.
    2. Environment variable GEMINI_API_KEY in .env.
    """
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["gemini_api_key", "GEMINI_API_KEY"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read api_key from db settings: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return os.getenv("GEMINI_API_KEY", "").strip()
