import os
import logging
from pathlib import Path
from typing import Optional
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

DEFAULT_ACTIVE_AI_PROVIDER = "gemini"

# Production default model: gemini-3.8-flash
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"

SUPPORTED_AI_PROVIDERS = ["gemini", "openai", "anthropic", "groq", "openrouter"]


def mask_api_key(key: Optional[str] = None) -> str:
    """Mask key securely for safe logging/display. Returns empty string if key is empty."""
    if not key:
        return ""
    k = str(key).strip()
    if len(k) > 8:
        return k[:4] + "*" * (len(k) - 8) + k[-4:]
    return "********"


def get_key_hint(key: Optional[str] = None) -> str:
    """
    Return a non-sensitive key hint (last 4 characters only).
    Never exposes sufficient characters to compromise security.
    """
    if not key:
        return ""
    k = str(key).strip()
    if len(k) >= 8:
        return f"••••{k[-4:]}"
    return "••••••••"


def get_active_ai_provider(db=None) -> str:
    """
    Single source of truth for active AI provider.
    Priority:
    1. SQLite settings table ('active_ai_provider' or 'ACTIVE_AI_PROVIDER') if present and valid.
    2. Environment variable ACTIVE_AI_PROVIDER in .env.
    3. Production default: DEFAULT_ACTIVE_AI_PROVIDER ('gemini').
    """
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["active_ai_provider", "ACTIVE_AI_PROVIDER"])).first()
            if rec and rec.value and rec.value.strip():
                val = rec.value.strip().lower()
                if val in SUPPORTED_AI_PROVIDERS:
                    return val
        except Exception as e:
            logger.debug(f"Could not read active provider from db settings: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_provider = os.getenv("ACTIVE_AI_PROVIDER", "").strip().lower()
    if env_provider in SUPPORTED_AI_PROVIDERS:
        return env_provider

    return DEFAULT_ACTIVE_AI_PROVIDER


def get_gemini_model(db=None) -> str:
    """
    Single source of truth for active Gemini Model.
    Priority:
    1. SQLite settings table ('gemini_model' or 'GEMINI_MODEL') if present and non-empty.
    2. Environment variable GEMINI_MODEL in .env.
    3. Production default: DEFAULT_GEMINI_MODEL ('gemini-3.8-flash').
    """
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["gemini_model", "GEMINI_MODEL"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read model from db settings: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_model = os.getenv("GEMINI_MODEL", "").strip()
    if env_model:
        return env_model

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


# ==============================================================================
# OPENAI BYOK CONFIGURATION
# ==============================================================================

def get_openai_model(db=None) -> str:
    """Single source of truth for OpenAI Model."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["openai_model", "OPENAI_MODEL"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read openai model from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_model = os.getenv("OPENAI_MODEL", "").strip()
    return env_model or DEFAULT_OPENAI_MODEL


def get_openai_api_key(db=None) -> str:
    """Single source of truth for OpenAI API Key."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["openai_api_key", "OPENAI_API_KEY"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read openai api_key from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return os.getenv("OPENAI_API_KEY", "").strip()


# ==============================================================================
# ANTHROPIC BYOK CONFIGURATION
# ==============================================================================

def get_anthropic_model(db=None) -> str:
    """Single source of truth for Anthropic Model."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["anthropic_model", "ANTHROPIC_MODEL"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read anthropic model from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_model = os.getenv("ANTHROPIC_MODEL", "").strip()
    return env_model or DEFAULT_ANTHROPIC_MODEL


def get_anthropic_api_key(db=None) -> str:
    """Single source of truth for Anthropic API Key."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["anthropic_api_key", "ANTHROPIC_API_KEY"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read anthropic api_key from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return os.getenv("ANTHROPIC_API_KEY", "").strip()


# ==============================================================================
# GROQ BYOK CONFIGURATION
# ==============================================================================

def get_groq_model(db=None) -> str:
    """Single source of truth for Groq Model."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["groq_model", "GROQ_MODEL"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read groq model from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_model = os.getenv("GROQ_MODEL", "").strip()
    return env_model or DEFAULT_GROQ_MODEL


def get_groq_api_key(db=None) -> str:
    """Single source of truth for Groq API Key."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["groq_api_key", "GROQ_API_KEY"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read groq api_key from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return os.getenv("GROQ_API_KEY", "").strip()


# ==============================================================================
# OPENROUTER GATEWAY CONFIGURATION
# ==============================================================================

def get_openrouter_model(db=None) -> str:
    """Single source of truth for OpenRouter Gateway Model."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["openrouter_model", "OPENROUTER_MODEL"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read openrouter model from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    env_model = os.getenv("OPENROUTER_MODEL", "").strip()
    return env_model or DEFAULT_OPENROUTER_MODEL


def get_openrouter_api_key(db=None) -> str:
    """Single source of truth for OpenRouter API Key."""
    if db is not None:
        try:
            from app.models import Setting
            rec = db.query(Setting).filter(Setting.key.in_(["openrouter_api_key", "OPENROUTER_API_KEY"])).first()
            if rec and rec.value and rec.value.strip():
                return rec.value.strip()
        except Exception as e:
            logger.debug(f"Could not read openrouter api_key from db: {e}")

    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return os.getenv("OPENROUTER_API_KEY", "").strip()

