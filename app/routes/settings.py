import os
import json
from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, Any
from dotenv import load_dotenv, set_key

from app.database import get_db, BASE_DIR, TEMP_DIR
from app.config import get_gemini_model, get_gemini_api_key, DEFAULT_GEMINI_MODEL
from app.models import Setting
from app.services.usage_tracker import GeminiUsageTracker
from app.services.subtitle_service import SubtitleService
from app.services.tts.vieneu_provider import VieNeuProvider

templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
router = APIRouter(tags=["settings"])
ENV_FILE = BASE_DIR / ".env"


class SubtitleSettingsPayload(BaseModel):
    font: Optional[str] = "Arial Bold"
    size: Optional[str] = "medium"
    color: Optional[str] = "#FFFFFF"
    outline_color: Optional[str] = "#000000"
    outline_width: Optional[float] = 2.2
    position: Optional[str] = "bottom"
    custom_y: Optional[float] = 0.84
    animation: Optional[str] = "fade"
    voice_name: Optional[str] = "Trúc Ly"


class PreviewVoicePayload(BaseModel):
    voice_name: Optional[str] = "Trúc Ly"
    text: Optional[str] = "Xin chào, đây là giọng đọc thử nghiệm của hệ thống AI Content Factory."


def get_current_settings(db: Session = None):
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    raw_key = get_gemini_api_key(db)
    masked_key = ""
    if raw_key:
        if len(raw_key) > 8:
            masked_key = raw_key[:4] + "*" * (len(raw_key) - 8) + raw_key[-4:]
        else:
            masked_key = "********"

    usage_stats = GeminiUsageTracker.get_usage(db)
    sub_prefs = SubtitleService.get_user_settings(db)
    vieneu_prov = VieNeuProvider()
    available_voices = vieneu_prov.get_available_voices()
    verified_fonts = SubtitleService.get_verified_fonts()

    from app.services.ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path, check_ffmpeg_available

    return {
        "gemini_api_key_masked": masked_key,
        "gemini_api_key_set": bool(raw_key),
        "gemini_model": get_gemini_model(db),
        "gemini_usage": usage_stats,
        "products_per_research": os.getenv("DEFAULT_PRODUCTS_COUNT", "10"),
        "keywords_per_product": os.getenv("DEFAULT_KEYWORDS_COUNT", "3"),
        "videos_per_product": os.getenv("DEFAULT_VIDEOS_COUNT", "5"),
        "tts_engine": os.getenv("TTS_ENGINE", "VieNeu-TTS"),
        "default_voice": sub_prefs.get("voice_name") or os.getenv("DEFAULT_VOICE", "Trúc Ly"),
        "auto_generate_voice": os.getenv("AUTO_GENERATE_VOICE", "true").lower() == "true",
        "original_folder": os.getenv("ORIGINAL_FOLDER", "downloads/original"),
        "package_folder": os.getenv("PACKAGE_FOLDER", "downloads/packages"),
        "final_folder": os.getenv("FINAL_FOLDER", "downloads/final"),
        # Phase 12 Auto Editor Settings
        "ffmpeg_path": os.getenv("FFMPEG_PATH", "") or get_ffmpeg_path() or "",
        "ffprobe_path": os.getenv("FFPROBE_PATH", "") or get_ffprobe_path() or "",
        "auto_edit_cover_type": os.getenv("AUTO_EDIT_COVER_TYPE", "blur"),
        "auto_edit_blur_strength": os.getenv("AUTO_EDIT_BLUR_STRENGTH", "10"),
        "auto_edit_source_audio": os.getenv("AUTO_EDIT_SOURCE_AUDIO", "low"),
        "auto_edit_subtitles": os.getenv("AUTO_EDIT_SUBTITLES", "true").lower() == "true",
        "auto_edit_hook": os.getenv("AUTO_EDIT_HOOK", "true").lower() == "true",
        "auto_edit_auto_render": os.getenv("AUTO_EDIT_AUTO_RENDER", "false").lower() == "true",
        "ffmpeg_status": check_ffmpeg_available(),
        # Phase 13 Subtitle & Voice Settings
        "subtitles": sub_prefs,
        "available_voices": available_voices,
        "verified_fonts": verified_fonts
    }


@router.get("/settings", response_class=HTMLResponse)
def get_settings_page(request: Request, db: Session = Depends(get_db)):
    settings_data = get_current_settings(db)
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={
            "settings": settings_data,
            "active_page": "settings",
            "message": None
        }
    )


@router.post("/settings", response_class=HTMLResponse)
def save_settings(
    request: Request,
    gemini_api_key: str = Form(""),
    gemini_model: str = Form(DEFAULT_GEMINI_MODEL),
    products_per_research: str = Form("10"),
    keywords_per_product: str = Form("3"),
    videos_per_product: str = Form("5"),
    tts_engine: str = Form("VieNeu-TTS"),
    default_voice: str = Form("Trúc Ly"),
    auto_generate_voice: str = Form("off"),
    original_folder: str = Form("downloads/original"),
    package_folder: str = Form("downloads/packages"),
    final_folder: str = Form("downloads/final"),
    ffmpeg_path: str = Form(""),
    ffprobe_path: str = Form(""),
    auto_edit_cover_type: str = Form("blur"),
    auto_edit_blur_strength: str = Form("10"),
    auto_edit_source_audio: str = Form("low"),
    auto_edit_subtitles: str = Form("off"),
    auto_edit_hook: str = Form("off"),
    auto_edit_auto_render: str = Form("off"),
    # Subtitle UX Settings
    subtitle_font: str = Form("Arial Bold"),
    subtitle_size: str = Form("medium"),
    subtitle_color: str = Form("#FFFFFF"),
    subtitle_outline_color: str = Form("#000000"),
    subtitle_outline_width: str = Form("2.2"),
    subtitle_position: str = Form("bottom"),
    subtitle_custom_y: str = Form("0.84"),
    subtitle_animation: str = Form("fade"),
    db: Session = Depends(get_db)
):
    if not ENV_FILE.exists():
        ENV_FILE.touch()

    # Only update API key if a non-masked new value is provided
    if gemini_api_key and not gemini_api_key.startswith("****") and "*" not in gemini_api_key:
        set_key(str(ENV_FILE), "GEMINI_API_KEY", gemini_api_key.strip())
        db_key = db.query(Setting).filter(Setting.key == "gemini_api_key").first()
        if not db_key:
            db_key = Setting(key="gemini_api_key", value=gemini_api_key.strip(), description="Gemini API Key")
            db.add(db_key)
        else:
            db_key.value = gemini_api_key.strip()

    model_clean = gemini_model.strip() or DEFAULT_GEMINI_MODEL
    set_key(str(ENV_FILE), "GEMINI_MODEL", model_clean)

    db_model = db.query(Setting).filter(Setting.key.in_(["gemini_model", "GEMINI_MODEL"])).first()
    if not db_model:
        db_model = Setting(key="gemini_model", value=model_clean, description="Gemini AI Model")
        db.add(db_model)
    else:
        db_model.value = model_clean
    db.commit()

    set_key(str(ENV_FILE), "DEFAULT_PRODUCTS_COUNT", products_per_research.strip())
    set_key(str(ENV_FILE), "DEFAULT_KEYWORDS_COUNT", keywords_per_product.strip())
    set_key(str(ENV_FILE), "DEFAULT_VIDEOS_COUNT", videos_per_product.strip())
    set_key(str(ENV_FILE), "TTS_ENGINE", tts_engine.strip())
    set_key(str(ENV_FILE), "DEFAULT_VOICE", default_voice.strip())
    set_key(str(ENV_FILE), "AUTO_GENERATE_VOICE", "true" if auto_generate_voice == "on" else "false")
    set_key(str(ENV_FILE), "ORIGINAL_FOLDER", original_folder.strip())
    set_key(str(ENV_FILE), "PACKAGE_FOLDER", package_folder.strip())
    set_key(str(ENV_FILE), "FINAL_FOLDER", final_folder.strip())

    # Phase 12 Settings
    if ffmpeg_path.strip():
        set_key(str(ENV_FILE), "FFMPEG_PATH", ffmpeg_path.strip())
    if ffprobe_path.strip():
        set_key(str(ENV_FILE), "FFPROBE_PATH", ffprobe_path.strip())

    set_key(str(ENV_FILE), "AUTO_EDIT_COVER_TYPE", auto_edit_cover_type.strip())
    set_key(str(ENV_FILE), "AUTO_EDIT_BLUR_STRENGTH", auto_edit_blur_strength.strip())
    set_key(str(ENV_FILE), "AUTO_EDIT_SOURCE_AUDIO", auto_edit_source_audio.strip())
    set_key(str(ENV_FILE), "AUTO_EDIT_SUBTITLES", "true" if auto_edit_subtitles == "on" else "false")
    set_key(str(ENV_FILE), "AUTO_EDIT_HOOK", "true" if auto_edit_hook == "on" else "false")
    set_key(str(ENV_FILE), "AUTO_EDIT_AUTO_RENDER", "true" if auto_edit_auto_render == "on" else "false")

    # Phase 13 Subtitle Settings
    try:
        sub_width_f = float(subtitle_outline_width.strip())
    except ValueError:
        sub_width_f = 2.2
    try:
        sub_pos_y_f = float(subtitle_custom_y.strip())
    except ValueError:
        sub_pos_y_f = 0.84

    SubtitleService.save_user_settings(db, {
        "font": subtitle_font.strip(),
        "size": subtitle_size.strip(),
        "color": subtitle_color.strip(),
        "outline_color": subtitle_outline_color.strip(),
        "outline_width": sub_width_f,
        "position": subtitle_position.strip(),
        "custom_y": sub_pos_y_f,
        "animation": subtitle_animation.strip(),
        "voice_name": default_voice.strip()
    })

    settings_data = get_current_settings(db)
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={
            "settings": settings_data,
            "active_page": "settings",
            "message": "Cài đặt hệ thống & phụ đề đã được lưu thành công!"
        }
    )


@router.get("/api/settings/subtitles")
def api_get_subtitles(db: Session = Depends(get_db)):
    """Retrieve saved subtitle configuration, verified fonts, and available VieNeu voices."""
    sub_prefs = SubtitleService.get_user_settings(db)
    fonts = SubtitleService.get_verified_fonts()
    prov = VieNeuProvider()
    voices = prov.get_available_voices()
    return JSONResponse(content={
        "success": True,
        "settings": sub_prefs,
        "fonts": fonts,
        "voices": voices
    })


@router.post("/api/settings/subtitles")
def api_save_subtitles(payload: SubtitleSettingsPayload, db: Session = Depends(get_db)):
    """Persist subtitle and voice settings into DB."""
    ok = SubtitleService.save_user_settings(db, payload.dict())
    if ok:
        return JSONResponse(content={"success": True, "message": "Cài đặt phụ đề đã được lưu thành công."})
    return JSONResponse(content={"success": False, "error": "Không thể lưu cài đặt phụ đề."}, status_code=500)


@router.get("/api/settings/voices")
def api_get_voices():
    """Retrieve all available VieNeu voices dynamically."""
    prov = VieNeuProvider()
    voices = prov.get_available_voices()
    return JSONResponse(content={"success": True, "voices": voices})


@router.post("/api/settings/preview-voice")
def api_preview_voice(payload: PreviewVoicePayload):
    """Synthesize a short sample sentence to preview the selected voice."""
    prov = VieNeuProvider()
    if not prov.is_available():
        return JSONResponse(content={"success": False, "error": "VieNeu-TTS engine chưa khả dụng."}, status_code=400)

    target_dir = TEMP_DIR / "voice_previews"
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / f"preview_{payload.voice_name.replace(' ', '_')}.wav"

    text = (payload.text or "Xin chào, đây là giọng đọc thử nghiệm.").strip()
    res = prov.synthesize(text=text, output_path=str(out_file), voice_name=payload.voice_name)

    if res.get("success"):
        import time
        audio_url = f"/temp/voice_previews/{out_file.name}?t={int(time.time())}"
        return JSONResponse(content={
            "success": True,
            "voice_name": payload.voice_name,
            "audio_url": audio_url,
            "duration": res.get("duration", 0.0)
        })
    else:
        return JSONResponse(content={"success": False, "error": res.get("error", "Lỗi tạo giọng nói nghe thử")}, status_code=500)


@router.post("/api/settings/test-gemini")
def api_test_gemini():
    from app.services.gemini_service import GeminiService
    service = GeminiService()
    return service.test_connection()


@router.post("/api/settings/test-voice")
def api_test_voice():
    from app.services.tts import TTSService
    tts = TTSService()
    return tts.test_voice()


@router.post("/api/settings/test-watcher")
def api_test_watcher():
    from app.database import FINAL_DIR
    return {
        "success": True,
        "watch_dir": str(FINAL_DIR),
        "exists": FINAL_DIR.exists(),
        "status": "WATCHING"
    }


@router.post("/api/settings/test-ffmpeg")
def api_test_ffmpeg():
    from app.services.ffmpeg_utils import check_ffmpeg_available
    return check_ffmpeg_available()


@router.get("/api/gemini/status")
def api_gemini_status(db: Session = Depends(get_db)):
    from app.services.gemini_status import GeminiStatusTracker
    from app.services.usage_tracker import GeminiUsageTracker
    status_data = GeminiStatusTracker.get_status(db=db)
    usage_data = GeminiUsageTracker.get_usage(db=db)
    return {
        **status_data,
        "requests_today": usage_data["requests_today"],
        "successful_requests": usage_data["successful_requests"],
        "quota_errors": usage_data["quota_errors"],
    }
