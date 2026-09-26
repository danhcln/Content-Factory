import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db, BASE_DIR, TEMP_DIR
from app.models import Video, Product
from app.services.auto_editor import AutoEditorService
from app.services.batch_editor import BatchAutoEditorService
from app.services.ffmpeg_utils import check_ffmpeg_available
from app.services.subtitle_service import SubtitleService
from app.services.tts.vieneu_provider import VieNeuProvider

logger = logging.getLogger("app.routes.auto_edit")

templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

router = APIRouter(tags=["auto_edit"])
auto_editor = AutoEditorService()
batch_editor = BatchAutoEditorService()


class BatchStartPayload(BaseModel):
    video_ids: List[str]
    cover_type: Optional[str] = "blur"
    blur_strength: Optional[int] = 10
    voice_name: Optional[str] = "Trúc Ly"


class SetRegionPayload(BaseModel):
    mode: str = "manual"
    preset: Optional[str] = "BOTTOM"
    x: Optional[float] = 0.05
    y: Optional[float] = 0.72
    width: Optional[float] = 0.90
    height: Optional[float] = 0.20


class RenderPayload(BaseModel):
    cover_type: str = "blur"
    blur_strength: int = 10
    source_audio: str = "low"
    include_subtitles: bool = True
    include_hook: bool = True


class VoicePayload(BaseModel):
    voice_name: Optional[str] = "Trúc Ly"


class SubtitlePayload(BaseModel):
    font_name: Optional[str] = None
    font_size: Optional[Any] = None
    primary_color: Optional[str] = None
    outline_color: Optional[str] = None
    outline_width: Optional[float] = None
    position: Optional[str] = None
    custom_pos_y: Optional[float] = None
    animation: Optional[str] = None
    sample_text: Optional[str] = None


@router.get("/auto-edit", response_class=HTMLResponse)
def get_auto_edit_queue(request: Request, db: Session = Depends(get_db)):
    """Queue of videos ready for Auto Video Editing."""
    videos = db.query(Video).filter(
        Video.status.in_(["DOWNLOADED", "SCRIPTING", "SCRIPT_READY", "AUDIO_GENERATING", "READY", "AUTO_EDIT_READY", "AUTO_EDIT_ERROR", "SUBTITLE_REGION_REVIEW_REQUIRED"])
    ).order_by(Video.id.desc()).all()

    ff_status = check_ffmpeg_available()
    available_voices = VieNeuProvider().get_available_voices()

    return templates.TemplateResponse(
        request=request,
        name="auto_edit.html",
        context={
            "videos": videos,
            "active_page": "auto_edit",
            "ffmpeg_status": ff_status,
            "available_voices": available_voices
        }
    )


@router.get("/auto-edit/{video_id}", response_class=HTMLResponse)
def get_auto_edit_detail(video_id: str, request: Request, db: Session = Depends(get_db)):
    """Interactive workspace for Auto Video Editing with live subtitle & voice styling controls."""
    video = db.query(Video).filter(Video.video_id == video_id).first()
    if not video:
        return HTMLResponse(content="<h1>Video không tồn tại</h1>", status_code=404)

    plan = auto_editor.load_edit_plan(video_id)
    ff_status = check_ffmpeg_available()

    # Parse subtitle region if saved in DB
    sub_region = plan.get("subtitle_region", {})
    if video.subtitle_region:
        try:
            sub_region = json.loads(video.subtitle_region)
        except Exception:
            pass

    preview_file = TEMP_DIR / video_id / "frame_preview.jpg"
    has_preview = preview_file.exists() and preview_file.stat().st_size > 0

    sub_settings = plan.get("subtitle_settings") or SubtitleService.get_user_settings(db)
    verified_fonts = SubtitleService.get_verified_fonts()
    available_voices = VieNeuProvider().get_available_voices()
    current_voice = plan.get("voice_name") or SubtitleService.get_user_settings(db).get("voice_name", "Trúc Ly")

    return templates.TemplateResponse(
        request=request,
        name="auto_edit_detail.html",
        context={
            "video": video,
            "product": video.product,
            "plan": plan,
            "sub_region": sub_region,
            "has_preview": has_preview,
            "active_page": "auto_edit",
            "ffmpeg_status": ff_status,
            "sub_settings": sub_settings,
            "verified_fonts": verified_fonts,
            "available_voices": available_voices,
            "current_voice": current_voice
        }
    )


@router.post("/api/auto-edit/analyze/{video_id}")
def api_analyze_source(video_id: str, db: Session = Depends(get_db)):
    res = auto_editor.analyze_source(db, video_id)
    status_code = 200 if res.get("success") else 400
    return JSONResponse(content=res, status_code=status_code)


@router.post("/api/auto-edit/detect-sub/{video_id}")
def api_detect_subtitle_region(video_id: str, db: Session = Depends(get_db)):
    res = auto_editor.detect_or_set_subtitle_region(db, video_id, mode="auto")
    return JSONResponse(content=res, status_code=200)


@router.post("/api/auto-edit/set-region/{video_id}")
def api_set_region(video_id: str, payload: SetRegionPayload, db: Session = Depends(get_db)):
    custom = None
    if payload.mode == "custom":
        custom = {
            "x": payload.x or 0.05,
            "y": payload.y or 0.72,
            "width": payload.width or 0.90,
            "height": payload.height or 0.20
        }
    res = auto_editor.detect_or_set_subtitle_region(
        db,
        video_id,
        mode=payload.mode,
        custom_region=custom,
        preset=payload.preset or "BOTTOM"
    )
    status_code = 200 if res.get("success") else 400
    return JSONResponse(content=res, status_code=status_code)


@router.post("/api/auto-edit/generate-script/{video_id}")
def api_generate_script(video_id: str, db: Session = Depends(get_db)):
    res = auto_editor.generate_timed_script(db, video_id)
    status_code = 200 if res.get("success") else 400
    return JSONResponse(content=res, status_code=status_code)


@router.post("/api/auto-edit/generate-voice/{video_id}")
def api_generate_voice(video_id: str, payload: Optional[VoicePayload] = None, db: Session = Depends(get_db)):
    voice_name = payload.voice_name if payload else None
    res = auto_editor.generate_voice_and_sync(db, video_id, voice_name=voice_name)
    status_code = 200 if res.get("success") else 400
    return JSONResponse(content=res, status_code=status_code)


@router.post("/api/auto-edit/generate-subtitles/{video_id}")
def api_generate_subtitles(video_id: str, payload: Optional[SubtitlePayload] = None, db: Session = Depends(get_db)):
    p = payload.dict() if payload else {}
    res = auto_editor.generate_subtitles(
        db,
        video_id,
        font_name=p.get("font_name"),
        font_size=p.get("font_size"),
        primary_color=p.get("primary_color"),
        outline_color=p.get("outline_color"),
        outline_width=p.get("outline_width"),
        position=p.get("position"),
        custom_pos_y=p.get("custom_pos_y"),
        animation=p.get("animation")
    )
    status_code = 200 if res.get("success") else 400
    return JSONResponse(content=res, status_code=status_code)


@router.post("/api/auto-edit/preview-subtitles/{video_id}")
def api_preview_subtitles(video_id: str, payload: Optional[SubtitlePayload] = None, db: Session = Depends(get_db)):
    p = payload.dict() if payload else {}
    res = auto_editor.generate_subtitle_preview_frame(
        db,
        video_id,
        font_name=p.get("font_name"),
        font_size=p.get("font_size"),
        primary_color=p.get("primary_color"),
        outline_color=p.get("outline_color"),
        outline_width=p.get("outline_width"),
        position=p.get("position"),
        custom_pos_y=p.get("custom_pos_y"),
        animation=p.get("animation"),
        sample_text=p.get("sample_text")
    )
    status_code = 200 if res.get("success") else 400
    return JSONResponse(content=res, status_code=status_code)


@router.post("/api/auto-edit/render/{video_id}")
def api_render_video(video_id: str, payload: RenderPayload, db: Session = Depends(get_db)):
    res = auto_editor.render_final_video(
        db,
        video_id,
        cover_type=payload.cover_type,
        blur_strength=payload.blur_strength,
        source_audio=payload.source_audio,
        include_subtitles=payload.include_subtitles,
        include_hook=payload.include_hook
    )
    status_code = 200 if res.get("success") else 400
    return JSONResponse(content=res, status_code=status_code)


@router.post("/api/auto-edit/switch-mode/{video_id}")
def api_switch_mode(video_id: str, db: Session = Depends(get_db)):
    res = auto_editor.switch_to_capcut_manual(db, video_id)
    status_code = 200 if res.get("success") else 400
    return JSONResponse(content=res, status_code=status_code)


@router.post("/api/auto-edit/batch/start")
def api_batch_start(payload: BatchStartPayload):
    res = batch_editor.start_batch(
        video_ids=payload.video_ids,
        cover_type=payload.cover_type or "blur",
        blur_strength=payload.blur_strength or 10,
        voice_name=payload.voice_name or "Trúc Ly"
    )
    status_code = 200 if res.get("success") else 400
    return JSONResponse(content=res, status_code=status_code)


@router.get("/api/auto-edit/batch/status")
def api_batch_status():
    status = batch_editor.get_status()
    return JSONResponse(content=status, status_code=200)


@router.post("/api/auto-edit/batch/retry-failed")
def api_batch_retry_failed():
    res = batch_editor.retry_failed()
    status_code = 200 if res.get("success") else 400
    return JSONResponse(content=res, status_code=status_code)
