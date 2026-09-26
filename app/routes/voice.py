from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path
from pydantic import BaseModel

from app.database import get_db
from app.models import Voice, Video
from app.services.script_service import ScriptService

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(tags=["voice"])
script_service = ScriptService()


class EditScriptPayload(BaseModel):
    script: str


@router.get("/voice", response_class=HTMLResponse)
def get_voice_page(request: Request, db: Session = Depends(get_db)):
    voices = db.query(Voice).order_by(Voice.id.desc()).all()
    # Also fetch videos that are downloaded and ready for scripting or voice
    downloaded_videos = db.query(Video).filter(
        Video.status.in_(["DOWNLOADED", "SCRIPTING", "SCRIPT_READY", "SCRIPT_REVIEW_REQUIRED", "SCRIPT_ERROR"])
    ).order_by(Video.id.desc()).all()

    return templates.TemplateResponse(
        request=request,
        name="voice.html",
        context={
            "voices": voices,
            "downloaded_videos": downloaded_videos,
            "active_page": "voice"
        }
    )


@router.post("/api/voice/generate/{video_id}")
def api_generate_script(video_id: str, db: Session = Depends(get_db)):
    result = script_service.generate_script_for_video(db, video_id, regenerate=False)
    status_code = 200 if result.get("success") else 400
    return JSONResponse(content=result, status_code=status_code)


@router.post("/api/voice/regenerate/{video_id}")
def api_regenerate_script(video_id: str, db: Session = Depends(get_db)):
    result = script_service.generate_script_for_video(db, video_id, regenerate=True)
    status_code = 200 if result.get("success") else 400
    return JSONResponse(content=result, status_code=status_code)


@router.post("/api/voice/edit/{video_id}")
def api_edit_script(video_id: str, payload: EditScriptPayload, db: Session = Depends(get_db)):
    result = script_service.edit_script(db, video_id, payload.script)
    status_code = 200 if result.get("success") else 400
    return JSONResponse(content=result, status_code=status_code)
