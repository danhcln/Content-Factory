from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path

from app.database import get_db
from app.models import Content, Video
from app.services.content_service import ContentService

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(tags=["content"])
content_service = ContentService()


@router.get("/content", response_class=HTMLResponse)
def get_content_page(request: Request, db: Session = Depends(get_db)):
    content_items = db.query(Content).order_by(Content.id.desc()).all()
    # Also fetch videos that are in READY state
    ready_videos = db.query(Video).filter(Video.status.in_(["READY", "CONTENT_ERROR"])).order_by(Video.id.desc()).all()
    return templates.TemplateResponse(
        request=request,
        name="content.html",
        context={
            "content_items": content_items,
            "ready_videos": ready_videos,
            "active_page": "content"
        }
    )


@router.post("/api/content/generate/{video_id}")
def api_generate_content(video_id: str, db: Session = Depends(get_db)):
    result = content_service.generate_content_for_video(db, video_id)
    status_code = 200 if result.get("success") else 400
    return JSONResponse(content=result, status_code=status_code)
