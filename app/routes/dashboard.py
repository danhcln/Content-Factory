from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pathlib import Path

from app.database import get_db
from app.models import Product, Video, Content, Publishing

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(tags=["dashboard"])


@router.get("/", response_class=HTMLResponse)
@router.get("/dashboard", response_class=HTMLResponse)
def get_dashboard(request: Request, db: Session = Depends(get_db)):
    # Calculate counts
    products_count = db.query(Product).count()
    videos_found = db.query(Video).count()
    approved_count = db.query(Video).filter(Video.approved == True).count()
    downloaded_count = db.query(Video).filter(Video.downloaded == True).count()
    waiting_capcut_count = db.query(Video).filter(Video.status == "WAITING_CAPCUT").count()
    content_ready_count = db.query(Video).filter(Video.status.in_(["CONTENT_READY", "READY"])).count()
    publishing_count = db.query(Video).filter(Video.status.in_(["PUBLISHING", "PARTIAL"])).count()
    completed_count = db.query(Video).filter(Video.status == "COMPLETED").count()

    # Recent videos
    recent_videos = db.query(Video).order_by(desc(Video.created_at)).limit(10).all()

    # Recent errors
    error_statuses = [
        "DOWNLOAD_ERROR",
        "SCRIPT_ERROR",
        "VOICE_ERROR",
        "CONTENT_ERROR",
        "SCRIPT_REVIEW_REQUIRED"
    ]
    recent_errors = db.query(Video).filter(Video.status.in_(error_statuses)).order_by(desc(Video.created_at)).limit(10).all()

    from app.services.ffmpeg_utils import check_ffmpeg_available
    from app.services.tts.vieneu_provider import VieNeuProvider
    from app.services.publishing.manager import PublishingManager
    from app.services.ai.status import get_active_provider_status

    ff_status = check_ffmpeg_available()
    vieneu_prov = VieNeuProvider()
    pub_mgr = PublishingManager()
    social_connections = pub_mgr.get_all_connections(db)
    active_ai = get_active_provider_status(db=db)

    system_status = {
        "server_running": True,
        "ffmpeg_ready": bool(ff_status.get("ready")),
        "vieneu_ready": bool(vieneu_prov.is_available()),
        "database_ready": True,
        "downloader_ready": True,
        "auto_editor_ready": bool(ff_status.get("ready")),
        "social_publishing_ready": True,
        "social_connections": social_connections,
        "ai_status": active_ai,
        "ai_ready": active_ai["configured"] and active_ai["status"] == "READY",
        "ai_name": active_ai["short_name"],
        "ai_model": active_ai["model"],
        "ai_badge_label": active_ai["badge_label"],
        "ai_badge_color": active_ai["badge_color"],
    }

    stats = {
        "products": products_count,
        "videos_found": videos_found,
        "approved": approved_count,
        "downloaded": downloaded_count,
        "waiting_capcut": waiting_capcut_count,
        "content_ready": content_ready_count,
        "publishing": publishing_count,
        "completed": completed_count,
    }

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "stats": stats,
            "recent_videos": recent_videos,
            "recent_errors": recent_errors,
            "system_status": system_status,
            "ai_status": active_ai,
            "active_page": "dashboard"
        }
    )
