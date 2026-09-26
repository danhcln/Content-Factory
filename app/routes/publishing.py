import os
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db, FINAL_DIR
from app.models import Publishing, Video, Content, SocialAccount
from app.services.publishing.manager import PublishingManager

logger = logging.getLogger("app.routes.publishing")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(tags=["publishing"])
publishing_manager = PublishingManager()

VALID_PLATFORMS = [
    "facebook_page",
    "tiktok",
    "threads",
    "instagram",
    "shopee",
    "youtube"
]


class TogglePlatformPayload(BaseModel):
    platform: str
    posted: bool


class SchedulePayload(BaseModel):
    scheduled_at: Optional[str] = None


def calculate_publishing_status(pub: Publishing) -> tuple[int, str]:
    """Calculate published_count (0-6) and status string across 6 platforms."""
    count = (
        int(bool(pub.facebook_page)) +
        int(bool(pub.tiktok)) +
        int(bool(pub.threads)) +
        int(bool(pub.instagram)) +
        int(bool(pub.shopee)) +
        int(bool(pub.youtube))
    )
    if count == 6:
        status = "COMPLETED"
    elif count > 0:
        status = "PARTIAL"
    else:
        status = "NOT PUBLISHED"
    return count, status


@router.get("/publishing", response_class=HTMLResponse)
def get_publishing_page(request: Request, db: Session = Depends(get_db)):
    """Publishing Dashboard with one-click multi-platform controls and manual fallback."""
    videos = db.query(Video).filter(
        Video.status.in_(["AUTO_EDIT_READY", "CONTENT_READY", "PUBLISHING", "PARTIAL", "COMPLETED", "APPROVED"])
    ).order_by(Video.id.desc()).all()

    # Ensure each video has a Publishing row
    for v in videos:
        if not v.publishing:
            pub = Publishing(video_id=v.video_id, published_count=0, status="NOT PUBLISHED")
            db.add(pub)
    db.commit()

    connections = publishing_manager.get_all_connections(db)

    return templates.TemplateResponse(
        request=request,
        name="publishing.html",
        context={
            "videos": videos,
            "connections": connections,
            "active_page": "publishing"
        }
    )


@router.get("/social-connections", response_class=HTMLResponse)
def get_social_connections_page(request: Request, db: Session = Depends(get_db)):
    """Social Accounts Connection management panel."""
    connections = publishing_manager.get_all_connections(db)
    return templates.TemplateResponse(
        request=request,
        name="social_connections.html",
        context={
            "connections": connections,
            "active_page": "social_connections"
        }
    )


@router.get("/api/publishing/connections")
def api_get_connections(db: Session = Depends(get_db)):
    """Retrieve connection readiness for all target social media platforms."""
    connections = publishing_manager.get_all_connections(db)
    return JSONResponse(content={"success": True, "connections": connections})


@router.post("/api/publishing/test-connection/{platform}")
def api_test_platform_connection(platform: str, db: Session = Depends(get_db)):
    """Test API connection for a specific platform."""
    res = publishing_manager.test_connection(platform, db=db)
    status_code = 200 if res.get("success") or res.get("status") == "MANUAL_REQUIRED" else 400
    return JSONResponse(status_code=status_code, content=res)


@router.post("/api/publishing/test-all-connections")
def api_test_all_connections(db: Session = Depends(get_db)):
    """Test API connections across all 7 platforms."""
    res = publishing_manager.test_all_connections(db=db)
    return JSONResponse(content={"success": True, "results": res})


@router.post("/api/publishing/publish-all/{video_id}")
def api_publish_to_all(video_id: str, db: Session = Depends(get_db)):
    """
    One-click Publish to All:
    Validates final video, loads platform captions, attempts official API upload
    for connected platforms, marks manual platforms, and returns full publishing report.
    """
    res = publishing_manager.publish_to_all(video_id=video_id, db=db)
    status_code = 200 if res.get("success") else 400
    return JSONResponse(status_code=status_code, content=res)


@router.post("/api/publishing/retry/{video_id}/{platform}")
def api_retry_failed_platform(video_id: str, platform: str, db: Session = Depends(get_db)):
    """Retry publishing for a single failed platform with duplicate protection."""
    res = publishing_manager.retry_failed(video_id=video_id, platform=platform, db=db)
    status_code = 200 if res.get("success") else 400
    return JSONResponse(status_code=status_code, content=res)


@router.post("/api/publishing/schedule/{video_id}")
def api_schedule_video(video_id: str, payload: SchedulePayload, db: Session = Depends(get_db)):
    """Schedule video publication timestamp."""
    pub = db.query(Publishing).filter(Publishing.video_id == video_id).first()
    if not pub:
        pub = Publishing(video_id=video_id)
        db.add(pub)

    if payload.scheduled_at:
        try:
            pub.scheduled_at = datetime.fromisoformat(payload.scheduled_at.replace("Z", "+00:00"))
            pub.publish_status = "SCHEDULED"
        except Exception as e:
            return JSONResponse(status_code=400, content={"success": False, "error": f"Invalid date format: {e}"})
    else:
        pub.scheduled_at = None
        pub.publish_status = "NOT_PUBLISHED"

    db.commit()
    return {"success": True, "video_id": video_id, "scheduled_at": pub.scheduled_at.isoformat() if pub.scheduled_at else None}


@router.post("/api/publishing/toggle/{video_id}")
def api_toggle_platform(
    video_id: str,
    payload: TogglePlatformPayload,
    db: Session = Depends(get_db)
):
    """Preserve manual toggle for fallback."""
    plat = payload.platform.strip().lower()
    if plat not in VALID_PLATFORMS:
        return JSONResponse(status_code=400, content={"success": False, "error": f"Nền tảng không hợp lệ: {plat}"})

    video = db.query(Video).filter(Video.video_id == video_id).first()
    if not video:
        return JSONResponse(status_code=404, content={"success": False, "error": f"Không tìm thấy video {video_id}"})

    pub = db.query(Publishing).filter(Publishing.video_id == video_id).first()
    if not pub:
        pub = Publishing(video_id=video_id)
        db.add(pub)

    setattr(pub, plat, bool(payload.posted))

    count, status = calculate_publishing_status(pub)
    pub.published_count = count
    pub.status = status
    pub.publish_status = status
    if count == 6:
        pub.publish_date = datetime.utcnow()
        video.status = "COMPLETED"
    elif count > 0:
        video.status = "PARTIAL"

    db.commit()

    return {
        "success": True,
        "video_id": video_id,
        "platform": plat,
        "posted": payload.posted,
        "published_count": count,
        "status": status,
        "video_status": video.status
    }


@router.get("/api/publishing/open-final/{video_id}")
def api_open_final_video(video_id: str):
    """Open final exported video in Windows default media player."""
    final_file = FINAL_DIR / f"{video_id}.mp4"
    if not final_file.exists():
        candidates = list(FINAL_DIR.glob(f"{video_id}.*"))
        if candidates:
            final_file = candidates[0]
        else:
            return JSONResponse(status_code=404, content={"success": False, "error": f"Video file not found at {final_file}"})

    try:
        if hasattr(os, "startfile"):
            os.startfile(str(final_file))
        else:
            subprocess.Popen(["explorer", str(final_file)])
        return {"success": True, "message": f"Opened {final_file.name}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
