import os
import subprocess
from datetime import datetime
from typing import Optional
from pathlib import Path
from pydantic import BaseModel

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db, FINAL_DIR
from app.models import Publishing, Video, Content

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(tags=["publishing"])

VALID_PLATFORMS = [
    "facebook_personal",
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


def calculate_publishing_status(pub: Publishing) -> tuple[int, str]:
    """Calculate published_count (0-7) and status string."""
    count = (
        int(bool(pub.facebook_personal)) +
        int(bool(pub.facebook_page)) +
        int(bool(pub.tiktok)) +
        int(bool(pub.threads)) +
        int(bool(pub.instagram)) +
        int(bool(pub.shopee)) +
        int(bool(pub.youtube))
    )
    if count == 7:
        status = "COMPLETED"
    elif count > 0:
        status = "PARTIAL"
    else:
        status = "NOT PUBLISHED"
    return count, status


@router.get("/publishing", response_class=HTMLResponse)
def get_publishing_page(request: Request, db: Session = Depends(get_db)):
    # Find all videos that have reached CONTENT_READY, PUBLISHING, PARTIAL, or COMPLETED
    videos = db.query(Video).filter(
        Video.status.in_(["CONTENT_READY", "PUBLISHING", "PARTIAL", "COMPLETED"])
    ).order_by(Video.id.desc()).all()

    # Ensure each has a Publishing row
    for v in videos:
        if not v.publishing:
            pub = Publishing(video_id=v.video_id, published_count=0, status="NOT PUBLISHED")
            db.add(pub)
    db.commit()

    return templates.TemplateResponse(
        request=request,
        name="publishing.html",
        context={
            "videos": videos,
            "active_page": "publishing"
        }
    )


@router.post("/api/publishing/toggle/{video_id}")
def api_toggle_platform(
    video_id: str,
    payload: TogglePlatformPayload,
    db: Session = Depends(get_db)
):
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

    # Set boolean value for platform
    setattr(pub, plat, bool(payload.posted))

    # Recalculate count and status
    count, status = calculate_publishing_status(pub)
    pub.published_count = count
    pub.status = status
    if count == 7:
        pub.publish_date = datetime.utcnow()
        video.status = "COMPLETED"
    elif count > 0:
        video.status = "PARTIAL"
    else:
        video.status = "CONTENT_READY"

    db.commit()
    db.refresh(pub)
    db.refresh(video)

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
        # Check case insensitivity
        candidates = list(FINAL_DIR.glob(f"{video_id}.*"))
        if candidates:
            final_file = candidates[0]
        else:
            return JSONResponse(status_code=404, content={"success": False, "error": f"Không tìm thấy file video final tại {final_file}"})

    try:
        if hasattr(os, "startfile"):
            os.startfile(str(final_file))
        else:
            subprocess.Popen(["explorer", str(final_file)])
        return {"success": True, "message": f"Đã mở {final_file.name}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
