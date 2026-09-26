import logging
import re
import urllib.parse
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import desc, nullslast, case

from app.models import Video, Product

logger = logging.getLogger("app.services.douyin")


def normalize_douyin_url(url: str) -> str:
    """Normalize Douyin URL by trimming whitespace and extracting clean URL."""
    if not url:
        return ""
    cleaned = url.strip()
    # If text contains a URL (e.g. from Douyin app share text like '7.12 复制打开抖音... https://v.douyin.com/xyz/')
    match = re.search(r'https?://[^\s]+', cleaned)
    if match:
        cleaned = match.group(0)
    return cleaned


def extract_douyin_video_id(url: str) -> Optional[str]:
    """
    Extract Douyin video ID (modal_id or aweme_id) from URL using urllib.parse and regex.
    Supports:
    - /video/<id>
    - /discover?modal_id=<id>
    - /jingxuan/search/...?...modal_id=<id>
    - /user/...?...modal_id=<id>
    - modal_id anywhere in query params
    """
    if not url:
        return None
    cleaned = normalize_douyin_url(url)
    if not cleaned:
        return None

    try:
        parsed = urllib.parse.urlparse(cleaned)
        # 1. Check query parameters for modal_id
        qs = urllib.parse.parse_qs(parsed.query)
        if "modal_id" in qs and qs["modal_id"]:
            m_val = qs["modal_id"][0].strip()
            if re.match(r"^\d{15,25}$", m_val):
                return m_val

        # 2. Check path for /video/<id>
        path_match = re.search(r"/video/(\d{15,25})", parsed.path)
        if path_match:
            return path_match.group(1)

        # 3. Check for v.douyin.com shortlink redirect
        if "v.douyin.com" in parsed.netloc:
            try:
                import httpx
                with httpx.Client(timeout=3.0, follow_redirects=True) as client:
                    resp = client.head(cleaned)
                    target = str(resp.url)
                    redir_id = extract_douyin_video_id(target)
                    if redir_id:
                        return redir_id
            except Exception:
                pass

        # 4. Fallback search anywhere in query or path string
        m_regex = re.search(r"modal_id=(\d{15,25})", cleaned)
        if m_regex:
            return m_regex.group(1)

        vid_regex = re.search(r"/video/(\d{15,25})", cleaned)
        if vid_regex:
            return vid_regex.group(1)
    except Exception as e:
        logger.warning(f"Error parsing Douyin video ID from URL: {e}")

    return None


def get_canonical_douyin_url(url: str) -> str:
    """
    Generate canonical Douyin URL: https://www.douyin.com/video/<video_id>
    if video_id/modal_id can be extracted. Otherwise returns normalized original URL.
    """
    vid = extract_douyin_video_id(url)
    if vid:
        return f"https://www.douyin.com/video/{vid}"
    return normalize_douyin_url(url)


def get_next_video_id(db: Session) -> str:
    """Generate sequential Video ID: V0001, V0002, etc."""
    videos = db.query(Video.video_id).filter(Video.video_id.like("V%")).all()
    max_num = 0
    for (vid,) in videos:
        m = re.match(r"^V(\d+)$", vid)
        if m:
            val = int(m.group(1))
            if val > max_num:
                max_num = val
    next_num = max_num + 1
    return f"V{next_num:04d}"


class DouyinService:
    def __init__(self):
        pass

    def check_duplicate_url(self, db: Session, url: str) -> Optional[Video]:
        """Check if URL already exists in database."""
        norm_url = normalize_douyin_url(url)
        if not norm_url:
            return None
        return db.query(Video).filter(Video.douyin_url == norm_url).first()

    def add_video(
        self,
        db: Session,
        douyin_url: str,
        product_id: Optional[str] = None,
        views: Optional[str] = None,
        thumbnail: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Manually or programmatically add a Douyin video with duplicate protection."""
        norm_url = normalize_douyin_url(douyin_url)
        if not norm_url:
            return {"success": False, "error": "Douyin URL không được để trống."}

        # Check duplicate
        existing = self.check_duplicate_url(db, norm_url)
        if existing:
            return {
                "success": False,
                "error": f"URL video này đã tồn tại trong hệ thống với mã {existing.video_id}.",
                "video_id": existing.video_id,
                "duplicate": True
            }

        # Verify product if product_id is provided
        if product_id:
            prod = db.query(Product).filter(Product.product_id == product_id).first()
            if not prod:
                logger.warning(f"Product ID {product_id} not found in database.")

        vid_id = get_next_video_id(db)
        new_video = Video(
            video_id=vid_id,
            product_id=product_id if product_id else None,
            douyin_url=norm_url,
            views=views.strip() if views else None,
            thumbnail=thumbnail.strip() if thumbnail else None,
            downloaded=False,
            approved=False,
            used=False,
            status="FOUND",
            notes=notes
        )

        db.add(new_video)
        db.commit()
        db.refresh(new_video)

        logger.info(f"Video {vid_id} added successfully for product {product_id}.")
        return {
            "success": True,
            "video_id": vid_id,
            "message": f"Đã thêm video {vid_id} thành công!"
        }

    def discover_videos_compliant(self, keyword: str, limit: int = 5) -> Dict[str, Any]:
        """
        Attempt compliant public search without bypassing logins, CAPTCHAs, or anti-bot protections.
        If automated search is restricted by Douyin bot protections, returns BLOCKED_EXTERNAL status.
        """
        logger.info(f"Compliant discovery requested for keyword: {keyword}")
        # Public search on Douyin requires dynamic JS signing tokens (a_bogus/msToken) and user session.
        # As per instructions, do not implement circumvention or bypass logic.
        return {
            "success": False,
            "status": "BLOCKED_EXTERNAL",
            "message": "Tính năng tự động quét Douyin bị hạn chế do yêu cầu bảo mật/đăng nhập của Douyin. Vui lòng sử dụng tính năng Nhập Video Thủ Công (Manual Import)."
        }

    def get_sorted_videos(self, db: Session, status_filter: Optional[str] = None, sort_by_views: bool = False):
        """Retrieve videos with optional view sorting (high -> low) while keeping NULL/empty views visible."""
        query = db.query(Video)
        if status_filter:
            query = query.filter(Video.status == status_filter)

        if sort_by_views:
            # Sort integer-convertible views descending, NULLs and non-numerics last
            # SQLite safe sort: cast views to integer if numeric
            query = query.order_by(
                case(
                    (Video.views.is_(None), 1),
                    (Video.views == '', 1),
                    else_=0
                ),
                desc(Video.views),
                desc(Video.id)
            )
        else:
            query = query.order_by(desc(Video.id))

        return query.all()
