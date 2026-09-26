import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from app.database import FINAL_DIR
from app.models import Video, Content, Publishing, SocialAccount
from app.services.publishing.base import BasePublisherProvider
from app.services.publishing.facebook_page import FacebookPageProvider
from app.services.publishing.instagram import InstagramProvider
from app.services.publishing.tiktok import TikTokProvider
from app.services.publishing.threads import ThreadsProvider
from app.services.publishing.youtube import YouTubeProvider
from app.services.publishing.shopee import ShopeeProvider

logger = logging.getLogger("app.services.publishing.manager")


class PublishingManager:
    """
    Central coordinator orchestrating social media connections, validation,
    one-click multi-platform publishing, and safe retry execution.
    """

    def __init__(self):
        self.providers: Dict[str, BasePublisherProvider] = {
            "facebook_page": FacebookPageProvider(),
            "instagram": InstagramProvider(),
            "tiktok": TikTokProvider(),
            "threads": ThreadsProvider(),
            "youtube": YouTubeProvider(),
            "shopee": ShopeeProvider()
        }

    def get_provider(self, platform: str) -> Optional[BasePublisherProvider]:
        return self.providers.get(platform.strip().lower())

    def get_all_connections(self, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        """Return connectivity and readiness status for all target social media platforms."""
        connections = []
        for key, prov in self.providers.items():
            status_info = prov.get_status(db)
            if db:
                acc = db.query(SocialAccount).filter(SocialAccount.platform == key).first()
                if acc:
                    status_info["db_display_name"] = acc.display_name
                    status_info["last_checked_at"] = acc.last_checked_at.isoformat() if acc.last_checked_at else None
                    status_info["last_error"] = acc.last_error
            connections.append(status_info)
        return connections

    def test_connection(self, platform: str, db: Optional[Session] = None) -> Dict[str, Any]:
        """Test API connectivity for a specific platform and record check in database."""
        prov = self.get_provider(platform)
        if not prov:
            return {"success": False, "status": "FAILED", "error": f"Unknown platform: {platform}"}

        res = prov.test_connection()

        if db:
            acc = db.query(SocialAccount).filter(SocialAccount.platform == platform).first()
            if not acc:
                acc = SocialAccount(platform=platform)
                db.add(acc)
            acc.is_connected = bool(res.get("success") and not prov.requires_manual)
            acc.status = res.get("status", "FAILED")
            acc.last_checked_at = datetime.utcnow()
            acc.last_error = None if res.get("success") else res.get("message")
            if res.get("account_info"):
                acc.display_name = str(res["account_info"].get("name") or res["account_info"].get("username") or res["account_info"].get("channel") or platform)
            db.commit()

        return res

    def test_all_connections(self, db: Optional[Session] = None) -> Dict[str, Any]:
        """Run connectivity checks across all platform providers."""
        results = {}
        for key in self.providers.keys():
            results[key] = self.test_connection(key, db=db)
        return results

    def publish_to_all(
        self,
        video_id: str,
        db: Session,
        final_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Execute one-click publication across all 7 platforms:
        1. Validate final video existence and format.
        2. Load tailored captions and hashtags per platform.
        3. Attempt upload for connected platforms; record manual requirement for unsupported platforms.
        4. Continue execution if an individual platform fails (no premature abort).
        5. Persist comprehensive results into database.
        """
        # Step 1: Validate video
        target_dir = final_dir or FINAL_DIR
        video_path = target_dir / f"{video_id}.mp4"
        if not video_path.exists():
            return {
                "success": False,
                "video_id": video_id,
                "error": f"Final video file not found at: {video_path}"
            }

        video_rec = db.query(Video).filter(Video.video_id == video_id).first()
        if not video_rec:
            return {"success": False, "video_id": video_id, "error": f"Video record {video_id} not found in database"}

        # Step 2: Load captions from Content table or fallback to product info
        content = video_rec.content
        product = video_rec.product
        prod_title = product.name_vietnamese if product else f"Sản phẩm {video_id}"
        fallback_caption = f"{prod_title} - Xem ngay để nhận ưu đãi!"
        fallback_tags = "#xuhuong #review #tienich"

        def get_caption_data(plat: str) -> tuple[str, str, Optional[str]]:
            if not content:
                return fallback_caption, fallback_tags, prod_title
            if plat == "facebook_page":
                return (content.facebook_page_caption or fallback_caption), (content.facebook_page_hashtags or fallback_tags), None
            elif plat == "instagram":
                return (content.instagram_caption or fallback_caption), (content.instagram_hashtags or fallback_tags), None
            elif plat == "tiktok":
                return (content.tiktok_caption or fallback_caption), (content.tiktok_hashtags or fallback_tags), None
            elif plat == "threads":
                return (content.threads_caption or fallback_caption), (content.threads_hashtags or fallback_tags), None
            elif plat == "youtube":
                return (content.youtube_description or fallback_caption), (content.youtube_hashtags or fallback_tags), (content.youtube_title or prod_title)
            elif plat == "shopee":
                return (content.shopee_caption or fallback_caption), (content.shopee_hashtags or fallback_tags), None
            return fallback_caption, fallback_tags, None

        # Step 3: Iterate through all platforms
        report_items = {}
        post_ids = {}
        errors = {}
        published_count = 0

        for key, prov in self.providers.items():
            caption, hashtags, title = get_caption_data(key)

            if prov.requires_manual:
                report_items[key] = {
                    "platform": key,
                    "name": prov.platform_name,
                    "status": "MANUAL_REQUIRED",
                    "success": False,
                    "message": "Manual publishing required (No official upload API available for general accounts)"
                }
                continue

            status_info = prov.get_status(db)
            if status_info["status"] != "READY":
                report_items[key] = {
                    "platform": key,
                    "name": prov.platform_name,
                    "status": "NOT_CONNECTED",
                    "success": False,
                    "message": status_info.get("message", "API credentials not configured in .env")
                }
                continue

            # Execute real upload
            try:
                res = prov.publish_video(
                    video_path=video_path,
                    caption=caption,
                    hashtags=hashtags,
                    title=title
                )
                report_items[key] = res
                if res.get("success"):
                    published_count += 1
                    if res.get("post_id"):
                        post_ids[key] = res["post_id"]
                else:
                    errors[key] = res.get("error", "Unknown publish failure")
            except Exception as pe:
                logger.error(f"Error publishing to {key}: {pe}")
                err_str = str(pe)
                report_items[key] = {
                    "platform": key,
                    "name": prov.platform_name,
                    "status": "FAILED",
                    "success": False,
                    "error": err_str
                }
                errors[key] = err_str

        # Step 4: Update Publishing database record
        pub = db.query(Publishing).filter(Publishing.video_id == video_id).first()
        if not pub:
            pub = Publishing(video_id=video_id)
            db.add(pub)

        pub.facebook_page = bool(report_items.get("facebook_page", {}).get("success"))
        pub.instagram = bool(report_items.get("instagram", {}).get("success"))
        pub.tiktok = bool(report_items.get("tiktok", {}).get("success"))
        pub.threads = bool(report_items.get("threads", {}).get("success"))
        pub.youtube = bool(report_items.get("youtube", {}).get("success"))
        # Manual platforms remain unchanged or set based on manual status
        pub.published_count = published_count
        pub.published_at = datetime.utcnow() if published_count > 0 else None
        pub.publish_status = "COMPLETED" if published_count >= 5 else ("PARTIAL" if published_count > 0 else "FAILED")
        pub.platform_post_id = json.dumps(post_ids)
        pub.error_message = json.dumps(errors) if errors else None

        db.commit()

        return {
            "success": True,
            "video_id": video_id,
            "published_count": published_count,
            "publish_status": pub.publish_status,
            "report": report_items,
            "post_ids": post_ids,
            "errors": errors
        }

    def retry_failed(
        self,
        video_id: str,
        platform: str,
        db: Session,
        final_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Safely retry publishing for a specific failed platform.
        Guarantees idempotency: rejects retry if already marked published.
        """
        plat = platform.strip().lower()
        prov = self.get_provider(plat)
        if not prov:
            return {"success": False, "error": f"Unknown platform: {plat}"}

        pub = db.query(Publishing).filter(Publishing.video_id == video_id).first()
        if pub and getattr(pub, plat, False):
            return {
                "success": False,
                "status": "ALREADY_PUBLISHED",
                "message": f"Video {video_id} is already marked published on {plat}. Duplicate upload prevented."
            }

        target_dir = final_dir or FINAL_DIR
        video_path = target_dir / f"{video_id}.mp4"
        if not video_path.exists():
            return {"success": False, "error": f"Video file not found at: {video_path}"}

        video_rec = db.query(Video).filter(Video.video_id == video_id).first()
        content = video_rec.content if video_rec else None

        caption = "Review sản phẩm chất lượng"
        hashtags = "#review #xuhuong"
        title = None
        if content:
            if plat == "facebook_page":
                caption = content.facebook_page_caption or caption
                hashtags = content.facebook_page_hashtags or hashtags
            elif plat == "instagram":
                caption = content.instagram_caption or caption
                hashtags = content.instagram_hashtags or hashtags
            elif plat == "tiktok":
                caption = content.tiktok_caption or caption
                hashtags = content.tiktok_hashtags or hashtags
            elif plat == "threads":
                caption = content.threads_caption or caption
                hashtags = content.threads_hashtags or hashtags
            elif plat == "youtube":
                caption = content.youtube_description or caption
                hashtags = content.youtube_hashtags or hashtags
                title = content.youtube_title

        res = prov.publish_video(video_path=video_path, caption=caption, hashtags=hashtags, title=title)

        if res.get("success") and pub:
            setattr(pub, plat, True)
            pub.published_count = (pub.published_count or 0) + 1
            if res.get("post_id"):
                existing_posts = {}
                try:
                    if pub.platform_post_id:
                        existing_posts = json.loads(pub.platform_post_id)
                except Exception:
                    pass
                existing_posts[plat] = res["post_id"]
                pub.platform_post_id = json.dumps(existing_posts)
            db.commit()

        return res
