from pathlib import Path
from typing import Dict, Any, Optional

from app.services.publishing.base import BasePublisherProvider


class ShopeeProvider(BasePublisherProvider):
    platform_key: str = "shopee"
    platform_name: str = "Shopee Video"
    requires_manual: bool = True

    def get_status(self, db: Optional[Any] = None) -> Dict[str, Any]:
        return {
            "platform": self.platform_key,
            "name": self.platform_name,
            "status": "MANUAL_REQUIRED",
            "connected": False,
            "account_name": "Shopee App (Manual Upload)",
            "account_id": None,
            "requires_manual": True,
            "message": "Shopee does not provide a public video feed publishing API for creators. Manual upload in Shopee mobile app is required."
        }

    def test_connection(self) -> Dict[str, Any]:
        return {
            "success": True,
            "status": "MANUAL_REQUIRED",
            "message": "Shopee Video is designated as Manual Required (Official API limitation)."
        }

    def publish_video(
        self,
        video_path: Path,
        caption: str,
        hashtags: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "status": "MANUAL_REQUIRED",
            "platform": self.platform_key,
            "post_id": None,
            "post_url": "https://creator.shopee.vn/",
            "message": "Shopee Video requires manual publishing via the Shopee mobile application or Shopee Creator Hub. Please use the Copy Caption feature.",
            "error": "Manual action required by platform"
        }
