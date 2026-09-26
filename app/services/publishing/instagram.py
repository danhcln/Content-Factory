import os
import json
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional

from app.services.publishing.base import BasePublisherProvider

logger = logging.getLogger("app.services.publishing.instagram")


class InstagramProvider(BasePublisherProvider):
    platform_key: str = "instagram"
    platform_name: str = "Instagram Reels"
    requires_manual: bool = False

    def __init__(self):
        self.api_version = "v19.0"

    def _get_credentials(self) -> tuple[Optional[str], Optional[str]]:
        token = os.getenv("INSTAGRAM_ACCESS_TOKEN", "").strip() or None
        account_id = os.getenv("INSTAGRAM_ACCOUNT_ID", "").strip() or None
        return token, account_id

    def get_status(self, db: Optional[Any] = None) -> Dict[str, Any]:
        token, account_id = self._get_credentials()
        if not token or not account_id:
            return {
                "platform": self.platform_key,
                "name": self.platform_name,
                "status": "NOT_CONNECTED",
                "connected": False,
                "account_name": None,
                "account_id": None,
                "requires_manual": self.requires_manual,
                "message": "Missing INSTAGRAM_ACCESS_TOKEN or INSTAGRAM_ACCOUNT_ID in environment"
            }
        return {
            "platform": self.platform_key,
            "name": self.platform_name,
            "status": "READY",
            "connected": True,
            "account_name": f"Instagram Business ({account_id})",
            "account_id": account_id,
            "requires_manual": self.requires_manual,
            "message": "Configured and ready"
        }

    def test_connection(self) -> Dict[str, Any]:
        token, account_id = self._get_credentials()
        if not token or not account_id:
            return {
                "success": False,
                "status": "NOT_CONNECTED",
                "message": "Cannot test connection: INSTAGRAM_ACCESS_TOKEN or INSTAGRAM_ACCOUNT_ID is not set in .env"
            }

        url = f"https://graph.facebook.com/{self.api_version}/{account_id}?fields=id,username,name&access_token={token}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "success": True,
                    "status": "READY",
                    "message": f"Successfully connected to Instagram: @{data.get('username', account_id)}",
                    "account_info": {
                        "username": data.get("username"),
                        "name": data.get("name"),
                        "id": data.get("id")
                    }
                }
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="ignore")
            logger.warning(f"Instagram API HTTPError {he.code}: {err_body}")
            return {
                "success": False,
                "status": "FAILED",
                "message": f"Instagram API Error ({he.code}): {err_body}"
            }
        except Exception as e:
            logger.error(f"Instagram connection test failed: {e}")
            return {
                "success": False,
                "status": "FAILED",
                "message": f"Connection check failed: {str(e)}"
            }

    def publish_video(
        self,
        video_path: Path,
        caption: str,
        hashtags: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        val = self.validate_media(video_path)
        if not val["valid"]:
            return {
                "success": False,
                "status": "FAILED",
                "platform": self.platform_key,
                "error": val["error"]
            }

        token, account_id = self._get_credentials()
        if not token or not account_id:
            return {
                "success": False,
                "status": "NOT_CONNECTED",
                "platform": self.platform_key,
                "error": "Instagram credentials not configured in .env"
            }

        full_caption = caption.strip()
        if hashtags and hashtags.strip():
            full_caption = f"{full_caption}\n\n{hashtags.strip()}"

        video_url = metadata.get("public_video_url") if metadata else None
        if not video_url:
            return {
                "success": False,
                "status": "FAILED",
                "platform": self.platform_key,
                "error": "Instagram Reels container requires a publicly accessible video URL (public_video_url)"
            }

        # Step 1: Create media container
        container_url = (
            f"https://graph.facebook.com/{self.api_version}/{account_id}/media"
            f"?media_type=REELS"
            f"&video_url={urllib.parse.quote(video_url)}"
            f"&caption={urllib.parse.quote(full_caption)}"
            f"&access_token={token}"
        )

        try:
            req = urllib.request.Request(container_url, method="POST")
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                cont_data = json.loads(resp.read().decode("utf-8"))
                container_id = cont_data.get("id")

            if not container_id:
                return {
                    "success": False,
                    "status": "FAILED",
                    "platform": self.platform_key,
                    "error": "Failed to create Instagram media container"
                }

            # Step 2: Publish container
            publish_url = (
                f"https://graph.facebook.com/{self.api_version}/{account_id}/media_publish"
                f"?creation_id={container_id}&access_token={token}"
            )
            pub_req = urllib.request.Request(publish_url, method="POST")
            with urllib.request.urlopen(pub_req, timeout=30.0) as resp:
                pub_data = json.loads(resp.read().decode("utf-8"))
                media_id = pub_data.get("id")

            return {
                "success": True,
                "status": "PUBLISHED",
                "platform": self.platform_key,
                "post_id": media_id,
                "post_url": f"https://www.instagram.com/p/{media_id}" if media_id else None,
                "error": None
            }
        except urllib.error.HTTPError as he:
            err = he.read().decode("utf-8", errors="ignore")
            logger.error(f"Instagram publish error {he.code}: {err}")
            return {
                "success": False,
                "status": "FAILED",
                "platform": self.platform_key,
                "error": f"API Error {he.code}: {err}"
            }
        except Exception as e:
            logger.error(f"Instagram publish failed: {e}")
            return {
                "success": False,
                "status": "FAILED",
                "platform": self.platform_key,
                "error": str(e)
            }
