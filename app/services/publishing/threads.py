import os
import json
import logging
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional

from app.services.publishing.base import BasePublisherProvider

logger = logging.getLogger("app.services.publishing.threads")


class ThreadsProvider(BasePublisherProvider):
    platform_key: str = "threads"
    platform_name: str = "Threads"
    requires_manual: bool = False

    def _get_credentials(self) -> tuple[Optional[str], Optional[str]]:
        token = os.getenv("THREADS_ACCESS_TOKEN", "").strip() or None
        user_id = os.getenv("THREADS_USER_ID", "").strip() or None
        return token, user_id

    def get_status(self, db: Optional[Any] = None) -> Dict[str, Any]:
        token, user_id = self._get_credentials()
        if not token:
            return {
                "platform": self.platform_key,
                "name": self.platform_name,
                "status": "NOT_CONNECTED",
                "connected": False,
                "account_name": None,
                "account_id": None,
                "requires_manual": self.requires_manual,
                "message": "Missing THREADS_ACCESS_TOKEN in environment"
            }
        return {
            "platform": self.platform_key,
            "name": self.platform_name,
            "status": "READY",
            "connected": True,
            "account_name": f"Threads Profile ({user_id or 'Authorized'})",
            "account_id": user_id,
            "requires_manual": self.requires_manual,
            "message": "Configured and ready"
        }

    def test_connection(self) -> Dict[str, Any]:
        token, _ = self._get_credentials()
        if not token:
            return {
                "success": False,
                "status": "NOT_CONNECTED",
                "message": "Cannot test connection: THREADS_ACCESS_TOKEN is not configured in .env"
            }

        url = f"https://graph.threads.net/v1.0/me?fields=id,username,name&access_token={token}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                uname = data.get("username", "Threads User")
                return {
                    "success": True,
                    "status": "READY",
                    "message": f"Successfully connected to Threads: @{uname}",
                    "account_info": data
                }
        except urllib.error.HTTPError as he:
            err = he.read().decode("utf-8", errors="ignore")
            logger.warning(f"Threads API check error {he.code}: {err}")
            return {
                "success": False,
                "status": "FAILED",
                "message": f"Threads API error ({he.code}): {err}"
            }
        except Exception as e:
            logger.error(f"Threads connection check failed: {e}")
            return {
                "success": False,
                "status": "FAILED",
                "message": f"Threads connection check failed: {str(e)}"
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

        token, user_id = self._get_credentials()
        if not token:
            return {
                "success": False,
                "status": "NOT_CONNECTED",
                "platform": self.platform_key,
                "error": "Threads credentials not configured in .env"
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
                "error": "Threads video publishing requires a publicly accessible video URL (public_video_url)"
            }

        uid = user_id or "me"
        # Step 1: Create Threads video media container
        init_url = (
            f"https://graph.threads.net/v1.0/{uid}/threads"
            f"?media_type=VIDEO"
            f"&video_url={urllib.parse.quote(video_url)}"
            f"&text={urllib.parse.quote(full_caption)}"
            f"&access_token={token}"
        )

        try:
            req = urllib.request.Request(init_url, method="POST")
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                cont_data = json.loads(resp.read().decode("utf-8"))
                creation_id = cont_data.get("id")

            if not creation_id:
                return {
                    "success": False,
                    "status": "FAILED",
                    "platform": self.platform_key,
                    "error": "Failed to create Threads media container"
                }

            # Step 2: Publish Threads container
            pub_url = (
                f"https://graph.threads.net/v1.0/{uid}/threads_publish"
                f"?creation_id={creation_id}"
                f"&access_token={token}"
            )
            pub_req = urllib.request.Request(pub_url, method="POST")
            with urllib.request.urlopen(pub_req, timeout=30.0) as resp:
                pub_data = json.loads(resp.read().decode("utf-8"))
                thread_id = pub_data.get("id")

            return {
                "success": True,
                "status": "PUBLISHED",
                "platform": self.platform_key,
                "post_id": thread_id,
                "post_url": f"https://www.threads.net/t/{thread_id}" if thread_id else None,
                "error": None
            }
        except urllib.error.HTTPError as he:
            err = he.read().decode("utf-8", errors="ignore")
            logger.error(f"Threads publish error {he.code}: {err}")
            return {
                "success": False,
                "status": "FAILED",
                "platform": self.platform_key,
                "error": f"API Error {he.code}: {err}"
            }
        except Exception as e:
            logger.error(f"Threads publish failed: {e}")
            return {
                "success": False,
                "status": "FAILED",
                "platform": self.platform_key,
                "error": str(e)
            }
