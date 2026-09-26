import os
import json
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional

from app.services.publishing.base import BasePublisherProvider

logger = logging.getLogger("app.services.publishing.facebook_page")


class FacebookPageProvider(BasePublisherProvider):
    platform_key: str = "facebook_page"
    platform_name: str = "Facebook Page"
    requires_manual: bool = False

    def __init__(self):
        self.api_version = "v19.0"

    def _get_credentials(self) -> tuple[Optional[str], Optional[str]]:
        token = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN", "").strip() or None
        page_id = os.getenv("FACEBOOK_PAGE_ID", "").strip() or None
        return token, page_id

    def get_status(self, db: Optional[Any] = None) -> Dict[str, Any]:
        token, page_id = self._get_credentials()
        if not token or not page_id:
            return {
                "platform": self.platform_key,
                "name": self.platform_name,
                "status": "NOT_CONNECTED",
                "connected": False,
                "account_name": None,
                "account_id": None,
                "requires_manual": self.requires_manual,
                "message": "Missing FACEBOOK_PAGE_ACCESS_TOKEN or FACEBOOK_PAGE_ID in environment"
            }
        return {
            "platform": self.platform_key,
            "name": self.platform_name,
            "status": "READY",
            "connected": True,
            "account_name": f"Facebook Page ({page_id})",
            "account_id": page_id,
            "requires_manual": self.requires_manual,
            "message": "Configured and ready"
        }

    def test_connection(self) -> Dict[str, Any]:
        token, page_id = self._get_credentials()
        if not token or not page_id:
            return {
                "success": False,
                "status": "NOT_CONNECTED",
                "message": "Cannot test connection: FACEBOOK_PAGE_ACCESS_TOKEN or FACEBOOK_PAGE_ID is not set in .env"
            }

        url = f"https://graph.facebook.com/{self.api_version}/{page_id}?fields=id,name,link&access_token={token}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "success": True,
                    "status": "READY",
                    "message": f"Successfully connected to Facebook Page: {data.get('name', page_id)}",
                    "account_info": {
                        "name": data.get("name"),
                        "id": data.get("id"),
                        "link": data.get("link")
                    }
                }
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="ignore")
            logger.warning(f"Facebook Page API HTTPError {he.code}: {err_body}")
            return {
                "success": False,
                "status": "FAILED",
                "message": f"Facebook Graph API Error ({he.code}): {err_body}"
            }
        except Exception as e:
            logger.error(f"Facebook Page connection test failed: {e}")
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

        token, page_id = self._get_credentials()
        if not token or not page_id:
            return {
                "success": False,
                "status": "NOT_CONNECTED",
                "platform": self.platform_key,
                "error": "Facebook Page credentials not configured in .env"
            }

        full_caption = caption.strip()
        if hashtags and hashtags.strip():
            full_caption = f"{full_caption}\n\n{hashtags.strip()}"

        # Real upload using multipart form to Meta Graph API
        url = f"https://graph.facebook.com/{self.api_version}/{page_id}/videos"
        boundary = "----WebKitFormBoundaryAIContentFactory7MA4YWxkTrZu0gW"
        
        try:
            with open(video_path, "rb") as f:
                video_data = f.read()

            body = bytearray()
            # access_token field
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(b'Content-Disposition: form-data; name="access_token"\r\n\r\n')
            body.extend(f"{token}\r\n".encode())

            # description field
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(b'Content-Disposition: form-data; name="description"\r\n\r\n')
            body.extend(f"{full_caption}\r\n".encode("utf-8"))

            if title:
                body.extend(f"--{boundary}\r\n".encode())
                body.extend(b'Content-Disposition: form-data; name="title"\r\n\r\n')
                body.extend(f"{title}\r\n".encode("utf-8"))

            # source file
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="source"; filename="{video_path.name}"\r\n'.encode())
            body.extend(b"Content-Type: video/mp4\r\n\r\n")
            body.extend(video_data)
            body.extend(b"\r\n")
            body.extend(f"--{boundary}--\r\n".encode())

            req = urllib.request.Request(
                url,
                data=bytes(body),
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=120.0) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                post_id = result.get("id")
                return {
                    "success": True,
                    "status": "PUBLISHED",
                    "platform": self.platform_key,
                    "post_id": post_id,
                    "post_url": f"https://facebook.com/{post_id}" if post_id else None,
                    "error": None
                }
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="ignore")
            logger.error(f"Facebook publish HTTPError {he.code}: {err_body}")
            return {
                "success": False,
                "status": "FAILED",
                "platform": self.platform_key,
                "error": f"API Error {he.code}: {err_body}"
            }
        except Exception as e:
            logger.error(f"Facebook publish failed: {e}")
            return {
                "success": False,
                "status": "FAILED",
                "platform": self.platform_key,
                "error": str(e)
            }
