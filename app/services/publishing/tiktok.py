import os
import json
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional

from app.services.publishing.base import BasePublisherProvider

logger = logging.getLogger("app.services.publishing.tiktok")


class TikTokProvider(BasePublisherProvider):
    platform_key: str = "tiktok"
    platform_name: str = "TikTok"
    requires_manual: bool = False

    def _get_credentials(self) -> tuple[Optional[str], Optional[str]]:
        token = os.getenv("TIKTOK_ACCESS_TOKEN", "").strip() or None
        open_id = os.getenv("TIKTOK_OPEN_ID", "").strip() or None
        return token, open_id

    def get_status(self, db: Optional[Any] = None) -> Dict[str, Any]:
        token, open_id = self._get_credentials()
        if not token:
            return {
                "platform": self.platform_key,
                "name": self.platform_name,
                "status": "NOT_CONNECTED",
                "connected": False,
                "account_name": None,
                "account_id": None,
                "requires_manual": self.requires_manual,
                "message": "Missing TIKTOK_ACCESS_TOKEN in environment"
            }
        return {
            "platform": self.platform_key,
            "name": self.platform_name,
            "status": "READY",
            "connected": True,
            "account_name": f"TikTok Creator ({open_id or 'Authorized'})",
            "account_id": open_id,
            "requires_manual": self.requires_manual,
            "message": "Configured and ready"
        }

    def test_connection(self) -> Dict[str, Any]:
        token, open_id = self._get_credentials()
        if not token:
            return {
                "success": False,
                "status": "NOT_CONNECTED",
                "message": "Cannot test connection: TIKTOK_ACCESS_TOKEN is not configured in .env"
            }

        url = "https://open.tiktokapis.com/v2/user/info/?fields=open_id,union_id,avatar_url,display_name"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                user_data = data.get("data", {}).get("user", {})
                dname = user_data.get("display_name", open_id or "TikTok Creator")
                return {
                    "success": True,
                    "status": "READY",
                    "message": f"Successfully connected to TikTok account: {dname}",
                    "account_info": user_data
                }
        except urllib.error.HTTPError as he:
            err = he.read().decode("utf-8", errors="ignore")
            logger.warning(f"TikTok API check error {he.code}: {err}")
            return {
                "success": False,
                "status": "FAILED",
                "message": f"TikTok API error ({he.code}): {err}"
            }
        except Exception as e:
            logger.error(f"TikTok connection check failed: {e}")
            return {
                "success": False,
                "status": "FAILED",
                "message": f"TikTok connection check failed: {str(e)}"
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

        token, _ = self._get_credentials()
        if not token:
            return {
                "success": False,
                "status": "NOT_CONNECTED",
                "platform": self.platform_key,
                "error": "TikTok credentials not configured in .env"
            }

        full_caption = caption.strip()
        if hashtags and hashtags.strip():
            full_caption = f"{full_caption} {hashtags.strip()}"

        # TikTok Content Posting API v2 init
        url = "https://open.tiktokapis.com/v2/post/publish/video/init/"
        payload = {
            "post_info": {
                "title": full_caption[:150],
                "privacy_level": "PUBLIC_TO_EVERYONE",
                "disable_duet": False,
                "disable_stitch": False,
                "disable_comment": False
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": val["size_bytes"],
                "chunk_size": val["size_bytes"],
                "total_chunk_count": 1
            }
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json; charset=UTF-8"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            err_code = data.get("error", {}).get("code", "0")
            if err_code != "ok" and err_code != 0:
                err_msg = data.get("error", {}).get("message", "Unknown TikTok API error")
                return {
                    "success": False,
                    "status": "FAILED",
                    "platform": self.platform_key,
                    "error": f"TikTok API error: {err_msg}"
                }

            upload_url = data.get("data", {}).get("upload_url")
            publish_id = data.get("data", {}).get("publish_id")

            # Upload video binary to provided upload_url
            if upload_url:
                with open(video_path, "rb") as vf:
                    vbytes = vf.read()
                up_req = urllib.request.Request(
                    upload_url,
                    data=vbytes,
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Range": f"bytes 0-{len(vbytes)-1}/{len(vbytes)}"
                    },
                    method="PUT"
                )
                with urllib.request.urlopen(up_req, timeout=120.0) as up_resp:
                    pass

            return {
                "success": True,
                "status": "PUBLISHED",
                "platform": self.platform_key,
                "post_id": publish_id,
                "post_url": f"https://www.tiktok.com/@creator/video/{publish_id}" if publish_id else None,
                "error": None
            }
        except urllib.error.HTTPError as he:
            err = he.read().decode("utf-8", errors="ignore")
            logger.error(f"TikTok publish error {he.code}: {err}")
            return {
                "success": False,
                "status": "FAILED",
                "platform": self.platform_key,
                "error": f"API Error {he.code}: {err}"
            }
        except Exception as e:
            logger.error(f"TikTok publish failed: {e}")
            return {
                "success": False,
                "status": "FAILED",
                "platform": self.platform_key,
                "error": str(e)
            }
