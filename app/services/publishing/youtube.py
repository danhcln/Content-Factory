import os
import json
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional

from app.services.publishing.base import BasePublisherProvider

logger = logging.getLogger("app.services.publishing.youtube")


class YouTubeProvider(BasePublisherProvider):
    platform_key: str = "youtube"
    platform_name: str = "YouTube Shorts"
    requires_manual: bool = False

    def _get_credentials(self) -> tuple[Optional[str], Optional[str]]:
        token = os.getenv("YOUTUBE_ACCESS_TOKEN", "").strip() or None
        api_key = os.getenv("YOUTUBE_API_KEY", "").strip() or None
        return token, api_key

    def get_status(self, db: Optional[Any] = None) -> Dict[str, Any]:
        token, api_key = self._get_credentials()
        if not token and not api_key:
            return {
                "platform": self.platform_key,
                "name": self.platform_name,
                "status": "NOT_CONNECTED",
                "connected": False,
                "account_name": None,
                "account_id": None,
                "requires_manual": self.requires_manual,
                "message": "Missing YOUTUBE_ACCESS_TOKEN or YOUTUBE_API_KEY in environment"
            }
        return {
            "platform": self.platform_key,
            "name": self.platform_name,
            "status": "READY",
            "connected": True,
            "account_name": "YouTube Channel (OAuth Authorized)",
            "account_id": "Authorized",
            "requires_manual": self.requires_manual,
            "message": "Configured and ready"
        }

    def test_connection(self) -> Dict[str, Any]:
        token, api_key = self._get_credentials()
        if not token and not api_key:
            return {
                "success": False,
                "status": "NOT_CONNECTED",
                "message": "Cannot test connection: YOUTUBE_ACCESS_TOKEN or YOUTUBE_API_KEY is not configured in .env"
            }

        headers = {}
        if token:
            url = "https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true"
            headers["Authorization"] = f"Bearer {token}"
        else:
            url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&chart=mostPopular&maxResults=1&key={api_key}"

        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                items = data.get("items", [])
                channel_title = items[0].get("snippet", {}).get("title", "YouTube Channel") if items else "YouTube API Valid"
                return {
                    "success": True,
                    "status": "READY",
                    "message": f"Successfully connected to YouTube: {channel_title}",
                    "account_info": {"channel": channel_title}
                }
        except urllib.error.HTTPError as he:
            err = he.read().decode("utf-8", errors="ignore")
            logger.warning(f"YouTube connection check HTTPError {he.code}: {err}")
            return {
                "success": False,
                "status": "FAILED",
                "message": f"YouTube API Error ({he.code}): {err}"
            }
        except Exception as e:
            logger.error(f"YouTube connection check failed: {e}")
            return {
                "success": False,
                "status": "FAILED",
                "message": f"YouTube connection check failed: {str(e)}"
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
                "error": "YouTube video uploading requires OAuth2 YOUTUBE_ACCESS_TOKEN with upload permission"
            }

        video_title = title or caption[:90]
        if "#shorts" not in video_title.lower():
            video_title = f"{video_title} #shorts"

        full_desc = caption.strip()
        if hashtags:
            full_desc = f"{full_desc}\n\n{hashtags.strip()}"

        snippet_metadata = {
            "snippet": {
                "title": video_title[:100],
                "description": full_desc,
                "tags": [t.strip("#") for t in (hashtags or "").split() if t.startswith("#")],
                "categoryId": "22"  # People & Blogs
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False
            }
        }

        # Step 1: Initiate resumable upload session
        init_url = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status"
        meta_bytes = json.dumps(snippet_metadata).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "video/mp4",
            "X-Upload-Content-Length": str(val["size_bytes"])
        }

        try:
            req = urllib.request.Request(init_url, data=meta_bytes, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                upload_location = resp.headers.get("Location")

            if not upload_location:
                return {
                    "success": False,
                    "status": "FAILED",
                    "platform": self.platform_key,
                    "error": "YouTube upload session did not return an upload Location URI"
                }

            # Step 2: Upload video payload
            with open(video_path, "rb") as vf:
                vbytes = vf.read()

            up_req = urllib.request.Request(
                upload_location,
                data=vbytes,
                headers={"Content-Type": "video/mp4"},
                method="PUT"
            )
            with urllib.request.urlopen(up_req, timeout=300.0) as up_resp:
                result = json.loads(up_resp.read().decode("utf-8"))
                video_id = result.get("id")

            return {
                "success": True,
                "status": "PUBLISHED",
                "platform": self.platform_key,
                "post_id": video_id,
                "post_url": f"https://www.youtube.com/shorts/{video_id}" if video_id else None,
                "error": None
            }
        except urllib.error.HTTPError as he:
            err = he.read().decode("utf-8", errors="ignore")
            logger.error(f"YouTube upload error {he.code}: {err}")
            return {
                "success": False,
                "status": "FAILED",
                "platform": self.platform_key,
                "error": f"API Error {he.code}: {err}"
            }
        except Exception as e:
            logger.error(f"YouTube upload failed: {e}")
            return {
                "success": False,
                "status": "FAILED",
                "platform": self.platform_key,
                "error": str(e)
            }
