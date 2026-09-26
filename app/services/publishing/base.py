import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional


class BasePublisherProvider(ABC):
    """
    Abstract base provider defining the standard contract for social media publishing.
    All platform providers must implement this interface.
    """
    platform_key: str = "base"
    platform_name: str = "Base Platform"
    requires_manual: bool = False

    @abstractmethod
    def get_status(self, db: Optional[Any] = None) -> Dict[str, Any]:
        """
        Return the current connection and readiness status:
        - READY: Credentials configured and validated
        - NOT_CONNECTED: Missing API credentials / authorization
        - MANUAL_REQUIRED: Platform has no official 3rd-party upload API for this account type
        - FAILED: Connection or authentication error
        """
        pass

    @abstractmethod
    def test_connection(self) -> Dict[str, Any]:
        """
        Perform an authentic API ping/validation check against the platform endpoint.
        Returns: {"success": bool, "status": str, "message": str, "account_info": Optional[Dict]}
        """
        pass

    def validate_media(self, video_path: Path) -> Dict[str, Any]:
        """
        Verify that target video file exists, is non-empty, and has an acceptable MP4 format.
        """
        path = Path(video_path)
        if not path.exists():
            return {"valid": False, "error": f"Video file not found: {path.name}"}
        if path.stat().st_size == 0:
            return {"valid": False, "error": f"Video file is empty: {path.name}"}
        if path.suffix.lower() not in [".mp4", ".mov"]:
            return {"valid": False, "error": f"Unsupported video container: {path.suffix}"}
        return {"valid": True, "size_bytes": path.stat().st_size}

    @abstractmethod
    def publish_video(
        self,
        video_path: Path,
        caption: str,
        hashtags: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute video publication using official platform API.
        Returns:
            {
                "success": bool,
                "status": str ("PUBLISHED", "FAILED", "MANUAL_REQUIRED"),
                "platform": str,
                "post_id": Optional[str],
                "post_url": Optional[str],
                "error": Optional[str]
            }
        """
        pass
