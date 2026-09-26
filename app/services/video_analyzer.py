import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from app.services.ffmpeg_utils import run_ffprobe, run_ffmpeg

logger = logging.getLogger("app.services.analyzer")


class VideoAnalyzer:
    """Extracts and validates video stream and container metadata using ffprobe."""

    @staticmethod
    def parse_fps(rate_str: Optional[str]) -> float:
        """Parse frame rate string like '30/1' or '29.97' into float."""
        if not rate_str or rate_str == "0/0":
            return 30.0
        if "/" in rate_str:
            num, den = rate_str.split("/", 1)
            try:
                den_f = float(den)
                return float(num) / den_f if den_f != 0 else 30.0
            except ValueError:
                return 30.0
        try:
            return float(rate_str)
        except ValueError:
            return 30.0

    @classmethod
    def analyze_video(cls, video_path: Path) -> Dict[str, Any]:
        """
        Extract video metadata and validate integrity.
        Rejects:
        - Missing file
        - Zero-byte file
        - Corrupted / unreadable media
        - Missing video stream
        """
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            return {
                "valid": False,
                "error": f"Tệp video không tồn tại: {video_path.name}",
                "status": "INVALID_SOURCE_VIDEO"
            }

        file_size = video_path.stat().st_size
        if file_size == 0:
            return {
                "valid": False,
                "error": f"Tệp video rỗng (0 bytes): {video_path.name}",
                "status": "INVALID_SOURCE_VIDEO"
            }

        args = [
            "-v", "error",
            "-show_format",
            "-show_streams",
            "-of", "json",
            str(video_path)
        ]

        try:
            proc = run_ffprobe(args, timeout=20.0)
            if proc.returncode != 0:
                return {
                    "valid": False,
                    "error": f"ffprobe không thể đọc tệp video: {proc.stderr[:200]}",
                    "status": "INVALID_SOURCE_VIDEO"
                }

            data = json.loads(proc.stdout)
        except Exception as e:
            return {
                "valid": False,
                "error": f"Lỗi phân tích video: {str(e)}",
                "status": "INVALID_SOURCE_VIDEO"
            }

        streams = data.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        if not video_stream:
            return {
                "valid": False,
                "error": f"Tệp video không chứa luồng hình ảnh (video stream): {video_path.name}",
                "status": "INVALID_SOURCE_VIDEO"
            }

        # Extract dimensions
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        if width <= 0 or height <= 0:
            return {
                "valid": False,
                "error": f"Kích thước video không hợp lệ ({width}x{height})",
                "status": "INVALID_SOURCE_VIDEO"
            }

        # Extract duration
        duration = 0.0
        fmt = data.get("format", {})
        if "duration" in fmt:
            try:
                duration = float(fmt["duration"])
            except ValueError:
                pass
        if duration <= 0 and "duration" in video_stream:
            try:
                duration = float(video_stream["duration"])
            except ValueError:
                pass

        if duration <= 0.0:
            return {
                "valid": False,
                "error": f"Thời lượng video không hợp lệ ({duration}s)",
                "status": "INVALID_SOURCE_VIDEO"
            }

        # Calculate FPS
        fps = cls.parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))

        # Aspect ratio determination
        aspect_ratio = "9:16" if height > width else ("16:9" if width > height else "1:1")

        # Audio stream info
        has_audio = audio_stream is not None
        audio_codec = audio_stream.get("codec_name") if audio_stream else None
        audio_duration = None
        if audio_stream and "duration" in audio_stream:
            try:
                audio_duration = float(audio_stream["duration"])
            except ValueError:
                pass

        return {
            "valid": True,
            "video_path": str(video_path),
            "filename": video_path.name,
            "file_size": file_size,
            "duration": round(duration, 2),
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "fps": round(fps, 2),
            "has_video": True,
            "video_codec": video_stream.get("codec_name", "unknown"),
            "has_audio": has_audio,
            "audio_codec": audio_codec,
            "audio_duration": round(audio_duration, 2) if audio_duration else None
        }

    @classmethod
    def extract_frame_preview(
        cls,
        video_path: Path,
        output_image_path: Path,
        timestamp_sec: float = 1.0
    ) -> bool:
        """
        Extract a representative video frame for subtitle region verification.
        Uses fast seek and high quality JPG extraction.
        """
        video_path = Path(video_path).resolve()
        output_image_path = Path(output_image_path).resolve()
        output_image_path.parent.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            return False

        # Build FFmpeg command for single frame capture
        args = [
            "-y",
            "-ss", f"{timestamp_sec:.2f}",
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(output_image_path)
        ]

        try:
            proc = run_ffmpeg(args, timeout=15.0)
            if proc.returncode == 0 and output_image_path.exists() and output_image_path.stat().st_size > 0:
                logger.info(f"Extracted frame preview at {timestamp_sec:.1f}s to {output_image_path}")
                return True

            # Fallback to timestamp 0.0s if timestamp_sec failed
            if timestamp_sec > 0.0:
                args[2] = "0.0"
                proc = run_ffmpeg(args, timeout=15.0)
                return proc.returncode == 0 and output_image_path.exists() and output_image_path.stat().st_size > 0

            return False
        except Exception as e:
            logger.error(f"Failed to extract frame preview: {e}")
            return False
