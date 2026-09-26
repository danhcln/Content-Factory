import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.models import Video, Voice, Product
from app.database import PACKAGES_DIR, BASE_DIR

logger = logging.getLogger("app.services.package")


class CapCutPackageService:
    def __init__(self):
        self.packages_dir = PACKAGES_DIR
        self.packages_dir.mkdir(parents=True, exist_ok=True)

    def create_package(self, db: Session, video_id: str) -> Dict[str, Any]:
        """
        Bundle assets for manual CapCut editing:
        downloads/packages/{video_id}/
          ├── original.mp4
          ├── voice.wav
          ├── script.txt
          └── info.json
        """
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            return {"success": False, "error": f"Không tìm thấy video {video_id}"}

        voice = db.query(Voice).filter(Voice.video_id == video_id).first()
        script_text = voice.script if voice else ""

        target_dir = self.packages_dir / video_id
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Resolve and copy original.mp4
        original_dest = target_dir / "original.mp4"
        if not original_dest.exists() or original_dest.stat().st_size == 0:
            original_src = None
            if video.local_file:
                # check if path is relative or absolute
                candidate = BASE_DIR / "downloads" / video.local_file
                if candidate.exists():
                    original_src = candidate
                elif Path(video.local_file).exists():
                    original_src = Path(video.local_file)

            if not original_src or not original_src.exists():
                # Check fallback downloads/original/Vxxxx_original.mp4
                fallback = BASE_DIR / "downloads" / "original" / f"{video_id}_original.mp4"
                if fallback.exists():
                    original_src = fallback

            if original_src and original_src.exists():
                shutil.copy2(original_src, original_dest)
            else:
                return {
                    "success": False,
                    "error": f"Không tìm thấy video gốc để đóng gói cho {video_id}."
                }

        # 2. Resolve and copy/ensure voice.wav
        voice_dest = target_dir / "voice.wav"
        if not voice_dest.exists() or voice_dest.stat().st_size == 0:
            voice_src = None
            if voice and voice.audio_file:
                candidate = BASE_DIR / "downloads" / voice.audio_file
                if candidate.exists():
                    voice_src = candidate
                elif Path(voice.audio_file).exists():
                    voice_src = Path(voice.audio_file)

            if voice_src and voice_src.exists():
                shutil.copy2(voice_src, voice_dest)
            else:
                return {
                    "success": False,
                    "error": f"Không tìm thấy file giọng đọc voice.wav cho {video_id}."
                }

        # 3. Write script.txt (UTF-8)
        script_dest = target_dir / "script.txt"
        with open(script_dest, "w", encoding="utf-8") as f:
            f.write(script_text or "")

        # 4. Write info.json
        product = video.product
        info_data = {
            "video_id": video_id,
            "product_id": video.product_id or "",
            "product_name": product.name_vietnamese if product else "",
            "source_url": video.douyin_url or "",
            "script": script_text,
            "created_at": datetime.utcnow().isoformat()
        }
        info_dest = target_dir / "info.json"
        with open(info_dest, "w", encoding="utf-8") as f:
            json.dump(info_data, f, ensure_ascii=False, indent=2)

        # 5. Update Video status in DB
        video.status = "WAITING_CAPCUT"
        db.commit()
        db.refresh(video)

        logger.info(f"CapCut package created for {video_id} at {target_dir}. Status: WAITING_CAPCUT")
        return {
            "success": True,
            "video_id": video_id,
            "package_dir": str(target_dir),
            "status": "WAITING_CAPCUT",
            "files": {
                "original": str(original_dest),
                "voice": str(voice_dest),
                "script": str(script_dest),
                "info": str(info_dest)
            }
        }

    def open_package_folder(self, video_id: str) -> Dict[str, Any]:
        """Open the package folder in Windows File Explorer."""
        target_dir = self.packages_dir / video_id
        if not target_dir.exists():
            return {"success": False, "error": f"Thư mục gói CapCut {target_dir} chưa được tạo."}

        try:
            if hasattr(os, "startfile"):
                os.startfile(str(target_dir))
            else:
                subprocess.Popen(["explorer", str(target_dir)])
            return {"success": True, "message": f"Đã mở thư mục {video_id}"}
        except Exception as e:
            logger.error(f"Failed to open folder {target_dir}: {e}")
            return {"success": False, "error": str(e)}
