import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Set, Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent

from app.database import FINAL_DIR, SessionLocal
from app.models import Video

logger = logging.getLogger("app.services.watcher")

VIDEO_PATTERN = re.compile(r"^(V[0-9A-Za-z_-]+)\.(mp4|mov|mkv|webm)$", re.IGNORECASE)


class FinalVideoHandler(FileSystemEventHandler):
    def __init__(self, watcher: "FinalFolderWatcher"):
        super().__init__()
        self.watcher = watcher

    def on_created(self, event):
        if not event.is_directory:
            self.watcher.handle_file_event(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            dest = getattr(event, 'dest_path', None)
            if dest:
                self.watcher.handle_file_event(Path(dest))


class FinalFolderWatcher:
    def __init__(
        self,
        watch_dir: Optional[Path] = None,
        on_ready_callback: Optional[Callable[[str], None]] = None
    ):
        self.watch_dir = watch_dir or FINAL_DIR
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.on_ready_callback = on_ready_callback
        self.processed_files: Set[str] = set()
        self.observer: Optional[Observer] = None
        self.running = False

    def is_valid_final_video(self, filename: str) -> Optional[str]:
        """Check if filename matches Vxxxx.mp4 pattern and extract Video ID."""
        m = VIDEO_PATTERN.match(filename)
        if m:
            return m.group(1).upper()
        return None

    def wait_for_file_stability(self, file_path: Path, max_checks: int = 5, delay: float = 0.5) -> bool:
        """Wait until file exists, is non-zero in size, and size has stabilized."""
        if not file_path.exists():
            return False

        last_size = -1
        for _ in range(max_checks):
            try:
                current_size = file_path.stat().st_size
                if current_size > 0 and current_size == last_size:
                    return True
                last_size = current_size
            except (OSError, FileNotFoundError):
                pass
            time.sleep(delay)

        return file_path.exists() and file_path.stat().st_size > 0

    def handle_file_event(self, file_path: Path) -> Optional[str]:
        """Process an incoming file from downloads/final/."""
        filename = file_path.name
        video_id = self.is_valid_final_video(filename)
        if not video_id:
            # Ignore unrelated files
            return None

        # Prevent duplicate processing of the same file path
        abs_key = str(file_path.resolve())
        if abs_key in self.processed_files:
            return None

        # Wait until file finishes writing
        stable = self.wait_for_file_stability(file_path)
        if not stable:
            logger.warning(f"File {file_path} failed stability check or is empty.")
            return None

        self.processed_files.add(abs_key)

        # Update database record
        db = SessionLocal()
        try:
            video = db.query(Video).filter(Video.video_id == video_id).first()
            if not video:
                logger.warning(f"Final video detected for {video_id} but record not found in SQLite.")
                return None

            if video.auto_edit_status == "AUTO_EDIT_READY" or video.status == "AUTO_EDIT_READY":
                video.status = "AUTO_EDIT_READY"
            else:
                video.status = "READY"
            video.final_video_path = f"downloads/final/{filename}"
            video.notes = f"Final video detected: {filename}"
            db.commit()
            logger.info(f"Video {video_id} updated to READY after detecting final export.")

            # Trigger content generation callback if provided
            if self.on_ready_callback:
                try:
                    self.on_ready_callback(video_id)
                except Exception as cb_err:
                    logger.error(f"Error in on_ready_callback for {video_id}: {cb_err}")

            return video_id
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update video record for {video_id}: {e}")
            return None
        finally:
            db.close()

    def start(self):
        """Start the background watchdog observer thread."""
        if self.running:
            return
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        handler = FinalVideoHandler(self)
        self.observer = Observer()
        self.observer.schedule(handler, str(self.watch_dir), recursive=False)
        self.observer.start()
        self.running = True
        logger.info(f"FinalFolderWatcher started monitoring: {self.watch_dir}")

    def stop(self):
        """Stop the background watchdog observer cleanly."""
        if self.observer and self.running:
            self.observer.stop()
            self.observer.join(timeout=3)
            self.running = False
            logger.info("FinalFolderWatcher stopped cleanly.")
