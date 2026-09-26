import sys
import time
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.database import SessionLocal, init_db, FINAL_DIR
from app.models import Product, Video, Voice
from app.services.folder_watcher import FinalFolderWatcher

def test_phase8():
    print("=== TESTING PHASE 8: FINAL FOLDER WATCHER (WATCHDOG) ===")
    init_db()
    client = TestClient(app)
    db: Session = SessionLocal()

    watcher = FinalFolderWatcher()

    # 1. Test Pattern Matching & Filename Validation
    print("\n1. Testing filename pattern matching & Video ID extraction...")
    assert watcher.is_valid_final_video("V0001.mp4") == "V0001"
    assert watcher.is_valid_final_video("v0123.mov") == "V0123"
    assert watcher.is_valid_final_video("V9999.MP4") == "V9999"
    # Unrelated files that should be ignored
    assert watcher.is_valid_final_video("temp_export.mp4") is None
    assert watcher.is_valid_final_video("V0001.mp4.crdownload") is None
    assert watcher.is_valid_final_video("notes.txt") is None
    assert watcher.is_valid_final_video(".DS_Store") is None
    print("  [OK] Video ID regex filtering verified: only valid Vxxxx.(mp4|mov|mkv|webm) accepted")

    # 2. Setup a target Video record in SQLite
    print("\n2. Setting up test Video record in WAITING_CAPCUT state...")
    test_vid = "VTEST_WATCH_01"
    db.query(Voice).filter(Voice.video_id == test_vid).delete()
    db.query(Video).filter(Video.video_id == test_vid).delete()
    db.commit()

    video = Video(
        video_id=test_vid,
        douyin_url="https://v.douyin.com/watch_test/",
        downloaded=True,
        status="WAITING_CAPCUT"
    )
    db.add(video)
    db.commit()
    print(f"  [OK] Video {test_vid} created with status WAITING_CAPCUT")

    # 3. Test File Stability & Event Processing
    print("\n3. Testing export file detection and status update to READY...")
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    sample_final_file = FINAL_DIR / f"{test_vid}.mp4"

    # Simulate CapCut export finish
    sample_final_file.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08freeFINAL_EXPORT_BYTES")

    # Process file event directly
    callback_called = []
    watcher.on_ready_callback = lambda vid: callback_called.append(vid)

    detected_id = watcher.handle_file_event(sample_final_file)
    assert detected_id == test_vid, f"Expected {test_vid}, got {detected_id}"
    assert len(callback_called) == 1 and callback_called[0] == test_vid, "Callback not triggered"

    # Verify SQLite record updated to READY
    v_updated = db.query(Video).filter(Video.video_id == test_vid).first()
    assert v_updated.status == "READY", f"Expected READY, got {v_updated.status}"
    assert f"{test_vid}.mp4" in v_updated.notes
    print(f"  [OK] Video {test_vid} successfully transitioned to READY. Notes: {v_updated.notes}")

    # 4. Test Duplicate Event Suppression
    print("\n4. Testing Duplicate Event Suppression (Idempotence)...")
    # Handling same file again should return None and not re-trigger
    second_run = watcher.handle_file_event(sample_final_file)
    assert second_run is None, "Duplicate event must be ignored"
    assert len(callback_called) == 1, "Callback should not be called again"
    print("  [OK] Duplicate file event successfully ignored without re-processing")

    # 5. Test Watcher Background Lifecycle (start & stop)
    print("\n5. Testing Watcher Background Lifecycle...")
    test_watcher_thread = FinalFolderWatcher()
    test_watcher_thread.start()
    assert test_watcher_thread.running is True, "Watcher should be running"
    time.sleep(0.5)
    test_watcher_thread.stop()
    assert test_watcher_thread.running is False, "Watcher should be stopped"
    print("  [OK] Background watchdog observer starts and stops cleanly")

    # 6. Test Settings API /api/settings/test-watcher
    print("\n6. Testing /api/settings/test-watcher API endpoint...")
    api_res = client.post("/api/settings/test-watcher")
    assert api_res.status_code == 200, f"API test watcher failed: {api_res.text}"
    data = api_res.json()
    assert data["success"] is True
    assert data["status"] == "WATCHING"
    print(f"  [OK] POST /api/settings/test-watcher returned 200: {data}")

    # Cleanup test file
    if sample_final_file.exists():
        sample_final_file.unlink()

    db.close()
    print("\n=== PHASE 8 TESTS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_phase8()
