import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.database import SessionLocal, init_db, ORIGINAL_DIR
from app.models import Video
from app.services.douyin_service import (
    extract_douyin_video_id,
    get_canonical_douyin_url,
    normalize_douyin_url
)
from app.services.downloader_service import DownloaderService, validate_downloaded_media

TEST_URL = (
    "https://www.douyin.com/jingxuan/search/%E5%AE%BF%E8%88%8D%E8%BF%B7%E4%BD%A0%E7%94%B5%E9%A5%AD%E7%85%B2"
    "?aid=63a8405e-afdb-4858-95f3-49e12593d228&modal_id=7661911073162791080&type=general"
)

def run_integration_test():
    print("=" * 70)
    print("TEST SNAPTIKTOK INTEGRATION IN AI CONTENT FACTORY")
    print("=" * 70)
    
    init_db()
    db = SessionLocal()
    downloader = DownloaderService()
    
    # 1. URL parsing test
    print("\n--- 1. Testing URL Parsing ---")
    modal_id = extract_douyin_video_id(TEST_URL)
    canonical = get_canonical_douyin_url(TEST_URL)
    norm_url = normalize_douyin_url(TEST_URL)
    
    print(f"Original URL : {TEST_URL}")
    print(f"Modal ID     : {modal_id}")
    print(f"Canonical URL: {canonical}")
    
    assert modal_id == "7661911073162791080", f"Expected 7661911073162791080, got {modal_id}"
    assert canonical == "https://www.douyin.com/video/7661911073162791080", f"Unexpected canonical: {canonical}"
    print("[PASS] URL parsing and canonical conversion verified.")

    # 2. Test direct download via DownloaderService
    print("\n--- 2. Testing Direct Download & ffprobe Verification ---")
    temp_target = ORIGINAL_DIR / "test_verify_douyin.mp4"
    if temp_target.exists():
        temp_target.unlink()

    dl_result = downloader.download_video(TEST_URL, str(temp_target))
    print("Download result:", dl_result)

    assert dl_result.get("success") is True, f"Expected download SUCCESS, got: {dl_result}"
    assert temp_target.exists(), "Target file was not created"
    assert temp_target.stat().st_size > 0, "Downloaded file is 0 bytes"
    
    # Verify with ffprobe
    is_valid, meta, err = validate_downloaded_media(temp_target)
    assert is_valid is True, f"ffprobe validation failed: {err}"
    assert meta["duration"] > 0, "Duration must be > 0"
    assert meta["width"] > 0 and meta["height"] > 0, "Resolution must be > 0"
    print(f"[PASS] File downloaded: {temp_target} ({meta['file_size']:,} bytes)")
    print(f"  Duration: {meta['duration']}s | Resolution: {meta['width']}x{meta['height']} | Video: {meta['video_codec']} | Audio: {meta['audio_codec']}")
    temp_target.unlink()

    # 3. Test DB Workflow (download_and_attach)
    print("\n--- 3. Testing DB Workflow Integration ---")
    # Clean previous record if exists
    db.query(Video).filter(Video.douyin_url == norm_url).delete()
    db.commit()

    db_res = downloader.download_and_attach(db, url=TEST_URL)
    print("DB attach result:", db_res)

    assert db_res.get("success") is True, f"Expected DB attach SUCCESS, got: {db_res}"
    vid_id = db_res["video_id"]
    assert vid_id.startswith("V"), f"Video ID must start with V: {vid_id}"

    saved_file = ORIGINAL_DIR / f"{vid_id}_original.mp4"
    assert saved_file.exists(), f"Saved file {saved_file} does not exist"
    assert saved_file.stat().st_size > 0, "Saved file is 0 bytes"

    # Verify DB record
    v_rec = db.query(Video).filter(Video.video_id == vid_id).first()
    assert v_rec is not None, "Video record not found in DB"
    assert v_rec.downloaded is True, "downloaded flag must be True"
    assert v_rec.status == "DOWNLOADED", f"Status must be DOWNLOADED, got {v_rec.status}"
    assert v_rec.local_file == f"original/{vid_id}_original.mp4"
    assert v_rec.douyin_url == norm_url, "Original URL must be preserved in DB"

    print(f"[PASS] DB workflow verified: Video {vid_id} saved as {v_rec.local_file} with status {v_rec.status}")

    # 4. Test Duplicate Protection
    print("\n--- 4. Testing Duplicate Protection ---")
    dup_res = downloader.download_and_attach(db, url=TEST_URL)
    assert dup_res["success"] is True
    assert dup_res["video_id"] == vid_id, f"Duplicate should return same video_id {vid_id}"
    print(f"[PASS] Duplicate URL handled gracefully, returned existing video {vid_id}.")

    db.close()
    print("\n" + "=" * 70)
    print("ALL SNAPTIKTOK INTEGRATION TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    run_integration_test()
