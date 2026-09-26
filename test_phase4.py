import sys
from pathlib import Path
import io

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.database import SessionLocal, init_db, ORIGINAL_DIR
from app.models import Product, Video
from app.services.douyin_service import DouyinService
from app.services.downloader_service import DownloaderService

def test_phase4():
    print("=== TESTING PHASE 4: REVIEW + VIDEO ACQUISITION ===")
    init_db()
    client = TestClient(app)
    db: Session = SessionLocal()
    douyin = DouyinService()
    downloader = DownloaderService()

    # Create test product & videos
    prod = db.query(Product).filter(Product.product_id == "PTEST_PHASE4").first()
    if not prod:
        prod = Product(
            product_id="PTEST_PHASE4",
            niche="Đồ gia dụng",
            name_vietnamese="Sản phẩm Phase 4",
            status="RESEARCHED"
        )
        db.add(prod)
        db.commit()

    # Clean previous test videos
    db.query(Video).filter(Video.product_id == "PTEST_PHASE4").delete()
    db.commit()

    # Add 3 test videos
    v1_res = douyin.add_video(db, "https://v.douyin.com/test_p4_v1/", product_id="PTEST_PHASE4")
    v2_res = douyin.add_video(db, "https://v.douyin.com/test_p4_v2/", product_id="PTEST_PHASE4")
    v3_res = douyin.add_video(db, "https://v.douyin.com/test_p4_v3/", product_id="PTEST_PHASE4")

    v1_id = v1_res["video_id"]
    v2_id = v2_res["video_id"]
    v3_id = v3_res["video_id"]
    print(f"  Created test videos: {v1_id}, {v2_id}, {v3_id} (All initial status: FOUND)")

    # 1. Test Single Approve
    print("\n1. Testing Single Approve via API...")
    appr_res = client.post("/api/videos/approve", json={"video_id": v1_id})
    assert appr_res.status_code == 200, f"Approve failed: {appr_res.text}"
    v1_rec = db.query(Video).filter(Video.video_id == v1_id).first()
    assert v1_rec.approved is True, "Video should be approved"
    assert v1_rec.status == "APPROVED", f"Status should be APPROVED, got {v1_rec.status}"
    print(f"  [OK] Single video {v1_id} approved -> status: APPROVED")

    # 2. Test Single Reject
    print("\n2. Testing Single Reject via API...")
    rej_res = client.post("/api/videos/reject", json={"video_id": v2_id})
    assert rej_res.status_code == 200, f"Reject failed: {rej_res.text}"
    v2_rec = db.query(Video).filter(Video.video_id == v2_id).first()
    assert v2_rec.status == "REJECTED", f"Status should be REJECTED, got {v2_rec.status}"
    print(f"  [OK] Single video {v2_id} rejected -> status: REJECTED")

    # 3. Test Bulk Approve
    print("\n3. Testing Bulk Approve via API...")
    # Add another video for bulk approve
    v4_res = douyin.add_video(db, "https://v.douyin.com/test_p4_v4/", product_id="PTEST_PHASE4")
    v4_id = v4_res["video_id"]
    bulk_res = client.post("/api/videos/bulk-approve", json={"video_ids": [v3_id, v4_id]})
    assert bulk_res.status_code == 200
    assert bulk_res.json()["approved_count"] == 2
    for vid in [v3_id, v4_id]:
        rec = db.query(Video).filter(Video.video_id == vid).first()
        assert rec.approved is True
        assert rec.status == "APPROVED"
    print(f"  [OK] Bulk approved {v3_id}, {v4_id} -> status: APPROVED")

    # 4. Test File Validation (Zero size, invalid extension, valid file)
    print("\n4. Testing File Validation...")
    # 4a. Zero-size file
    zero_attach = downloader.attach_local_video(db, v1_id, b"", "empty.mp4")
    assert zero_attach["success"] is False
    assert "0 byte" in zero_attach["error"]
    print("  [OK] Zero-byte file rejected")

    # 4b. Invalid extension
    bad_ext_attach = downloader.attach_local_video(db, v1_id, b"fake_content", "test.exe")
    assert bad_ext_attach["success"] is False
    assert "không được hỗ trợ" in bad_ext_attach["error"]
    print("  [OK] Unsupported file extension rejected")

    # 5. Test Local Video Import & Filename Generation (Vxxxx_original.mp4)
    print("\n5. Testing Local Video Import & Filename Generation...")
    sample_mp4_bytes = b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08free"  # Minimal dummy MP4 header
    attach_res = downloader.attach_local_video(db, v1_id, sample_mp4_bytes, "source_video.mp4")
    assert attach_res["success"] is True
    assert attach_res["status"] == "DOWNLOADED"
    expected_rel = f"original/{v1_id}_original.mp4"
    assert attach_res["local_file"] == expected_rel

    # Verify physical file on disk
    expected_disk_path = ORIGINAL_DIR / f"{v1_id}_original.mp4"
    assert expected_disk_path.exists(), f"Physical file missing at {expected_disk_path}"
    assert expected_disk_path.stat().st_size > 0, "Saved file is empty"
    print(f"  [OK] File saved successfully on disk: {expected_disk_path} ({expected_disk_path.stat().st_size} bytes)")

    # 6. Test Status Transition in Database
    print("\n6. Testing Database Record Status Transition...")
    v1_rec_updated = db.query(Video).filter(Video.video_id == v1_id).first()
    assert v1_rec_updated.downloaded is True, "Downloaded flag should be True"
    assert v1_rec_updated.status == "DOWNLOADED", f"Status should be DOWNLOADED, got {v1_rec_updated.status}"
    assert v1_rec_updated.local_file == expected_rel
    print(f"  [OK] Status transition verified: APPROVED -> DOWNLOADED for {v1_id}")

    # 7. Test attach endpoint via HTTP API
    print("\n7. Testing POST /api/videos/attach-local/{video_id} via API...")
    files = {"file": ("my_downloaded_video.mp4", io.BytesIO(sample_mp4_bytes), "video/mp4")}
    api_attach_res = client.post(f"/api/videos/attach-local/{v3_id}", files=files)
    assert api_attach_res.status_code == 200, f"API attach failed: {api_attach_res.text}"
    v3_rec = db.query(Video).filter(Video.video_id == v3_id).first()
    assert v3_rec.status == "DOWNLOADED"
    print(f"  [OK] POST /api/videos/attach-local/{v3_id} succeeded -> status: DOWNLOADED")

    db.close()
    print("\n=== PHASE 4 TESTS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_phase4()
