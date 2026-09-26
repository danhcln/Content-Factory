import sys
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.database import SessionLocal, init_db
from app.models import Product, Video
from app.services.douyin_service import DouyinService, get_next_video_id, normalize_douyin_url

def test_phase3():
    print("=== TESTING PHASE 3: DOUYIN VIDEO RESEARCH ===")
    init_db()
    client = TestClient(app)
    db: Session = SessionLocal()
    douyin_service = DouyinService()

    # Clean existing test videos if any
    db.query(Video).filter(Video.douyin_url.like("%test_douyin%")).delete()
    db.commit()

    # Create a test product for association
    test_prod = db.query(Product).filter(Product.product_id == "PTEST01").first()
    if not test_prod:
        test_prod = Product(
            product_id="PTEST01",
            niche="TestNiche",
            name_vietnamese="Sản phẩm thử nghiệm",
            name_chinese="测试产品",
            douyin_keywords="测试关键词",
            status="RESEARCHED"
        )
        db.add(test_prod)
        db.commit()

    # 1. Test Video ID generation (V0001, V0002...)
    print("\n1. Testing sequential Video ID generation...")
    vid_id1 = get_next_video_id(db)
    assert vid_id1.startswith("V"), f"Video ID must start with V, got {vid_id1}"
    print(f"  [OK] First Video ID: {vid_id1}")

    # 2. Test Manual URL Import with Product association and Optional Views
    print("\n2. Testing Manual URL Import with optional views...")
    res1 = douyin_service.add_video(
        db=db,
        douyin_url="https://v.douyin.com/test_douyin_001/",
        product_id="PTEST01",
        views="88000"
    )
    assert res1["success"] is True, f"Failed to add video: {res1}"
    created_id1 = res1["video_id"]
    print(f"  [OK] Video created: {created_id1} with 88000 views")

    # 3. Test URL-only record (no views, no thumbnail)
    print("\n3. Testing URL-only record (Views = NULL, Thumbnail = NULL)...")
    res2 = douyin_service.add_video(
        db=db,
        douyin_url="https://v.douyin.com/test_douyin_002/",
        product_id="PTEST01",
        views=None,
        thumbnail=None
    )
    assert res2["success"] is True, f"Failed to add URL-only video: {res2}"
    created_id2 = res2["video_id"]
    v2_record = db.query(Video).filter(Video.video_id == created_id2).first()
    assert v2_record.views is None, "Views should be None"
    assert v2_record.thumbnail is None, "Thumbnail should be None"
    assert v2_record.status == "FOUND", f"Expected FOUND status, got {v2_record.status}"
    print(f"  [OK] URL-only video saved with status FOUND: {created_id2}")

    # 4. Test Duplicate URL rejection
    print("\n4. Testing Duplicate URL Rejection...")
    dup_res = douyin_service.add_video(
        db=db,
        douyin_url="https://v.douyin.com/test_douyin_001/",  # Same as first
        product_id="PTEST01",
        views="99999"
    )
    assert dup_res["success"] is False, "Duplicate URL must be rejected!"
    assert dup_res.get("duplicate") is True, "Duplicate flag should be True"
    assert "đã tồn tại" in dup_res["error"], f"Expected duplicate error message, got {dup_res['error']}"
    print(f"  [OK] Duplicate URL successfully prevented: {dup_res['error']}")

    # 5. Test Sorting with NULL views (NULLs must remain visible)
    print("\n5. Testing sorting with NULL views...")
    # Add a third video with 250000 views
    douyin_service.add_video(
        db=db,
        douyin_url="https://v.douyin.com/test_douyin_003/",
        product_id="PTEST01",
        views="250000"
    )
    # Add a fourth video with 15000 views
    douyin_service.add_video(
        db=db,
        douyin_url="https://v.douyin.com/test_douyin_004/",
        product_id="PTEST01",
        views="15000"
    )

    sorted_videos = douyin_service.get_sorted_videos(db, sort_by_views=True)
    all_urls = [v.douyin_url for v in sorted_videos if "test_douyin" in v.douyin_url]
    assert len(all_urls) == 4, f"All 4 test videos must remain visible, found {len(all_urls)}"
    # Video 2 has NULL views, it must still be in the list
    v2_in_list = any(v.douyin_url == "https://v.douyin.com/test_douyin_002/" for v in sorted_videos)
    assert v2_in_list, "Video without views must remain visible in sorted list!"
    print(f"  [OK] Sorted videos count: {len(all_urls)}. Non-view video preserved.")

    # 6. Test Review Queue display & endpoints via HTTP client
    print("\n6. Testing Review Queue display & API endpoints...")
    # GET /review
    get_res = client.get("/review")
    assert get_res.status_code == 200
    assert "Review Queue" in get_res.text or "Hàng Đợi Duyệt" in get_res.text
    print("  [OK] GET /review responded 200")

    # GET /review?sort=views
    sort_res = client.get("/review?sort=views")
    assert sort_res.status_code == 200
    print("  [OK] GET /review?sort=views responded 200")

    # POST /api/videos/manual-import via JSON API
    api_import_res = client.post(
        "/api/videos/manual-import",
        json={
            "douyin_url": "https://v.douyin.com/test_douyin_api_005/",
            "product_id": "PTEST01",
            "views": "50000"
        }
    )
    assert api_import_res.status_code == 200, f"API import failed: {api_import_res.text}"
    api_data = api_import_res.json()
    assert api_data["success"] is True
    print(f"  [OK] POST /api/videos/manual-import created {api_data['video_id']}")

    # 7. Test automated compliant discovery status (BLOCKED_EXTERNAL)
    print("\n7. Testing automated compliant discovery status...")
    disc_res = douyin_service.discover_videos_compliant("居家好物", limit=5)
    assert disc_res["status"] == "BLOCKED_EXTERNAL", f"Expected BLOCKED_EXTERNAL, got {disc_res}"
    print("  [OK] Compliant discovery marked BLOCKED_EXTERNAL safely without circumvention logic.")

    db.close()
    print("\n=== PHASE 3 TESTS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_phase3()
