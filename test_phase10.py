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
from app.models import Product, Video, Content, Publishing

def test_phase10():
    print("=== TESTING PHASE 10: PUBLISHING QUEUE (6 PLATFORMS) ===")
    init_db()
    client = TestClient(app)
    db: Session = SessionLocal()

    test_vid = "VTEST_PUB_01"
    test_pid = "PTEST_PUB_01"

    # Setup Product, Video, Content, Publishing records
    db.query(Publishing).filter(Publishing.video_id == test_vid).delete()
    db.query(Content).filter(Content.video_id == test_vid).delete()
    db.query(Video).filter(Video.video_id == test_vid).delete()
    db.commit()

    prod = db.query(Product).filter(Product.product_id == test_pid).first()
    if not prod:
        prod = Product(product_id=test_pid, niche="Gia dụng", name_vietnamese="Sản phẩm Test Đăng Bài")
        db.add(prod)

    video = Video(
        video_id=test_vid,
        product_id=test_pid,
        douyin_url="https://v.douyin.com/pub_test/",
        status="CONTENT_READY"
    )
    content = Content(
        video_id=test_vid,
        facebook_page_caption="Caption FB page",
        facebook_page_hashtags="#fbpage",
        tiktok_caption="Caption TikTok",
        tiktok_hashtags="#tiktok",
        threads_caption="Caption Threads",
        threads_hashtags="#threads",
        instagram_caption="Caption Instagram",
        instagram_hashtags="#instagram",
        shopee_caption="Caption Shopee",
        shopee_hashtags="#shopee",
        youtube_title="Tiêu đề YouTube Shorts",
        youtube_description="Mô tả YouTube Shorts",
        youtube_hashtags="#Shorts",
        status="CONTENT_READY"
    )
    publishing = Publishing(
        video_id=test_vid,
        published_count=0,
        status="NOT PUBLISHED"
    )
    db.add_all([video, content, publishing])
    db.commit()
    print(f"  Initialized {test_vid} with Content and 0/6 Publishing record")

    # 1. Verify initial 0/6 -> NOT PUBLISHED
    print("\n1. Testing initial state: 0/6 = NOT PUBLISHED...")
    pub_0 = db.query(Publishing).filter(Publishing.video_id == test_vid).first()
    assert pub_0.published_count == 0, f"Expected count 0, got {pub_0.published_count}"
    assert pub_0.status == "NOT PUBLISHED", f"Expected NOT PUBLISHED, got {pub_0.status}"
    print("  [OK] 0/6 verified: NOT PUBLISHED")

    # 2. Toggle 1 platform -> 1/6 = PARTIAL
    print("\n2. Testing toggle 1 platform (TikTok) -> 1/6 = PARTIAL...")
    res_1 = client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "tiktok", "posted": True})
    assert res_1.status_code == 200, f"Toggle failed: {res_1.text}"
    data_1 = res_1.json()
    assert data_1["published_count"] == 1, f"Expected 1, got {data_1['published_count']}"
    assert data_1["status"] == "PARTIAL", f"Expected PARTIAL, got {data_1['status']}"
    assert data_1["video_status"] == "PARTIAL"
    print("  [OK] 1/6 verified: PARTIAL")

    # 3. Toggle 2 more platforms (Instagram, Threads) -> 3/6 = PARTIAL
    print("\n3. Testing toggle to 3 platforms -> 3/6 = PARTIAL...")
    client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "threads", "posted": True})
    res_3 = client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "instagram", "posted": True})
    assert res_3.status_code == 200
    data_3 = res_3.json()
    assert data_3["published_count"] == 3
    assert data_3["status"] == "PARTIAL"
    print("  [OK] 3/6 verified: PARTIAL")

    # 4. Toggle 2 more platforms (FB Page, Shopee) -> 5/6 = PARTIAL
    print("\n4. Testing toggle to 5 platforms -> 5/6 = PARTIAL...")
    client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "facebook_page", "posted": True})
    res_5 = client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "shopee", "posted": True})
    assert res_5.status_code == 200
    data_5 = res_5.json()
    assert data_5["published_count"] == 5
    assert data_5["status"] == "PARTIAL"
    print("  [OK] 5/6 verified: PARTIAL")

    # 5. Toggle the final 6th platform (YouTube) -> EXACTLY 6/6 = COMPLETED
    print("\n5. Testing toggle final 6th platform (YouTube) -> 6/6 = COMPLETED...")
    res_6 = client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "youtube", "posted": True})
    assert res_6.status_code == 200
    data_6 = res_6.json()
    assert data_6["published_count"] == 6, f"Expected 6, got {data_6['published_count']}"
    assert data_6["status"] == "COMPLETED", f"Expected COMPLETED, got {data_6['status']}"
    assert data_6["video_status"] == "COMPLETED"

    # Verify SQLite persistence
    db.expire_all()
    pub_db = db.query(Publishing).filter(Publishing.video_id == test_vid).first()
    v_db = db.query(Video).filter(Video.video_id == test_vid).first()
    assert pub_db.published_count == 6, f"Expected 6, got {pub_db.published_count}"
    assert pub_db.status == "COMPLETED"
    assert pub_db.publish_date is not None, "publish_date must be set upon 6/6 completion"
    assert v_db.status == "COMPLETED"
    print("  [OK] EXACTLY 6/6 -> COMPLETED with publish_date timestamp verified in SQLite")

    # 6. Test Un-toggling a platform (Revert 6/6 -> 5/6 PARTIAL)
    print("\n6. Testing untoggle 1 platform (Revert to 5/6 PARTIAL)...")
    res_revert = client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "youtube", "posted": False})
    data_rev = res_revert.json()
    assert data_rev["published_count"] == 5
    assert data_rev["status"] == "PARTIAL"
    assert data_rev["video_status"] == "PARTIAL"
    print("  [OK] Untoggle properly reverted state to 5/6 PARTIAL")

    # Re-toggle back to 6/6
    client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "youtube", "posted": True})

    # 7. Test Refresh Persistence (GET /publishing preserves checked state)
    print("\n7. Testing GET /publishing page rendering & persistence...")
    get_res = client.get("/publishing")
    assert get_res.status_code == 200
    assert test_vid in get_res.text
    assert "6/6" in get_res.text
    assert "COMPLETED" in get_res.text
    print("  [OK] GET /publishing correctly rendered video with 6/6 COMPLETED state preserved")

    db.close()
    print("\n=== PHASE 10 TESTS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_phase10()
