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
    print("=== TESTING PHASE 10: PUBLISHING QUEUE (7 PLATFORMS) ===")
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
        facebook_personal_caption="Caption FB cá nhân",
        facebook_personal_hashtags="#fbpersonal",
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
    print(f"  Initialized {test_vid} with Content and 0/7 Publishing record")

    # 1. Verify initial 0/7 -> NOT PUBLISHED
    print("\n1. Testing initial state: 0/7 = NOT PUBLISHED...")
    pub_0 = db.query(Publishing).filter(Publishing.video_id == test_vid).first()
    assert pub_0.published_count == 0, f"Expected count 0, got {pub_0.published_count}"
    assert pub_0.status == "NOT PUBLISHED", f"Expected NOT PUBLISHED, got {pub_0.status}"
    print("  [OK] 0/7 verified: NOT PUBLISHED")

    # 2. Toggle 1 platform -> 1/7 = PARTIAL
    print("\n2. Testing toggle 1 platform (TikTok) -> 1/7 = PARTIAL...")
    res_1 = client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "tiktok", "posted": True})
    assert res_1.status_code == 200, f"Toggle failed: {res_1.text}"
    data_1 = res_1.json()
    assert data_1["published_count"] == 1, f"Expected 1, got {data_1['published_count']}"
    assert data_1["status"] == "PARTIAL", f"Expected PARTIAL, got {data_1['status']}"
    assert data_1["video_status"] == "PARTIAL"
    print("  [OK] 1/7 verified: PARTIAL")

    # 3. Toggle 2 more platforms (FB Personal, Instagram) -> 3/7 = PARTIAL
    print("\n3. Testing toggle to 3 platforms -> 3/7 = PARTIAL...")
    client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "facebook_personal", "posted": True})
    res_3 = client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "instagram", "posted": True})
    assert res_3.status_code == 200
    data_3 = res_3.json()
    assert data_3["published_count"] == 3
    assert data_3["status"] == "PARTIAL"
    print("  [OK] 3/7 verified: PARTIAL")

    # 4. Toggle 3 more platforms (FB Page, Threads, Shopee) -> 6/7 = PARTIAL
    print("\n4. Testing toggle to 6 platforms -> 6/7 = PARTIAL...")
    client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "facebook_page", "posted": True})
    client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "threads", "posted": True})
    res_6 = client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "shopee", "posted": True})
    assert res_6.status_code == 200
    data_6 = res_6.json()
    assert data_6["published_count"] == 6
    assert data_6["status"] == "PARTIAL"
    print("  [OK] 6/7 verified: PARTIAL")

    # 5. Toggle the final 7th platform (YouTube) -> EXACTLY 7/7 = COMPLETED
    print("\n5. Testing toggle final 7th platform (YouTube) -> 7/7 = COMPLETED...")
    res_7 = client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "youtube", "posted": True})
    assert res_7.status_code == 200
    data_7 = res_7.json()
    assert data_7["published_count"] == 7, f"Expected 7, got {data_7['published_count']}"
    assert data_7["status"] == "COMPLETED", f"Expected COMPLETED, got {data_7['status']}"
    assert data_7["video_status"] == "COMPLETED"

    # Verify SQLite persistence
    db.expire_all()
    pub_db = db.query(Publishing).filter(Publishing.video_id == test_vid).first()
    v_db = db.query(Video).filter(Video.video_id == test_vid).first()
    assert pub_db.published_count == 7, f"Expected 7, got {pub_db.published_count}"
    assert pub_db.status == "COMPLETED"
    assert pub_db.publish_date is not None, "publish_date must be set upon 7/7 completion"
    assert v_db.status == "COMPLETED"
    print("  [OK] EXACTLY 7/7 -> COMPLETED with publish_date timestamp verified in SQLite")

    # 6. Test Un-toggling a platform (Revert 7/7 -> 6/7 PARTIAL)
    print("\n6. Testing untoggle 1 platform (Revert to 6/7 PARTIAL)...")
    res_revert = client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "youtube", "posted": False})
    data_rev = res_revert.json()
    assert data_rev["published_count"] == 6
    assert data_rev["status"] == "PARTIAL"
    assert data_rev["video_status"] == "PARTIAL"
    print("  [OK] Untoggle properly reverted state to 6/7 PARTIAL")

    # Re-toggle back to 7/7
    client.post(f"/api/publishing/toggle/{test_vid}", json={"platform": "youtube", "posted": True})

    # 7. Test Refresh Persistence (GET /publishing preserves checked state)
    print("\n7. Testing GET /publishing page rendering & persistence...")
    get_res = client.get("/publishing")
    assert get_res.status_code == 200
    assert test_vid in get_res.text
    assert "7/7" in get_res.text
    assert "COMPLETED" in get_res.text
    print("  [OK] GET /publishing correctly rendered video with 7/7 COMPLETED state preserved")

    db.close()
    print("\n=== PHASE 10 TESTS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_phase10()
