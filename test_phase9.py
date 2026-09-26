import json
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.database import SessionLocal, init_db
from app.models import Product, Video, Voice, Content, Publishing
from app.services.content_service import ContentService, validate_content_json
from app.services.gemini_service import get_api_key

def test_phase9():
    print("=== TESTING PHASE 9: AUTOMATIC 7-PLATFORM CAPTIONS ===")
    init_db()
    client = TestClient(app)
    db: Session = SessionLocal()
    content_service = ContentService()

    # 1. Test JSON Schema Validation
    print("\n1. Testing 7-platform JSON schema validation...")
    sample_valid_json = {
        "facebook_personal": {"caption": "Trải nghiệm chiếc nồi mini tiện lợi này sau 1 tuần dùng thử.", "hashtags": "#giadung #noimini"},
        "facebook_page": {"caption": "Nồi cơm mini đa năng - Giải pháp nấu ăn nhanh gọn cho người bận rộn.", "hashtags": "#noicommini #dodungbep"},
        "tiktok": {"caption": "Ai ở một mình nhất định phải sắm em nồi này nha!", "hashtags": "#tiktokmademebuyit #learnontiktok"},
        "threads": {"caption": "Một mình thì ăn gì cho nhanh mà không tốn công rửa bát?", "hashtags": "#lifestyle"},
        "instagram": {"caption": "Góc bếp nhỏ xinh cùng chiếc nồi đa năng.", "hashtags": "#homestyle #kitchendesign"},
        "shopee": {"caption": "Nồi cơm điện mini đa năng chống dính cao cấp.", "hashtags": "#shopeevideo #noicomdien"},
        "youtube": {"title": "Nồi cơm điện mini tiện lợi cho người sống một mình", "description": "Xem đánh giá chi tiết về chiếc nồi mini đa năng.", "hashtags": "#Shorts #Review"}
    }
    v_ok = validate_content_json(sample_valid_json)
    assert v_ok["valid"] is True, f"Valid JSON rejected: {v_ok}"
    print("  [OK] Valid 7-platform JSON passed schema validation")

    # Incomplete schema (missing youtube)
    incomplete_json = {k: v for k, v in sample_valid_json.items() if k != "youtube"}
    v_bad = validate_content_json(incomplete_json)
    assert v_bad["valid"] is False and "youtube" in v_bad["error"]
    print("  [OK] Incomplete JSON (missing platform) correctly rejected")

    # 2. Setup Test Data (Product, Video, Voice in READY state)
    print("\n2. Setting up test Video & Script...")
    test_vid = "VTEST_CONTENT_01"
    test_pid = "PTEST_CONTENT_01"

    db.query(Publishing).filter(Publishing.video_id == test_vid).delete()
    db.query(Content).filter(Content.video_id == test_vid).delete()
    db.query(Voice).filter(Voice.video_id == test_vid).delete()
    db.query(Video).filter(Video.video_id == test_vid).delete()
    db.query(Product).filter(Product.product_id == test_pid).delete()
    db.commit()

    prod = Product(
        product_id=test_pid,
        niche="Đồ gia dụng thông minh",
        name_vietnamese="Nồi cơm mini đa năng",
        status="RESEARCHED"
    )
    video = Video(
        video_id=test_vid,
        product_id=test_pid,
        douyin_url="https://v.douyin.com/content_test/",
        downloaded=True,
        status="READY"
    )
    voice = Voice(
        video_id=test_vid,
        script="Nồi cơm mini đa năng giúp nấu cơm, hấp trứng và luộc rau cùng lúc. Lòng nồi chống dính dễ vệ sinh.",
        status="VOICE_READY"
    )
    db.add_all([prod, video, voice])
    db.commit()

    # 3. Test Content Generation with Mocked 1-call Response
    print("\n3. Testing 1-Call 7-Platform Content Generation & Database Persistence...")
    gen_res = content_service.generate_content_for_video(
        db,
        test_vid,
        mocked_response=sample_valid_json
    )
    assert gen_res["success"] is True, f"Generation failed: {gen_res}"
    assert gen_res["status"] == "CONTENT_READY"

    # Verify CONTENT table row
    content_row = db.query(Content).filter(Content.video_id == test_vid).first()
    assert content_row is not None, "Content row must exist in SQLite"
    assert content_row.tiktok_caption == "Ai ở một mình nhất định phải sắm em nồi này nha!"
    assert content_row.tiktok_hashtags == "#tiktokmademebuyit #learnontiktok"
    assert content_row.facebook_personal_caption == "Trải nghiệm chiếc nồi mini tiện lợi này sau 1 tuần dùng thử."
    assert content_row.facebook_page_caption == "Nồi cơm mini đa năng - Giải pháp nấu ăn nhanh gọn cho người bận rộn."
    assert content_row.threads_caption == "Một mình thì ăn gì cho nhanh mà không tốn công rửa bát?"
    assert content_row.instagram_caption == "Góc bếp nhỏ xinh cùng chiếc nồi đa năng."
    assert content_row.shopee_caption == "Nồi cơm điện mini đa năng chống dính cao cấp."
    assert content_row.youtube_title == "Nồi cơm điện mini tiện lợi cho người sống một mình"
    assert content_row.status == "CONTENT_READY"
    print("  [OK] All 7 platforms accurately saved in CONTENT table")

    # Verify PUBLISHING row initialized
    pub_row = db.query(Publishing).filter(Publishing.video_id == test_vid).first()
    assert pub_row is not None, "Publishing row must be initialized"
    assert pub_row.published_count == 0
    assert pub_row.status == "NOT PUBLISHED"
    print("  [OK] Publishing queue record initialized: 0/7 (NOT PUBLISHED)")

    # Verify Video status updated to CONTENT_READY
    v_db = db.query(Video).filter(Video.video_id == test_vid).first()
    assert v_db.status == "CONTENT_READY"
    print("  [OK] Video status updated to CONTENT_READY")

    # 4. Test API endpoint POST /api/content/generate/{video_id}
    print("\n4. Testing /api/content/generate API endpoint...")
    with patch.object(ContentService, "generate_content_for_video", return_value={"success": True, "status": "CONTENT_READY"}):
        api_res = client.post(f"/api/content/generate/{test_vid}")
        assert api_res.status_code == 200, f"API failed: {api_res.text}"
        print("  [OK] POST /api/content/generate responded 200")

    # 5. Live test if API key is present
    live_key = get_api_key()
    if live_key:
        print("\n5. Live Gemini API Key found. Checking live integration test for captions...")
        try:
            live_res = content_service.generate_content_for_video(db, test_vid, max_retries=0)
            print("  Live caption gen result:", live_res.get("status"))
        except Exception as e:
            print("  Live caption gen result:", str(e)[:100])
    else:
        print("\n5. Live API Key not configured in .env.")
        print("  [NOTE] Live caption generation marked: BLOCKED_EXTERNAL (Expected until user enters API key)")

    db.close()
    print("\n=== PHASE 9 TESTS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_phase9()
