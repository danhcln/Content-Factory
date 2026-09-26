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
from app.models import Product, Video, Voice
from app.services.script_service import ScriptService
from app.services.gemini_service import GeminiService

def test_phase5():
    print("=== TESTING PHASE 5: SCRIPT GENERATION & VALIDATION ===")
    init_db()
    client = TestClient(app)
    db: Session = SessionLocal()
    script_service = ScriptService()

    # 1. Test Deterministic Python Script Validator
    print("\n1. Testing deterministic Python script validator...")
    # Empty
    v_empty = script_service.validate_script("")
    assert v_empty["valid"] is False and v_empty["status"] == "SCRIPT_REVIEW_REQUIRED"
    print("  [OK] Empty script rejected")

    # Too short (< 40 chars)
    v_short = script_service.validate_script("Ngắn quá bạn ơi!")
    assert v_short["valid"] is False and "quá ngắn" in v_short["reason"]
    print("  [OK] Short script rejected")

    # Placeholder rejection
    v_place = script_service.validate_script("Đây là [placeholder] cho sản phẩm rất tuyệt vời và hữu ích trong cuộc sống hàng ngày.")
    assert v_place["valid"] is False and "placeholder" in v_place["reason"]
    print("  [OK] Placeholder script rejected")

    # Raw JSON malformed rejection
    v_json = script_service.validate_script('{"script": "Lời bình hay nhất thế giới cho sản phẩm tiện ích nhà bếp gia đình."}')
    assert v_json["valid"] is False and v_json["status"] == "SCRIPT_ERROR"
    print("  [OK] Raw JSON block rejected")

    # Valid script
    sample_valid_script = (
        "Bạn có bao giờ gặp khó khăn khi phải nấu ăn một mình trong căn phòng trọ chật hẹp? "
        "Chiếc nồi cơm điện mini đa năng này chính là vị cứu tinh của bạn. "
        "Vừa nấu cơm, vừa hấp trứng lại có thể hầm canh nhanh chóng chỉ trong một nút bấm. "
        "Thiết kế nhỏ gọn, chống dính cao cấp dễ dàng vệ sinh sau khi dùng."
    )
    v_ok = script_service.validate_script(sample_valid_script)
    assert v_ok["valid"] is True and v_ok["status"] == "SCRIPT_READY"
    print("  [OK] Valid Vietnamese script passed with SCRIPT_READY")

    # 2. Setup 2 distinct Videos for Strict Video ID Association Test
    print("\n2. Setting up 2 distinct videos for association test...")
    prod1 = db.query(Product).filter(Product.product_id == "PTEST_P5_1").first()
    if not prod1:
        prod1 = Product(product_id="PTEST_P5_1", niche="Gia dụng", name_vietnamese="Nồi mini V1", status="RESEARCHED")
        db.add(prod1)
    prod2 = db.query(Product).filter(Product.product_id == "PTEST_P5_2").first()
    if not prod2:
        prod2 = Product(product_id="PTEST_P5_2", niche="Gia dụng", name_vietnamese="Quạt mini V2", status="RESEARCHED")
        db.add(prod2)
    db.commit()

    # Clean existing
    db.query(Voice).filter(Voice.video_id.in_(["VTEST_001", "VTEST_002"])).delete()
    db.query(Video).filter(Video.video_id.in_(["VTEST_001", "VTEST_002"])).delete()
    db.commit()

    video1 = Video(video_id="VTEST_001", product_id="PTEST_P5_1", douyin_url="https://v.douyin.com/p5_v1/", downloaded=True, status="DOWNLOADED")
    video2 = Video(video_id="VTEST_002", product_id="PTEST_P5_2", douyin_url="https://v.douyin.com/p5_v2/", downloaded=True, status="DOWNLOADED")
    db.add_all([video1, video2])
    db.commit()

    # 3. Test Script Generation & Strict Association
    print("\n3. Testing Script Generation with Mocked Gemini & Strict Association...")
    script_v1_text = (
        "Kịch bản dành riêng cho Nồi mini V1. Giúp nấu cơm dẻo thơm và giữ ấm cả ngày dài. "
        "Rất thích hợp cho các bạn học sinh sinh viên sống tự lập một mình."
    )
    script_v2_text = (
        "Kịch bản dành riêng cho Quạt mini V2. Mùa hè nóng bức mà có chiếc quạt tích điện pin trâu này "
        "thì đi học đi làm cả ngày không lo đổ mồ hôi. Gió cực êm và mát lạnh."
    )

    with patch("app.services.script_service.get_api_key", return_value="MOCK_KEY"):
        # Generate for VTEST_001
        with patch.object(GeminiService, "test_connection", return_value={"success": True}):
            with patch.object(GeminiService, "call_gemini", side_effect=[script_v1_text, script_v2_text]):
                res_v1 = script_service.generate_script_for_video(db, "VTEST_001")
                assert res_v1["success"] is True
                assert res_v1["video_id"] == "VTEST_001"
                assert res_v1["status"] == "SCRIPT_READY"

                # Generate for VTEST_002
                res_v2 = script_service.generate_script_for_video(db, "VTEST_002")
                assert res_v2["success"] is True
                assert res_v2["video_id"] == "VTEST_002"

    # CRITICAL CHECK: Verify strict database association
    voice1 = db.query(Voice).filter(Voice.video_id == "VTEST_001").first()
    voice2 = db.query(Voice).filter(Voice.video_id == "VTEST_002").first()

    assert voice1 is not None, "Voice record for VTEST_001 must exist"
    assert voice2 is not None, "Voice record for VTEST_002 must exist"
    assert "Nồi mini V1" in voice1.script, "VTEST_001 got wrong script!"
    assert "Quạt mini V2" in voice2.script, "VTEST_002 got wrong script!"
    assert voice1.script != voice2.script, "Scripts must be distinct!"
    assert "Quạt mini" not in voice1.script, "LEAK DETECTED: VTEST_002 content leaked into VTEST_001!"
    print("  [OK] CRITICAL: Strict video/script association verified without cross-contamination")

    # 4. Test Script Editing & Re-validation
    print("\n4. Testing Script Editing & Re-validation...")
    # 4a. Edit with invalid text
    edit_bad = script_service.edit_script(db, "VTEST_001", "Quá ngắn")
    assert edit_bad["success"] is False
    assert edit_bad["status"] == "SCRIPT_REVIEW_REQUIRED"
    v1_db = db.query(Video).filter(Video.video_id == "VTEST_001").first()
    assert v1_db.status == "SCRIPT_REVIEW_REQUIRED"
    print("  [OK] Invalid edit updated status to SCRIPT_REVIEW_REQUIRED")

    # 4b. Edit with valid text
    edit_good = script_service.edit_script(db, "VTEST_001", sample_valid_script)
    assert edit_good["success"] is True
    assert edit_good["status"] == "SCRIPT_READY"
    v1_db = db.query(Video).filter(Video.video_id == "VTEST_001").first()
    assert v1_db.status == "SCRIPT_READY"
    print("  [OK] Valid edit restored status to SCRIPT_READY")

    # 5. Test API endpoints
    print("\n5. Testing Voice API endpoints (/api/voice/edit, /api/voice/generate)...")
    api_edit_res = client.post(
        "/api/voice/edit/VTEST_001",
        json={"script": sample_valid_script}
    )
    assert api_edit_res.status_code == 200, f"API edit failed: {api_edit_res.text}"
    print("  [OK] POST /api/voice/edit/VTEST_001 succeeded")

    db.close()
    print("\n=== PHASE 5 TESTS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_phase5()
