import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.database import SessionLocal, init_db
from app.models import Product
from app.services.gemini_service import (
    GeminiService,
    clean_json_response,
    get_next_product_id,
    get_api_key
)

def test_phase2():
    print("=== TESTING PHASE 2: GEMINI RESEARCH ===")
    init_db()
    client = TestClient(app)
    db: Session = SessionLocal()

    # 1. Test clean_json_response with various markdown artifacts
    print("\n1. Testing JSON response cleaning...")
    sample_raw = """```json
    [
        {"name_vietnamese": "Nồi chiên", "name_chinese": "空气炸锅", "douyin_keywords": "好物", "content_angle": "Góc", "hook": "Hook"}
    ]
    ```"""
    cleaned = clean_json_response(sample_raw)
    parsed = json.loads(cleaned)
    assert isinstance(parsed, list), "Parsed output should be a list"
    assert parsed[0]["name_chinese"] == "空气炸锅"
    print("  [OK] clean_json_response successfully handled markdown fenced blocks")

    # 2. Test sequential product ID generation (P0001, P0002...)
    print("\n2. Testing sequential Product ID generation...")
    # Clean previous test products if needed
    db.query(Product).filter(Product.niche == "TestNichePhase2").delete()
    db.commit()

    initial_id = get_next_product_id(db)
    assert initial_id.startswith("P"), f"ID must start with P, got {initial_id}"
    print(f"  [OK] Current next product ID: {initial_id}")

    # Add a temporary test product to test increment
    p1 = Product(
        product_id=initial_id,
        niche="TestNichePhase2",
        name_vietnamese="Test SP 1",
        name_chinese="测试产品1",
        douyin_keywords="测试",
        status="NEW"
    )
    db.add(p1)
    db.commit()

    next_id = get_next_product_id(db)
    initial_num = int(initial_id[1:])
    next_num = int(next_id[1:])
    assert next_num == initial_num + 1, f"Expected {initial_num + 1}, got {next_num}"
    print(f"  [OK] Sequential generation verified: {initial_id} -> {next_id}")

    # Cleanup test product
    db.delete(p1)
    db.commit()

    # 3. Test missing-key handling (no crash, returns error dict or raises ValueError)
    print("\n3. Testing missing API key handling...")
    with patch("app.services.gemini_service.get_api_key", return_value=""):
        service = GeminiService()
        status_res = service.test_connection()
        assert status_res["success"] is False, "Should fail gracefully when API key is missing"
        assert "not configured" in status_res["error"] or "missing" in status_res["error"].lower()
        print("  [OK] Missing API key handled gracefully in test_connection:", status_res)

        try:
            service.generate_products("Đồ gia dụng", 5)
            assert False, "Should have raised ValueError for missing API key"
        except ValueError as ve:
            print("  [OK] generate_products raised ValueError when API key is missing:", ve)

    # 4. Test malformed Gemini response handling
    print("\n4. Testing malformed Gemini response handling...")
    with patch("app.services.gemini_service.get_api_key", return_value="FAKE_KEY_FOR_MOCK"):
        with patch.object(GeminiService, "generate_products", side_effect=ValueError("Malformed AI response format: Expecting value")):
            res = client.post("/research", data={"niche": "Đồ gia dụng", "product_count": 5})
            assert res.status_code == 400 or res.status_code == 200, f"Unexpected code {res.status_code}"
            assert "Malformed AI response" in res.text or "Error" in res.text or "Lỗi" in res.text
            print("  [OK] Malformed response handled without crashing app")

    # 5. Test deterministic mocked Gemini product generation & database persistence
    print("\n5. Testing product generation & persistence via mocked Gemini...")
    mock_products = [
        {
            "name_vietnamese": "Máy hút bụi giường nệm UV",
            "name_chinese": "除螨仪家用小型吸尘器",
            "douyin_keywords": "除螨仪实测 居家好物 开箱测评",
            "content_angle": "Giải pháp bảo vệ làn da khỏi bụi mịn và mạt rệp",
            "hook": "Nếu bạn hay bị ngứa ngáy nổi mụn, xem ngay chiếc máy này!"
        },
        {
            "name_vietnamese": "Bình giữ nhiệt hiển thị nhiệt độ",
            "name_chinese": "智能保温杯显温水杯",
            "douyin_keywords": "智能显温保温杯 学生党便携水杯 办公好物",
            "content_angle": "Bình nước thông minh cho dân văn phòng và học sinh",
            "hook": "Chiếc bình nước giúp bạn không bao giờ bị bỏng miệng!"
        }
    ]

    with patch("app.services.gemini_service.get_api_key", return_value="MOCK_API_KEY"):
        with patch.object(GeminiService, "generate_products", return_value=mock_products):
            # Test POST /research
            post_res = client.post(
                "/research",
                data={"niche": "Đồ gia dụng thông minh", "product_count": 2},
                follow_redirects=True
            )
            assert post_res.status_code == 200, f"Expected 200 after redirect, got {post_res.status_code}"
            print("  [OK] POST /research executed successfully")

            # Verify records exist in SQLite
            saved_prods = db.query(Product).filter(Product.niche == "Đồ gia dụng thông minh").all()
            assert len(saved_prods) >= 2, f"Expected at least 2 products saved, found {len(saved_prods)}"
            print(f"  [OK] Verified {len(saved_prods)} products saved in SQLite database")
            for p in saved_prods[-2:]:
                assert p.product_id.startswith("P"), f"Invalid product ID {p.product_id}"
                assert len(p.name_chinese) > 0, "Chinese name missing"
                assert len(p.douyin_keywords) > 0, "Douyin keywords missing"
                print(f"    - [{p.product_id}] {p.name_vietnamese} | {p.name_chinese} | Keywords: {p.douyin_keywords}")

    # 6. Test Settings test-gemini endpoint
    print("\n6. Testing /api/settings/test-gemini endpoint...")
    api_test_res = client.post("/api/settings/test-gemini")
    assert api_test_res.status_code == 200, f"Expected 200, got {api_test_res.status_code}"
    test_json = api_test_res.json()
    print("  [OK] /api/settings/test-gemini response:", test_json)

    # 7. Check live API key state
    live_key = get_api_key()
    if live_key:
        print("\n7. Live Gemini API Key found. Performing live integration test...")
        service = GeminiService()
        conn = service.test_connection()
        print("  Live test result:", conn)
    else:
        print("\n7. Live Gemini API Key not configured in .env.")
        print("  [NOTE] Live external Gemini test marked: BLOCKED_EXTERNAL (Expected until user enters API key in Settings)")

    db.close()
    print("\n=== PHASE 2 TESTS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_phase2()
