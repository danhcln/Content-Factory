import sys
import time
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.database import SessionLocal, init_db
from app.models import Product, Video, Voice, Content, Publishing
from app.services.gemini_service import GeminiService, get_next_product_id, get_model_name
from app.services.script_service import ScriptService
from app.services.content_service import ContentService


def run_real_gemini_tests():
    print("==========================================================")
    print("      EXECUTING REAL GEMINI 3.8 FLASH VERIFICATION       ")
    print("==========================================================")

    init_db()
    db = SessionLocal()
    gemini_svc = GeminiService()
    script_svc = ScriptService()
    content_svc = ContentService()

    active_model = get_model_name(db)
    print(f"\n[CONFIG] Active Model: {active_model}")
    assert active_model == "gemini-3.8-flash", f"Expected gemini-3.8-flash, got {active_model}"

    # ---------------------------------------------------------
    # 7. REAL TEST — MODEL CONNECTION
    # ---------------------------------------------------------
    print("\n--- TEST 7: REAL GEMINI CONNECTION ---")
    conn_res = gemini_svc.test_connection(db=db)
    print("Connection result:", conn_res)
    assert conn_res["success"] is True, f"Real Gemini connection failed: {conn_res.get('error')}"
    assert conn_res["model"] == "gemini-3.8-flash"
    print(">>> REAL GEMINI CONNECTION = PASS")

    # ---------------------------------------------------------
    # 8. REAL TEST — PRODUCT RESEARCH
    # ---------------------------------------------------------
    print("\n--- TEST 8: REAL PRODUCT RESEARCH & DB SAVE ---")
    time.sleep(2)
    niche = "đồ công nghệ"
    print(f"Calling Gemini generate_products with niche='{niche}', count=1...")
    prods = gemini_svc.generate_products(niche=niche, count=1, db=db)
    assert len(prods) >= 1, "Expected at least 1 product generated"
    p = prods[0]

    name_vn = p.get("name_vietnamese", "").strip()
    name_cn = p.get("name_chinese", "").strip()
    keywords = p.get("douyin_keywords", "").strip()
    angle = p.get("content_angle", "").strip()
    hook = p.get("hook", "").strip()

    print(f"Product Name (VN): {name_vn}")
    print(f"Product Name (CN): {name_cn}")
    print(f"Douyin Keywords : {keywords}")
    print(f"Content Angle    : {angle}")
    print(f"Hook             : {hook}")

    assert name_vn, "Missing Vietnamese product name"
    assert name_cn, "Missing Chinese product name"
    assert keywords, "Missing Chinese Douyin keywords"
    assert angle, "Missing content angle"
    assert hook, "Missing hook"
    print(">>> PRODUCT RESEARCH = PASS")

    # Save to SQLite
    pid = get_next_product_id(db)
    prod_rec = Product(
        product_id=pid,
        niche=niche,
        name_vietnamese=name_vn,
        name_chinese=name_cn,
        douyin_keywords=keywords,
        content_angle=angle,
        hook=hook,
        status="RESEARCHED"
    )
    db.add(prod_rec)
    db.commit()

    saved_prod = db.query(Product).filter(Product.product_id == pid).first()
    assert saved_prod is not None, "Product record was not saved in SQLite"
    print(f"Verified product saved in SQLite: {saved_prod.product_id}")
    print(">>> DATABASE SAVE = PASS")

    # ---------------------------------------------------------
    # 9. REAL TEST — SCRIPT GENERATION & VALIDATION
    # ---------------------------------------------------------
    print("\n--- TEST 9: REAL SCRIPT GENERATION & VALIDATION ---")
    time.sleep(2)
    # Create a safe test Video record tied to our test Product
    vid_id = f"VTEST_{int(time.time())}"
    video_rec = Video(
        video_id=vid_id,
        product_id=pid,
        douyin_url=f"https://www.douyin.com/video/test_{vid_id}",
        views=15000,
        status="DOWNLOADED"
    )
    db.add(video_rec)
    db.commit()

    print(f"Calling real script generation for Video {vid_id}...")
    script_res = script_svc.generate_script_for_video(db=db, video_id=vid_id)
    print("Script generation result status:", script_res.get("status"))
    assert script_res["success"] is True, f"Script generation failed: {script_res.get('error')}"

    script_text = script_res.get("script", "")
    print(f"Generated Script ({len(script_text)} chars):\n{script_text}\n")

    assert len(script_text) >= 40, f"Script too short: {len(script_text)} chars"
    assert "placeholder" not in script_text.lower(), "Script contains placeholder"
    assert script_res["status"] == "SCRIPT_READY"

    # Verify database persistence & strict Video ID association
    voice_rec = db.query(Voice).filter(Voice.video_id == vid_id).first()
    assert voice_rec is not None, "Voice record not found for video"
    assert voice_rec.script == script_text, "Script in DB does not match generated script"
    assert voice_rec.status == "SCRIPT_READY"
    print(f"Verified script in SQLite for {voice_rec.video_id}")
    print(">>> SCRIPT GENERATION = PASS")
    print(">>> SCRIPT VALIDATION = PASS")

    # ---------------------------------------------------------
    # 10. REAL TEST — 7-PLATFORM CONTENT & JSON VALIDATION
    # ---------------------------------------------------------
    print("\n--- TEST 10: REAL 7-PLATFORM CONTENT & JSON VALIDATION ---")
    time.sleep(2)
    print(f"Calling real 1-call 7-platform content generation for Video {vid_id}...")
    content_res = content_svc.generate_content_for_video(db=db, video_id=vid_id)
    assert content_res["success"] is True, f"7-Platform generation failed: {content_res.get('error')}"
    assert content_res["status"] == "CONTENT_READY"

    c_data = content_res["content"]
    required_platforms = [
        "facebook_personal", "facebook_page", "tiktok",
        "threads", "instagram", "shopee", "youtube"
    ]
    for plat in required_platforms:
        assert plat in c_data, f"Missing platform {plat} in response"
        print(f"  [{plat.upper()}]:")
        for k, v in c_data[plat].items():
            print(f"    {k}: {v[:80]}..." if len(str(v)) > 80 else f"    {k}: {v}")

    # Check YouTube has title, description, hashtags
    assert c_data["youtube"].get("title"), "Missing youtube title"
    assert c_data["youtube"].get("description"), "Missing youtube description"

    # Check Content record saved in SQLite
    content_rec = db.query(Content).filter(Content.video_id == vid_id).first()
    assert content_rec is not None, "Content record not found in SQLite"
    assert content_rec.status == "CONTENT_READY"
    assert content_rec.youtube_title == c_data["youtube"]["title"]

    # Check Publishing record initialized in SQLite
    pub_rec = db.query(Publishing).filter(Publishing.video_id == vid_id).first()
    assert pub_rec is not None, "Publishing record not found in SQLite"
    assert pub_rec.status == "NOT PUBLISHED"

    print(">>> 7-PLATFORM CONTENT = PASS")
    print(">>> JSON VALIDATION = PASS")

    # Cleanup test records
    print("\nCleaning up test records...")
    db.delete(pub_rec)
    db.delete(content_rec)
    db.delete(voice_rec)
    db.delete(video_rec)
    db.commit()
    db.close()

    print("\n==========================================================")
    print("    ALL REAL GEMINI 3.8 FLASH TESTS PASSED SUCCESSFULLY!  ")
    print("==========================================================")
    return True


if __name__ == "__main__":
    success = run_real_gemini_tests()
    if not success:
        sys.exit(1)
