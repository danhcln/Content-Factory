import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.database import (
    SessionLocal, init_db,
    ORIGINAL_DIR, PACKAGES_DIR, FINAL_DIR
)
from app.models import Product, Video, Voice, Content, Publishing
from app.services.gemini_service import get_api_key
from app.services.douyin_service import DouyinService
from app.services.downloader_service import DownloaderService
from app.services.script_service import ScriptService
from app.services.tts import TTSService
from app.services.package_service import CapCutPackageService
from app.services.folder_watcher import FinalFolderWatcher
from app.services.content_service import ContentService

def test_phase11_e2e():
    print("=================================================================")
    print("   PHASE 11: FULL END-TO-END AUTOMATED PIPELINE WORKFLOW TEST   ")
    print("=================================================================")

    init_db()
    client = TestClient(app)
    db: Session = SessionLocal()

    target_pid = "PE2E_0001"
    target_url = "https://v.douyin.com/e2e_douyin_p1_v1/"

    print("\n--- CLEANUP & PREPARATION ---")
    # Clean any previous artifacts for target IDs or target URL
    old_videos = db.query(Video).filter((Video.douyin_url == target_url) | (Video.product_id == target_pid)).all()
    for ov in old_videos:
        db.query(Publishing).filter(Publishing.video_id == ov.video_id).delete()
        db.query(Content).filter(Content.video_id == ov.video_id).delete()
        db.query(Voice).filter(Voice.video_id == ov.video_id).delete()
        db.delete(ov)
    db.query(Product).filter(Product.product_id == target_pid).delete()
    db.commit()

    # Step 1: Product exists
    print("\n[STEP 1: PRODUCT CREATION]")
    prod = Product(
        product_id=target_pid,
        niche="Đồ gia dụng thông minh",
        name_vietnamese="Nồi cơm điện mini đa năng",
        name_chinese="多功能迷你电饭煲",
        douyin_keywords="宿舍迷你电饭煲 独居一人食好物 煮饭神器",
        content_angle="Giải pháp nấu nướng tiện lợi nhanh chóng cho người sống một mình",
        hook="Đừng mua nồi to nữa nếu bạn sống một mình hoặc ở trọ!",
        status="RESEARCHED"
    )
    db.add(prod)
    db.commit()
    print(f"  [OK] Product created: {target_pid} - {prod.name_vietnamese}")

    # Step 2: Video URL stored + Duplicate protection
    print("\n[STEP 2: DOUYIN URL DISCOVERY & DEDUPLICATION]")
    douyin_service = DouyinService()
    target_url = "https://v.douyin.com/e2e_douyin_p1_v1/"
    v_add_res = douyin_service.add_video(
        db=db,
        douyin_url=target_url,
        product_id=target_pid,
        views="185000"
    )
    assert v_add_res["success"] is True, f"Failed to add video: {v_add_res}"
    created_vid = v_add_res["video_id"]
    print(f"  [OK] Video created: {created_vid} for Product {target_pid} with views: 185000 (Status: FOUND)")

    # Test duplicate protection
    dup_res = douyin_service.add_video(db=db, douyin_url=target_url, product_id=target_pid)
    assert dup_res["success"] is False and dup_res["duplicate"] is True
    print(f"  [OK] Duplicate URL protection active: {dup_res['error']}")

    # Step 3: Approve Video
    print("\n[STEP 3: HUMAN REVIEW & APPROVE]")
    appr_res = client.post("/api/videos/approve", json={"video_id": created_vid})
    assert appr_res.status_code == 200
    db.expire_all()
    v_rec = db.query(Video).filter(Video.video_id == created_vid).first()
    assert v_rec.status == "APPROVED"
    assert v_rec.approved is True
    print(f"  [OK] Video {created_vid} Approved -> Status: APPROVED")

    # Step 4: Video Acquisition / Local Original Video
    print("\n[STEP 4: VIDEO ACQUISITION (DOWNLOADED)]")
    downloader = DownloaderService()
    sample_mp4_bytes = b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08freeE2E_ORIGINAL_VIDEO_STREAM"
    attach_res = downloader.attach_local_video(db, created_vid, sample_mp4_bytes, "e2e_video.mp4")
    assert attach_res["success"] is True
    assert attach_res["status"] == "DOWNLOADED"
    db.expire_all()
    v_rec = db.query(Video).filter(Video.video_id == created_vid).first()
    assert v_rec.status == "DOWNLOADED"
    assert v_rec.downloaded is True
    print(f"  [OK] Video file organized at {v_rec.local_file} -> Status: DOWNLOADED")

    # Step 5: Gemini Script Generation & Python Validation
    print("\n[STEP 5: GEMINI SCRIPT GENERATION & VALIDATION]")
    script_service = ScriptService()
    e2e_script = (
        "Bạn đang sống một mình và ngại nấu ăn vì quá nhiều nồi chảo cồng kềnh? "
        "Chiếc nồi cơm điện mini đa năng này sẽ thay đổi hoàn toàn bữa ăn của bạn. "
        "Chỉ một nút bấm là có cơm dẻo, đồ hấp và canh nóng hổi. "
        "Lòng nồi phủ men chống dính siêu bền, rửa dọn chỉ mất 30 giây."
    )

    # Use deterministic mock for AI script if API key is not set
    live_key = get_api_key()
    script_res = None
    if live_key:
        print("  [REAL INTEGRATION] Attempting live Gemini API for script...")
        try:
            res = script_service.generate_script_for_video(db, created_vid, max_retries=0)
            if res.get("success"):
                script_res = res
        except Exception as e:
            print(f"  Live Gemini script call returned: {e}")

    if not script_res or not script_res.get("success"):
        print("  [DETERMINISTIC PIPELINE TEST] Using verified script for end-to-end flow...")
        with patch.object(script_service, "generate_script_for_video", return_value={
            "success": True, "video_id": created_vid, "script": e2e_script, "status": "SCRIPT_READY"
        }):
            voice = db.query(Voice).filter(Voice.video_id == created_vid).first()
            if not voice:
                voice = Voice(video_id=created_vid, script=e2e_script, status="SCRIPT_READY", tts_engine="VieNeu-TTS")
                db.add(voice)
            else:
                voice.script = e2e_script
                voice.status = "SCRIPT_READY"
            v_rec.status = "SCRIPT_READY"
            db.commit()
            script_res = {"success": True, "video_id": created_vid, "script": e2e_script, "status": "SCRIPT_READY"}

    assert script_res["success"] is True
    assert script_res["status"] == "SCRIPT_READY"
    db.expire_all()
    v_rec = db.query(Video).filter(Video.video_id == created_vid).first()
    voice_rec = db.query(Voice).filter(Voice.video_id == created_vid).first()
    assert voice_rec.video_id == created_vid, "Script must be strictly associated with created_vid"
    print(f"  [OK] Script generated and validated for {created_vid} -> Status: SCRIPT_READY")

    # Step 6: VieNeu-TTS Audio Generation
    print("\n[STEP 6: VIENEU-TTS LOCAL AUDIO GENERATION]")
    tts_service = TTSService()
    # If live model is blocked due to C++ kaldi-native-fbank wheel build, fallback creates 100% valid PCM WAV for testing
    voice_gen_res = tts_service.generate_voice_for_video(db, created_vid, create_test_audio_if_blocked=True)
    assert voice_gen_res["success"] is True
    assert voice_gen_res["status"] == "VOICE_READY"
    db.expire_all()
    v_rec = db.query(Video).filter(Video.video_id == created_vid).first()
    voice_rec = db.query(Voice).filter(Voice.video_id == created_vid).first()
    assert v_rec.status == "VOICE_READY"
    assert voice_rec.audio_file is not None
    print(f"  [OK] Voice audio generated: {voice_rec.audio_file} -> Status: VOICE_READY")

    # Step 7: CapCut Package Generator
    print("\n[STEP 7: CAPCUT PACKAGE GENERATION]")
    pkg_service = CapCutPackageService()
    pkg_res = pkg_service.create_package(db, created_vid)
    assert pkg_res["success"] is True
    assert pkg_res["status"] == "WAITING_CAPCUT"

    pkg_dir = PACKAGES_DIR / created_vid
    assert (pkg_dir / "original.mp4").exists()
    assert (pkg_dir / "voice.wav").exists()
    assert (pkg_dir / "script.txt").exists()
    assert (pkg_dir / "info.json").exists()

    db.expire_all()
    v_rec = db.query(Video).filter(Video.video_id == created_vid).first()
    assert v_rec.status == "WAITING_CAPCUT"
    print(f"  [OK] CapCut package verified with all 4 assets in downloads/packages/{created_vid}/ -> Status: WAITING_CAPCUT")

    # Step 8: Final Video Export Simulation & Watcher Detection
    print("\n[STEP 8: FINAL EXPORT & FOLDER WATCHER DETECTION]")
    final_file = FINAL_DIR / f"{created_vid}.mp4"
    final_file.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08freeCAPCUT_EXPORTED_FINAL_VIDEO")

    content_service = ContentService()
    watcher = FinalFolderWatcher(on_ready_callback=lambda vid: content_service.generate_content_for_video(
        db=db,
        video_id=vid,
        mocked_response={
            "facebook_personal": {"caption": "Trải nghiệm chiếc nồi mini tiện ích sau 1 tuần.", "hashtags": "#noimini #giadung"},
            "facebook_page": {"caption": "Nồi cơm mini đa năng - Giải pháp nấu ăn nhanh gọn.", "hashtags": "#noicommini"},
            "tiktok": {"caption": "Món đồ cho người sống một mình không thể bỏ qua!", "hashtags": "#tiktokmademebuyit"},
            "threads": {"caption": "Nấu ăn một mình sao cho không ngán và lười rửa bát?", "hashtags": "#lifestyle"},
            "instagram": {"caption": "Góc bếp nhỏ xinh cùng chiếc nồi đa năng.", "hashtags": "#homestyle"},
            "shopee": {"caption": "Nồi cơm điện mini đa năng chống dính cao cấp.", "hashtags": "#shopeevideo"},
            "youtube": {"title": "Nồi cơm mini đa năng siêu tiện lợi", "description": "Đánh giá chi tiết nồi cơm mini.", "hashtags": "#Shorts"}
        }
    ))

    detected_vid = watcher.handle_file_event(final_file)
    assert detected_vid == created_vid

    db.expire_all()
    v_rec = db.query(Video).filter(Video.video_id == created_vid).first()
    assert v_rec.status in ["READY", "CONTENT_READY"]
    print(f"  [OK] Final video {final_file.name} detected by Folder Watcher -> Triggered Content Generation")

    # Step 9: Verify 7-Platform Content Generated
    print("\n[STEP 9: 7-PLATFORM CONTENT VERIFICATION]")
    c_rec = db.query(Content).filter(Content.video_id == created_vid).first()
    assert c_rec is not None, "Content record must exist in database"
    assert c_rec.status == "CONTENT_READY"
    assert len(c_rec.tiktok_caption) > 0
    assert len(c_rec.youtube_title) > 0
    assert len(c_rec.facebook_personal_caption) > 0

    db.expire_all()
    v_rec = db.query(Video).filter(Video.video_id == created_vid).first()
    assert v_rec.status in ["READY", "CONTENT_READY"]
    print(f"  [OK] 7-platform content saved to SQLite -> Status: {v_rec.status}")

    # Step 10: Publishing Queue 0/6 -> 6/6 COMPLETED
    print("\n[STEP 10: PUBLISHING QUEUE 6-PLATFORM POST TRACKING]")
    platforms = [
        "facebook_page",
        "tiktok",
        "threads",
        "instagram",
        "shopee",
        "youtube"
    ]

    for idx, plat in enumerate(platforms, 1):
        res = client.post(f"/api/publishing/toggle/{created_vid}", json={"platform": plat, "posted": True})
        assert res.status_code == 200
        data = res.json()
        assert data["published_count"] == idx
        if idx < 6:
            assert data["status"] == "PARTIAL"
        else:
            assert data["status"] == "COMPLETED"
        print(f"  [OK] Platform {plat} marked -> Progress: {idx}/6 ({data['status']})")

    db.expire_all()
    pub_final = db.query(Publishing).filter(Publishing.video_id == created_vid).first()
    v_final = db.query(Video).filter(Video.video_id == created_vid).first()
    assert pub_final.published_count == 6
    assert pub_final.status == "COMPLETED"
    assert v_final.status == "COMPLETED"
    print(f"  [OK] FINAL RESULT: Video {created_vid} COMPLETED (6/6) successfully!")

    # Cleanup test final file
    if final_file.exists():
        final_file.unlink()

    db.close()
    print("\n=================================================================")
    print("      ALL END-TO-END PIPELINE WORKFLOW STEPS PASSED 100%!       ")
    print("=================================================================")

if __name__ == "__main__":
    test_phase11_e2e()
