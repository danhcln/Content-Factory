import json
import sys
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.database import SessionLocal, init_db, PACKAGES_DIR, ORIGINAL_DIR
from app.models import Product, Video, Voice
from app.services.package_service import CapCutPackageService
from app.services.tts.vieneu_provider import create_pcm_wav_file

def test_phase7():
    print("=== TESTING PHASE 7: CAPCUT PACKAGE GENERATOR ===")
    init_db()
    client = TestClient(app)
    db: Session = SessionLocal()
    pkg_service = CapCutPackageService()

    # 1. Setup test Product, Video, Voice assets
    print("\n1. Setting up test assets (Product, Video, Voice)...")
    test_vid = "VTEST_PKG_01"
    test_pid = "PTEST_PKG_01"

    db.query(Voice).filter(Voice.video_id == test_vid).delete()
    db.query(Video).filter(Video.video_id == test_vid).delete()
    db.commit()

    prod = db.query(Product).filter(Product.product_id == test_pid).first()
    if not prod:
        prod = Product(
            product_id=test_pid,
            niche="Đồ gia dụng thông minh",
            name_vietnamese="Máy cắt rau củ quả đa năng",
            status="RESEARCHED"
        )
        db.add(prod)
        db.commit()

    # Prepare dummy original.mp4 in downloads/original/
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    sample_mp4 = ORIGINAL_DIR / f"{test_vid}_original.mp4"
    sample_mp4.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08freeDATA")

    # Create dummy voice.wav
    pkg_dir = PACKAGES_DIR / test_vid
    pkg_dir.mkdir(parents=True, exist_ok=True)
    sample_wav = pkg_dir / "voice.wav"
    create_pcm_wav_file(sample_wav, duration_seconds=1.0)

    sample_script = "Kịch bản mẫu cho máy cắt rau củ đa năng giúp việc nấu ăn trở nên cực kỳ nhanh chóng và tiện lợi."

    video = Video(
        video_id=test_vid,
        product_id=test_pid,
        douyin_url="https://v.douyin.com/pkg_test_url/",
        local_file=f"original/{test_vid}_original.mp4",
        downloaded=True,
        status="VOICE_READY"
    )
    voice = Voice(
        video_id=test_vid,
        script=sample_script,
        audio_file=f"packages/{test_vid}/voice.wav",
        tts_engine="VieNeu-TTS",
        status="VOICE_READY"
    )
    db.add_all([video, voice])
    db.commit()
    print(f"  [OK] Initial assets prepared for {test_vid}")

    # 2. Execute Package Creation
    print("\n2. Executing CapCut package creation...")
    pkg_res = pkg_service.create_package(db, test_vid)
    assert pkg_res["success"] is True, f"Package creation failed: {pkg_res}"
    assert pkg_res["status"] == "WAITING_CAPCUT"
    print("  [OK] create_package returned success")

    # 3. Verify All 4 Required Package Files on Disk
    print("\n3. Verifying required files in downloads/packages/{video_id}/...")
    expected_pkg_dir = PACKAGES_DIR / test_vid
    assert expected_pkg_dir.exists() and expected_pkg_dir.is_dir(), f"Package folder missing: {expected_pkg_dir}"

    # original.mp4
    orig_file = expected_pkg_dir / "original.mp4"
    assert orig_file.exists() and orig_file.stat().st_size > 0, "original.mp4 missing or empty"
    print(f"  [OK] original.mp4 exists ({orig_file.stat().st_size} bytes)")

    # voice.wav
    voice_file = expected_pkg_dir / "voice.wav"
    assert voice_file.exists() and voice_file.stat().st_size > 0, "voice.wav missing or empty"
    print(f"  [OK] voice.wav exists ({voice_file.stat().st_size} bytes)")

    # script.txt
    script_file = expected_pkg_dir / "script.txt"
    assert script_file.exists() and script_file.stat().st_size > 0, "script.txt missing or empty"
    script_content = script_file.read_text(encoding="utf-8")
    assert script_content == sample_script, "script.txt content mismatch"
    print(f"  [OK] script.txt matches stored script ({len(script_content)} chars)")

    # info.json
    info_file = expected_pkg_dir / "info.json"
    assert info_file.exists() and info_file.stat().st_size > 0, "info.json missing or empty"
    info_dict = json.loads(info_file.read_text(encoding="utf-8"))
    required_keys = ["video_id", "product_id", "product_name", "source_url", "script", "created_at"]
    for k in required_keys:
        assert k in info_dict, f"Missing key '{k}' in info.json"
    assert info_dict["video_id"] == test_vid
    assert info_dict["product_id"] == test_pid
    assert info_dict["product_name"] == "Máy cắt rau củ quả đa năng"
    assert info_dict["source_url"] == "https://v.douyin.com/pkg_test_url/"
    print(f"  [OK] info.json verified with all required fields: {list(info_dict.keys())}")

    # 4. Verify Database Status Transition to WAITING_CAPCUT
    print("\n4. Verifying database status transition...")
    db_video = db.query(Video).filter(Video.video_id == test_vid).first()
    assert db_video.status == "WAITING_CAPCUT", f"Expected WAITING_CAPCUT, got {db_video.status}"
    print(f"  [OK] Database Video {test_vid} status updated to WAITING_CAPCUT")

    # 5. Test API endpoint POST /api/videos/create-package/{video_id}
    print("\n5. Testing /api/videos/create-package API endpoint...")
    api_res = client.post(f"/api/videos/create-package/{test_vid}")
    assert api_res.status_code == 200, f"API create package failed: {api_res.text}"
    assert api_res.json()["status"] == "WAITING_CAPCUT"
    print("  [OK] POST /api/videos/create-package returned 200 WAITING_CAPCUT")

    db.close()
    print("\n=== PHASE 7 TESTS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_phase7()
