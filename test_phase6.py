import sys
import wave
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.database import SessionLocal, init_db, PACKAGES_DIR
from app.models import Product, Video, Voice
from app.services.tts import TTSService, get_tts_provider
from app.services.tts.base import BaseTTSProvider
from app.services.tts.vieneu_provider import VieNeuProvider, create_pcm_wav_file

def test_phase6():
    print("=== TESTING PHASE 6: VIENEU-TTS LOCAL ===")
    init_db()
    client = TestClient(app)
    db: Session = SessionLocal()
    tts_service = TTSService()

    # 1. Test Provider Architecture & Inheritance
    print("\n1. Testing TTS Provider Architecture...")
    provider = get_tts_provider("VieNeu-TTS")
    assert isinstance(provider, BaseTTSProvider), "VieNeuProvider must inherit from BaseTTSProvider"
    voices = provider.get_available_voices()
    assert len(voices) >= 4, f"Expected at least 4 default preset voices, found {len(voices)}"
    voice_ids = [v["id"] for v in voices]
    print(f"  [OK] Provider verified. Available preset voices: {voice_ids}")

    # 2. Test Live Model Status (Accurately reporting BLOCKED_EXTERNAL without faking success)
    print("\n2. Checking live VieNeu-TTS environment status...")
    is_avail = provider.is_available()
    print(f"  is_available(): {is_avail}")
    test_result = tts_service.test_voice("Kiểm tra thử nghiệm giọng đọc.")
    if is_avail:
        print("  [OK] VieNeu-TTS model is available live:", test_result)
    else:
        assert test_result["status"] == "BLOCKED_EXTERNAL", f"Expected BLOCKED_EXTERNAL, got {test_result}"
        print(f"  [OK] Verified accurate reporting: {test_result['status']} (No faking success)")

    # 3. Test Local Audio File Generation & Integrity
    print("\n3. Testing Local PCM WAV Generation & audio integrity...")
    test_wav = PACKAGES_DIR / "VTEST_AUDIO" / "voice.wav"
    ok = create_pcm_wav_file(test_wav, duration_seconds=1.5, sample_rate=24000)
    assert ok is True, "Failed to create WAV file"
    assert test_wav.exists() and test_wav.stat().st_size > 0, "Generated WAV is missing or empty"

    # Verify standard WAV headers using Python's standard wave library
    with wave.open(str(test_wav), 'r') as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        duration = nframes / float(framerate)
        assert channels == 1, f"Expected mono audio, got {channels}"
        assert sampwidth == 2, f"Expected 16-bit PCM, got {sampwidth}"
        assert framerate == 24000, f"Expected 24kHz, got {framerate}"
        assert duration >= 1.4, f"Expected ~1.5s duration, got {duration}"
        print(f"  [OK] Valid WAV audio verified: {duration:.2f}s, {framerate}Hz, {channels}ch, {test_wav.stat().st_size} bytes")

    # 4. Test Video / Voice Record Association and Status Transition to VOICE_READY
    print("\n4. Testing Voice record audio path storage and status transition...")
    test_vid = "VTEST_TTS_01"
    db.query(Voice).filter(Voice.video_id == test_vid).delete()
    db.query(Video).filter(Video.video_id == test_vid).delete()
    db.commit()

    v_rec = Video(video_id=test_vid, douyin_url="https://v.douyin.com/tts_test/", downloaded=True, status="SCRIPT_READY")
    voice_rec = Voice(video_id=test_vid, script="Kịch bản thử nghiệm âm thanh cục bộ VieNeu-TTS.", status="SCRIPT_READY")
    db.add_all([v_rec, voice_rec])
    db.commit()

    # Generate voice with test audio creation enabled for offline testing
    res = tts_service.generate_voice_for_video(db, test_vid, voice_name="vi-VN-Standard-A", create_test_audio_if_blocked=True)
    assert res["success"] is True, f"Voice generation failed: {res}"
    assert res["status"] == "VOICE_READY"

    # Verify DB persistence
    v_updated = db.query(Video).filter(Video.video_id == test_vid).first()
    voice_updated = db.query(Voice).filter(Voice.video_id == test_vid).first()
    assert v_updated.status == "VOICE_READY", f"Video status should be VOICE_READY, got {v_updated.status}"
    assert voice_updated.status == "VOICE_READY"
    assert voice_updated.audio_file == f"packages/{test_vid}/voice.wav"
    assert (PACKAGES_DIR / test_vid / "voice.wav").exists()
    print(f"  [OK] Database updated: Video {test_vid} -> VOICE_READY, audio_file: {voice_updated.audio_file}")

    # 5. Test Settings test-voice API endpoint
    print("\n5. Testing /api/settings/test-voice API endpoint...")
    api_res = client.post("/api/settings/test-voice")
    assert api_res.status_code == 200, f"API test voice failed: {api_res.text}"
    api_data = api_res.json()
    assert "status" in api_data
    print(f"  [OK] POST /api/settings/test-voice responded 200: Status = {api_data['status']}")

    db.close()
    print("\n=== PHASE 6 TESTS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_phase6()
