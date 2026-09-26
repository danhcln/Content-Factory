import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
try:
    from sqlalchemy.orm import Session
except ImportError:
    Session = Any

try:
    from app.models import Video, Voice
    from app.database import PACKAGES_DIR, ORIGINAL_DIR, BASE_DIR
except ImportError:
    Video = None
    Voice = None
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    PACKAGES_DIR = BASE_DIR / "downloads" / "packages"
    ORIGINAL_DIR = BASE_DIR / "downloads" / "original"
from app.services.tts.vieneu_provider import VieNeuProvider, create_pcm_wav_file

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path=None, override=False):
        p = Path(dotenv_path) if dotenv_path else Path(".env")
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if override or k not in os.environ:
                        os.environ[k] = v

logger = logging.getLogger("app.services.tts")

# Factory for providers
PROVIDERS = {
    "VieNeu-TTS": VieNeuProvider
}


def get_tts_provider(engine_name: str = "VieNeu-TTS"):
    cls = PROVIDERS.get(engine_name, VieNeuProvider)
    return cls()


class TTSService:
    def __init__(self):
        load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
        self.engine_name = os.getenv("TTS_ENGINE", "VieNeu-TTS")
        self.default_voice = os.getenv("DEFAULT_VOICE", "vi-VN-Standard-A")
        self.provider = get_tts_provider(self.engine_name)

    def test_voice(self, sample_text: str = "Xin chào, đây là giọng đọc thử nghiệm của AI Content Factory.") -> Dict[str, Any]:
        """Test the TTS provider with a short Vietnamese sentence."""
        test_out = ORIGINAL_DIR / "test_sample_voice.wav"
        res = self.provider.synthesize(sample_text, str(test_out), voice_name=self.default_voice)
        return res

    def generate_voice_for_video(
        self,
        db: Session,
        video_id: str,
        voice_name: Optional[str] = None,
        create_test_audio_if_blocked: bool = False
    ) -> Dict[str, Any]:
        """Synthesize voice for a video from its stored script and save to DB."""
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            return {"success": False, "error": f"Không tìm thấy video {video_id}"}

        voice = db.query(Voice).filter(Voice.video_id == video_id).first()
        if not voice or not voice.script or not voice.script.strip():
            return {"success": False, "error": f"Video {video_id} chưa có kịch bản hợp lệ."}

        target_voice = voice_name or self.default_voice
        target_dir = PACKAGES_DIR / video_id
        target_dir.mkdir(parents=True, exist_ok=True)
        output_file = target_dir / "voice.wav"

        # Attempt synthesis with provider
        synth_res = self.provider.synthesize(voice.script, str(output_file), voice_name=target_voice)

        if synth_res.get("success"):
            rel_path = f"packages/{video_id}/voice.wav"
            voice.audio_file = rel_path
            voice.voice_name = target_voice
            voice.tts_engine = self.engine_name
            voice.status = "VOICE_READY"

            video.status = "VOICE_READY"
            db.commit()
            return {
                "success": True,
                "video_id": video_id,
                "audio_file": rel_path,
                "status": "VOICE_READY"
            }
        else:
            if create_test_audio_if_blocked:
                # Create valid deterministic WAV file for test mode
                created = create_pcm_wav_file(output_file, duration_seconds=2.0)
                if created:
                    rel_path = f"packages/{video_id}/voice.wav"
                    voice.audio_file = rel_path
                    voice.voice_name = target_voice
                    voice.tts_engine = self.engine_name
                    voice.status = "VOICE_READY"

                    video.status = "VOICE_READY"
                    db.commit()
                    return {
                        "success": True,
                        "video_id": video_id,
                        "audio_file": rel_path,
                        "status": "VOICE_READY",
                        "mocked": True
                    }

            # If real run and blocked
            status_code = synth_res.get("status", "VOICE_ERROR")
            voice.status = status_code
            video.status = status_code
            video.notes = synth_res.get("error", "Lỗi tạo giọng nói")
            db.commit()
            return synth_res
