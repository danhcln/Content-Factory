import math
import os
import struct
import sys
import unittest
import wave
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.services.tts.vieneu_provider import VieNeuProvider
from app.services.ffmpeg_utils import run_ffprobe


class TestVieNeuReal(unittest.TestCase):
    def setUp(self):
        self.provider = VieNeuProvider()
        self.output_dir = Path("test_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_01_provider_availability(self):
        """Verify VieNeuProvider discovers the isolated .venv_vieneu environment."""
        self.assertTrue(self.provider.is_available(), "VieNeuProvider should report available when .venv_vieneu exists")
        voices = self.provider.get_available_voices()
        self.assertGreaterEqual(len(voices), 4)
        voice_ids = [v["id"] for v in voices]
        self.assertIn("Trúc Ly", voice_ids)
        self.assertIn("Minh Quân Pro", voice_ids)

    def test_02_real_speech_synthesis_and_validation(self):
        """Verify real VieNeu-TTS synthesis produces valid, non-silent audio."""
        test_wav = self.output_dir / "vieneu_real_test.wav"
        text = "Xin chào, đây là bài kiểm tra giọng đọc tiếng Việt."

        res = self.provider.synthesize(text=text, output_path=str(test_wav), voice_name="Trúc Ly")
        self.assertTrue(res["success"], f"Synthesis should succeed: {res.get('error')}")
        self.assertEqual(res["status"], "VOICE_READY")
        self.assertTrue(test_wav.exists(), "Output WAV must exist")
        self.assertGreater(test_wav.stat().st_size, 10000, "Audio file size should be substantial")

        # Audio properties validation
        with wave.open(str(test_wav), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            duration = nframes / float(framerate)
            raw_bytes = wf.readframes(nframes)

        self.assertEqual(channels, 1, "Output should be mono")
        self.assertEqual(framerate, 48000, "VieNeu v3 Turbo native sample rate is 48 kHz")
        self.assertGreater(duration, 1.0, "Duration should be > 1s")
        self.assertLess(duration, 5.0, "Duration should be < 5s for short sentence")

        # Digital silence check
        samples = struct.unpack(f"<{nframes}h", raw_bytes)
        rms = math.sqrt(sum(s**2 for s in samples) / len(samples))
        self.assertGreater(rms, 500, f"Audio should not be digital silence (RMS={rms})")

        # ffprobe validation
        probe_res = run_ffprobe(["-v", "error", "-show_entries", "stream=codec_name,sample_rate", "-of", "json", str(test_wav)])
        self.assertEqual(probe_res.returncode, 0)
        self.assertIn("pcm_s16le", probe_res.stdout)

    def test_03_empty_text_error_handling(self):
        """Verify provider returns error when text is empty."""
        test_wav = self.output_dir / "empty_test.wav"
        res = self.provider.synthesize(text="", output_path=str(test_wav))
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "VOICE_ERROR")


if __name__ == "__main__":
    unittest.main()
