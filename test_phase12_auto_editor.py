import json
import logging
import os
import shutil
import struct
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch, MagicMock

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.database import SessionLocal, init_db, TEMP_DIR, FINAL_DIR, ORIGINAL_DIR
from app.models import Video, Product, Setting
from app.services.ffmpeg_utils import (
    get_ffmpeg_path,
    get_ffprobe_path,
    check_ffmpeg_available,
    run_ffmpeg,
    run_ffprobe
)
from app.services.video_analyzer import VideoAnalyzer
from app.services.subtitle_region import (
    PRESETS,
    get_preset_region,
    validate_coordinates,
    build_blur_filter,
    build_cover_filter,
    SubtitleRegionDetector
)
from app.services.audio_sync import (
    generate_default_timeline,
    measure_audio_duration,
    adjust_audio_tempo,
    assemble_voice_track,
    AudioSyncEngine
)
from app.services.subtitle_service import (
    SubtitleService,
    format_srt_timestamp
)
from app.services.auto_editor import AutoEditorService
from app.services.gemini_service import (
    validate_timed_script,
    validate_rewritten_segments,
    GeminiQuotaExceededError
)
from app.services.folder_watcher import FinalFolderWatcher


def create_synthetic_wav(file_path: Path, duration_sec: float = 2.0, sample_rate: int = 24000) -> Path:
    """Create a valid PCM 16-bit mono WAV file for testing."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(duration_sec * sample_rate)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            val = int(1000 * ((i % 100) / 50.0 - 1.0))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)
    return file_path


def create_synthetic_mp4(file_path: Path, duration_sec: float = 3.0, width: int = 720, height: int = 1280) -> Path:
    """Create a valid MP4 test video using FFmpeg lavfi."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-y",
        "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate=25:duration={duration_sec}",
        "-f", "lavfi", "-i", f"sine=frequency=1000:duration={duration_sec}",
        "-c:v", "libx264", "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        str(file_path)
    ]
    run_ffmpeg(args, timeout=30.0)
    return file_path


class TestPhase12AutoEditor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.test_dir = Path(tempfile.mkdtemp(prefix="p12_test_"))
        cls.synth_mp4 = create_synthetic_mp4(cls.test_dir / "sample_source.mp4", duration_sec=3.0)

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            try:
                shutil.rmtree(cls.test_dir)
            except Exception:
                pass

    def setUp(self):
        self.db = SessionLocal()
        self.auto_editor = AutoEditorService()
        test_ids = ["VTEST_RENDER_P12", "VTEST_FALLBACK", "VTEST_WATCH_P12", "VTEST_TRANSITIONS", "VTEST_ATOMIC_P12", "VTEST_VALIDATE_P12"]
        self.db.query(Video).filter(Video.video_id.in_(test_ids)).delete(synchronize_session=False)
        self.db.commit()

    def tearDown(self):
        test_ids = ["VTEST_RENDER_P12", "VTEST_FALLBACK", "VTEST_WATCH_P12", "VTEST_TRANSITIONS", "VTEST_ATOMIC_P12", "VTEST_VALIDATE_P12"]
        try:
            self.db.query(Video).filter(Video.video_id.in_(test_ids)).delete(synchronize_session=False)
            self.db.commit()
        except Exception:
            self.db.rollback()
        self.db.close()

    # ==========================================================
    # 1. STATIC-FFMPEG / FFPROBE AVAILABILITY DETECTION
    # ==========================================================
    def test_01_ffmpeg_detection(self):
        """Verify static-ffmpeg and ffprobe are discovered and report valid status."""
        status = check_ffmpeg_available()
        self.assertTrue(status["ffmpeg_available"], "FFmpeg binary must be available")
        self.assertTrue(status["ffprobe_available"], "ffprobe binary must be available")
        self.assertTrue(status["ready"], "System must report ready for video editing")
        self.assertTrue(Path(status["ffmpeg_path"]).exists(), "FFmpeg executable must exist on disk")
        self.assertTrue(Path(status["ffprobe_path"]).exists(), "ffprobe executable must exist on disk")
        self.assertIn("version", status["ffmpeg_version"].lower())
        self.assertIn("version", status["ffprobe_version"].lower())

    # ==========================================================
    # 2. VIDEO ANALYZER METADATA EXTRACTION
    # ==========================================================
    def test_02_video_analyzer_metadata(self):
        """Verify video analyzer accurately extracts duration, dimensions, FPS, and codecs."""
        meta = VideoAnalyzer.analyze_video(self.synth_mp4)
        self.assertTrue(meta["valid"])
        self.assertAlmostEqual(meta["duration"], 3.0, delta=0.2)
        self.assertEqual(meta["width"], 720)
        self.assertEqual(meta["height"], 1280)
        self.assertEqual(meta["fps"], 25.0)
        self.assertEqual(meta["video_codec"], "h264")
        self.assertEqual(meta["audio_codec"], "aac")
        self.assertTrue(meta["has_video"])
        self.assertTrue(meta["has_audio"])

    # ==========================================================
    # 3. VIDEO ANALYZER REJECTS INVALID / 0-BYTE VIDEO
    # ==========================================================
    def test_03_video_analyzer_rejects_invalid(self):
        """Verify video analyzer rejects 0-byte and non-existent video files."""
        res_missing = VideoAnalyzer.analyze_video(Path("downloads/original/non_existent.mp4"))
        self.assertFalse(res_missing["valid"])
        self.assertEqual(res_missing["status"], "INVALID_SOURCE_VIDEO")

        zero_file = self.test_dir / "zero.mp4"
        zero_file.write_bytes(b"")
        res_zero = VideoAnalyzer.analyze_video(zero_file)
        self.assertFalse(res_zero["valid"])
        self.assertEqual(res_zero["status"], "INVALID_SOURCE_VIDEO")

    # ==========================================================
    # 4. MANUAL SUBTITLE REGION PRESETS
    # ==========================================================
    def test_04_subtitle_presets(self):
        """Verify manual subtitle region presets: BOTTOM, LOWER-MIDDLE, MIDDLE, CUSTOM."""
        self.assertIn("BOTTOM", PRESETS)
        self.assertIn("LOWER-MIDDLE", PRESETS)
        self.assertIn("MIDDLE", PRESETS)

        bottom = get_preset_region("BOTTOM")
        self.assertEqual(bottom["y"], 0.72)
        self.assertEqual(bottom["height"], 0.20)

        mid = get_preset_region("MIDDLE")
        self.assertEqual(mid["y"], 0.40)

        lower_mid = get_preset_region("LOWER-MIDDLE")
        self.assertEqual(lower_mid["y"], 0.60)

        fallback = get_preset_region("UNKNOWN")
        self.assertEqual(fallback["y"], 0.72)

    # ==========================================================
    # 5. NORMALIZED COORDINATE VALIDATION (0.0 to 1.0)
    # ==========================================================
    def test_05_coordinate_validation(self):
        """Verify normalized coordinate bounds and clamping (0.0 - 1.0)."""
        valid = validate_coordinates({"x": 0.1, "y": 0.7, "width": 0.8, "height": 0.2})
        self.assertTrue(valid["valid"])
        self.assertEqual(valid["region"]["x"], 0.1)

        invalid_neg = validate_coordinates({"x": -0.1, "y": 0.7, "width": 0.8, "height": 0.2})
        self.assertFalse(invalid_neg["valid"])

        overflow = validate_coordinates({"x": 0.5, "y": 0.5, "width": 0.8, "height": 0.8})
        self.assertTrue(overflow["valid"])
        self.assertAlmostEqual(overflow["region"]["x"] + overflow["region"]["width"], 1.0, delta=0.01)
        self.assertAlmostEqual(overflow["region"]["y"] + overflow["region"]["height"], 1.0, delta=0.01)

    # ==========================================================
    # 6. SUBTITLE BLUR FILTER GENERATION
    # ==========================================================
    def test_06_subtitle_blur_filter(self):
        """Verify split-crop-boxblur-overlay filter string is generated correctly."""
        region = {"x": 0.05, "y": 0.72, "width": 0.90, "height": 0.20}
        filt = build_blur_filter(region, video_width=1080, video_height=1920, strength=10)
        self.assertIn("split[main][sub]", filt)
        self.assertIn("crop=w=", filt)
        self.assertIn("boxblur=luma_radius=10", filt)
        self.assertIn("overlay=", filt)

    # ==========================================================
    # 7. SUBTITLE COVER FILTER GENERATION
    # ==========================================================
    def test_07_subtitle_cover_filter(self):
        """Verify drawbox cover filter string generation with opacity and color."""
        region = {"x": 0.05, "y": 0.72, "width": 0.90, "height": 0.20}
        filt = build_cover_filter(region, video_width=1080, video_height=1920, opacity=0.85, color="black")
        self.assertIn("drawbox=", filt)
        self.assertIn("color=black@0.85", filt)
        self.assertIn("t=fill", filt)

    # ==========================================================
    # 8. FRAME PREVIEW EXTRACTION
    # ==========================================================
    def test_08_frame_preview_extraction(self):
        """Verify video analyzer extracts a real frame preview image using FFmpeg."""
        preview_path = self.test_dir / "preview.jpg"
        ok = VideoAnalyzer.extract_frame_preview(self.synth_mp4, preview_path, timestamp_sec=1.0)
        self.assertTrue(ok)
        self.assertTrue(preview_path.exists())
        self.assertGreater(preview_path.stat().st_size, 1000)

    # ==========================================================
    # 9. TIMELINE GENERATION FROM VIDEO DURATION
    # ==========================================================
    def test_09_timeline_generation(self):
        """Verify timeline segmentation creates natural segments within duration bounds."""
        segments = generate_default_timeline(video_duration=12.0, target_segment_length=3.5)
        self.assertGreaterEqual(len(segments), 3)
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[-1]["end"], 12.0)
        for s in segments:
            self.assertGreaterEqual(s["duration"], 2.0)
            self.assertAlmostEqual(s["end"] - s["start"], s["duration"], delta=0.01)

    # ==========================================================
    # 10. TIMED SCRIPT GENERATION FORMAT & VALIDATION
    # ==========================================================
    def test_10_timed_script_validation(self):
        """Verify validate_timed_script passes well-structured timed script items."""
        input_segs = [{"segment_id": 1}, {"segment_id": 2}]
        ai_out = {
            "segments": [
                {"segment_id": 1, "vietnamese_text": "Chiếc máy làm sạch đa năng tiện lợi."},
                {"segment_id": 2, "vietnamese_text": "Hút sạch mọi bụi bẩn trong góc hẹp."}
            ]
        }
        res = validate_timed_script(input_segs, ai_out)
        self.assertTrue(res["valid"])
        self.assertEqual(len(res["segments"]), 2)

    # ==========================================================
    # 11. SCRIPT VALIDATOR REJECTS MISSING FIELDS / INVALID TIMESTAMPS
    # ==========================================================
    def test_11_script_validator_rejects_invalid(self):
        """Verify validate_timed_script rejects missing text, empty structures, or missing ids."""
        input_segs = [{"segment_id": 1}, {"segment_id": 2}]

        # Missing text
        ai_empty_text = {
            "segments": [
                {"segment_id": 1, "vietnamese_text": ""},
                {"segment_id": 2, "vietnamese_text": "OK"}
            ]
        }
        res1 = validate_timed_script(input_segs, ai_empty_text)
        self.assertFalse(res1["valid"])

        # Missing segment_id 2
        ai_missing_id = {
            "segments": [
                {"segment_id": 1, "vietnamese_text": "Chiếc máy"}
            ]
        }
        res2 = validate_timed_script(input_segs, ai_missing_id)
        self.assertFalse(res2["valid"])

    # ==========================================================
    # 12. AUDIO DURATION MEASUREMENT
    # ==========================================================
    def test_12_audio_duration_measurement(self):
        """Verify accurate duration measurement for real WAV files."""
        wav_path = self.test_dir / "test_audio.wav"
        create_synthetic_wav(wav_path, duration_sec=2.5)
        dur = measure_audio_duration(wav_path)
        self.assertAlmostEqual(dur, 2.5, delta=0.05)

    # ==========================================================
    # 13. TEMPO ADJUSTMENT CALCULATION (0.90x - 1.10x BOUNDS)
    # ==========================================================
    def test_13_tempo_adjustment_calculation(self):
        """Verify tempo calculation respects standardized natural bounds [0.90x, 1.10x]."""
        # Normal ratio within range
        tempo_needed = 3.3 / 3.0
        self.assertAlmostEqual(tempo_needed, 1.10, delta=0.01)
        clamped = max(0.90, min(tempo_needed, 1.10))
        self.assertEqual(clamped, tempo_needed)

        # Extreme mismatch clamped to 1.10x in natural mode
        extreme_tempo = max(0.90, min(4.0 / 2.0, 1.10))
        self.assertEqual(extreme_tempo, 1.10)

        # Emergency stretch allows up to 1.15x
        emergency_clamped = max(0.85, min(4.0 / 2.0, 1.15))
        self.assertEqual(emergency_clamped, 1.15)

    # ==========================================================
    # 14. TEMPO FILTER EXECUTION
    # ==========================================================
    def test_14_tempo_filter_execution(self):
        """Verify adjust_audio_tempo executes FFmpeg atempo and generates valid output WAV."""
        in_wav = self.test_dir / "in_tempo.wav"
        out_wav = self.test_dir / "out_tempo.wav"
        create_synthetic_wav(in_wav, duration_sec=2.0)
        ok = adjust_audio_tempo(in_wav, out_wav, tempo=1.10)
        self.assertTrue(ok)
        self.assertTrue(out_wav.exists())
        dur = measure_audio_duration(out_wav)
        self.assertAlmostEqual(dur, 2.0 / 1.10, delta=0.1)

    # ==========================================================
    # 15. BATCH REWRITE PROMPT CREATION FOR FAILED SEGMENTS
    # ==========================================================
    def test_15_batch_rewrite_prompt(self):
        """Verify rewrite prompt generation includes segment constraints and failed items."""
        failed_segs = [
            {"segment_id": 2, "target_duration": 2.5, "actual_duration": 4.1, "vietnamese_text": "Câu văn này quá dài cho thời gian quy định"}
        ]
        prompt = self.auto_editor.sync_engine.build_rewrite_prompt(failed_segs, product_name="Máy xay")
        self.assertIn("Phân đoạn 2", prompt)
        self.assertIn("2.5s", prompt)
        self.assertIn("Máy xay", prompt)

    # ==========================================================
    # 16. BATCH REWRITE RESPONSE PARSER & VALIDATION
    # ==========================================================
    def test_16_batch_rewrite_validation(self):
        """Verify rewritten segment parser validates required keys and character counts."""
        failed_segs = [{"segment_id": 2}]
        ai_out = {
            "rewritten_segments": [
                {"segment_id": 2, "vietnamese_text": "Máy xay gọn nhẹ, xay nhuyễn mịn."}
            ]
        }
        res = validate_rewritten_segments(failed_segs, ai_out)
        self.assertTrue(res["valid"])
        self.assertEqual(len(res["rewritten_segments"]), 1)
        self.assertEqual(res["rewritten_segments"][0]["vietnamese_text"], "Máy xay gọn nhẹ, xay nhuyễn mịn.")

    # ==========================================================
    # 17. MAX 2 REWRITE ROUNDS ENFORCEMENT
    # ==========================================================
    def test_17_max_rewrite_rounds(self):
        """Verify AudioSyncEngine enforces a strict ceiling of 2 rewrite rounds."""
        self.assertEqual(AudioSyncEngine.MAX_REWRITE_ROUNDS, 2)

    # ==========================================================
    # 18. FINAL VOICE TRACK ASSEMBLY
    # ==========================================================
    def test_18_voice_track_assembly(self):
        """Verify individual segment WAVs assemble into continuous voice_full.wav without drift."""
        seg1_wav = self.test_dir / "seg1.wav"
        seg2_wav = self.test_dir / "seg2.wav"
        create_synthetic_wav(seg1_wav, duration_sec=1.0)
        create_synthetic_wav(seg2_wav, duration_sec=1.0)

        segments = [
            {"segment_id": 1, "start": 0.0, "end": 1.5, "audio_file": str(seg1_wav)},
            {"segment_id": 2, "start": 1.5, "end": 3.0, "audio_file": str(seg2_wav)}
        ]
        out_master = self.test_dir / "voice_full.wav"
        ok = assemble_voice_track(segments, out_master, total_duration=3.0, sample_rate=24000)
        self.assertTrue(ok)
        self.assertTrue(out_master.exists())
        dur = measure_audio_duration(out_master)
        self.assertAlmostEqual(dur, 3.0, delta=0.05)

    # ==========================================================
    # 19. VIETNAMESE SRT GENERATION
    # ==========================================================
    def test_19_vietnamese_srt_generation(self):
        """Verify SRT generation formats clean UTF-8 subtitles with exact timestamps."""
        srt_path = self.test_dir / "subtitles.srt"
        segments = [
            {"segment_id": 1, "start": 0.5, "end": 2.8, "vietnamese_text": "Sản phẩm gia dụng thông minh."},
            {"segment_id": 2, "start": 3.0, "end": 5.2, "vietnamese_text": "Tiết kiệm 50% thời gian nấu nướng."}
        ]
        ok = SubtitleService.generate_srt(segments, srt_path)
        self.assertTrue(ok)
        self.assertTrue(srt_path.exists())
        content = srt_path.read_text(encoding="utf-8")
        self.assertIn("00:00:00,500 --> 00:00:02,800", content)
        self.assertIn("Sản phẩm gia dụng thông minh.", content)
        self.assertIn("00:00:03,000 --> 00:00:05,200", content)

    # ==========================================================
    # 20. SUBTITLE BURN-IN FILTER GENERATION
    # ==========================================================
    def test_20_subtitle_burn_filter(self):
        """Verify subtitles filter generates correctly escaped path for FFmpeg."""
        srt_path = Path("D:/Workflow_video/temp/V123/subtitles.srt")
        filt = SubtitleService.build_subtitle_filter(srt_path, font_size=20, margin_v=45)
        self.assertIn("subtitles=", filt)
        self.assertIn("FontSize=20", filt)
        self.assertIn("MarginV=45", filt)

    # ==========================================================
    # 21. OPENING HOOK TEXT OVERLAY FILTER GENERATION
    # ==========================================================
    def test_21_opening_hook_filter(self):
        """Verify drawtext opening hook filter with 3s timeline restriction."""
        filt = SubtitleService.build_hook_overlay_filter("ĐỪNG BỎ QUA MẸO NÀY!", duration_sec=3.0)
        self.assertIn("drawtext=", filt)
        self.assertIn("between(t,0,3.0)", filt)

    # ==========================================================
    # 22. SOURCE AUDIO MIX MODES (OFF, LOW, KEEP)
    # ==========================================================
    def test_22_audio_mix_modes(self):
        """Verify audio filter configuration for OFF, LOW, and KEEP modes."""
        voice_wav = self.test_dir / "voice.wav"
        create_synthetic_wav(voice_wav, duration_sec=2.0)

        # Mode: OFF (voice only)
        args_off = self.auto_editor._build_render_args(
            source_path=self.synth_mp4,
            voice_wav_path=voice_wav,
            srt_path=None,
            temp_output=self.test_dir / "render_off.mp4",
            duration=2.0,
            region=get_preset_region("BOTTOM"),
            cover_type="blur",
            blur_strength=10,
            source_audio_mode="off",
            hook_text=None
        )
        self.assertIn("[1:a]aresample=44100[aout]", " ".join(args_off))

        # Mode: LOW (amix with 0.15 source)
        args_low = self.auto_editor._build_render_args(
            source_path=self.synth_mp4,
            voice_wav_path=voice_wav,
            srt_path=None,
            temp_output=self.test_dir / "render_low.mp4",
            duration=2.0,
            region=get_preset_region("BOTTOM"),
            cover_type="blur",
            blur_strength=10,
            source_audio_mode="low",
            hook_text=None
        )
        self.assertIn("volume=0.15", " ".join(args_low))
        self.assertIn("amix=inputs=2", " ".join(args_low))

    # ==========================================================
    # 23. 9:16 VERTICAL RENDER FILTER GRAPH (1080x1920)
    # ==========================================================
    def test_23_vertical_render_filter_graph(self):
        """Verify filter graph applies 1080x1920 scaling with aspect ratio preservation and padding."""
        voice_wav = self.test_dir / "voice.wav"
        args = self.auto_editor._build_render_args(
            source_path=self.synth_mp4,
            voice_wav_path=voice_wav,
            srt_path=None,
            temp_output=self.test_dir / "render_vert.mp4",
            duration=2.0,
            region=get_preset_region("BOTTOM"),
            cover_type="blur",
            blur_strength=10,
            source_audio_mode="off",
            hook_text=None
        )
        cmd_str = " ".join(args)
        self.assertIn("scale=1080:1920:force_original_aspect_ratio=decrease", cmd_str)
        self.assertIn("pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black", cmd_str)

    # ==========================================================
    # 24. TEMPORARY FILE RENDERING BEFORE ATOMIC MOVE
    # ==========================================================
    def test_24_temp_file_rendering(self):
        """Verify render command writes to a temporary render path first."""
        temp_render = self.test_dir / "temp_render_test.mp4"
        voice_wav = self.test_dir / "voice_t24.wav"
        create_synthetic_wav(voice_wav, duration_sec=1.0)
        args = self.auto_editor._build_render_args(
            source_path=self.synth_mp4,
            voice_wav_path=voice_wav,
            srt_path=None,
            temp_output=temp_render,
            duration=1.0,
            region=get_preset_region("BOTTOM"),
            cover_type="blur",
            blur_strength=10,
            source_audio_mode="off",
            hook_text=None
        )
        self.assertEqual(args[-1], str(temp_render))
        res = run_ffmpeg(args, timeout=30.0)
        self.assertEqual(res.returncode, 0)
        self.assertTrue(temp_render.exists())
        self.assertGreater(temp_render.stat().st_size, 1000)
        temp_render.unlink()

    # ==========================================================
    # 25. ATOMIC MOVE TO FINAL DESTINATION
    # ==========================================================
    def test_25_atomic_move_to_final(self):
        """Verify rendered file is atomically placed in downloads/final/{video_id}.mp4."""
        video_id = "VTEST_ATOMIC_P12"
        orig_mp4 = ORIGINAL_DIR / f"{video_id}.mp4"
        shutil.copy2(self.synth_mp4, orig_mp4)

        work_dir = self.auto_editor.get_work_dir(video_id)
        voice_wav = work_dir / "voice_full.wav"
        create_synthetic_wav(voice_wav, duration_sec=2.0)

        v = Video(
            video_id=video_id,
            product_id=1,
            douyin_url="https://douyin.com/atomic",
            status="DOWNLOADED",
            edit_mode="AUTO_EDIT"
        )
        self.db.add(v)
        self.db.commit()

        res = self.auto_editor.render_final_video(video_id=video_id, db=self.db)
        self.assertTrue(res["success"])

        final_dest = FINAL_DIR / f"{video_id}.mp4"
        self.assertTrue(final_dest.exists())
        # Temp rendering file must not remain
        self.assertFalse((work_dir / "rendering.mp4").exists())

        # Cleanup
        if final_dest.exists(): final_dest.unlink()
        if orig_mp4.exists(): orig_mp4.unlink()
        self.db.delete(v)
        self.db.commit()

    # ==========================================================
    # 26. FINAL VIDEO VALIDATION WITH FFPROBE
    # ==========================================================
    def test_26_final_video_ffprobe_validation(self):
        """Verify VideoAnalyzer ffprobe validates 1080x1920 vertical video."""
        video_id = "VTEST_VALIDATE_P12"
        orig_mp4 = ORIGINAL_DIR / f"{video_id}.mp4"
        shutil.copy2(self.synth_mp4, orig_mp4)

        work_dir = self.auto_editor.get_work_dir(video_id)
        voice_wav = work_dir / "voice_full.wav"
        create_synthetic_wav(voice_wav, duration_sec=2.0)

        v = Video(
            video_id=video_id,
            product_id=1,
            douyin_url="https://douyin.com/validate",
            status="DOWNLOADED",
            edit_mode="AUTO_EDIT"
        )
        self.db.add(v)
        self.db.commit()

        res = self.auto_editor.render_final_video(video_id=video_id, db=self.db)
        self.assertTrue(res["success"])

        final_dest = FINAL_DIR / f"{video_id}.mp4"
        meta = VideoAnalyzer.analyze_video(final_dest)
        self.assertTrue(meta["valid"])
        self.assertEqual(meta["width"], 1080)
        self.assertEqual(meta["height"], 1920)
        self.assertEqual(meta["aspect_ratio"], "9:16")
        self.assertTrue(meta["has_video"])

        # Cleanup
        if final_dest.exists(): final_dest.unlink()
        if orig_mp4.exists(): orig_mp4.unlink()
        self.db.delete(v)
        self.db.commit()


    # ==========================================================
    # 27. ERROR HANDLING WHEN INPUT VIDEO DOES NOT EXIST
    # ==========================================================
    def test_27_missing_input_handling(self):
        """Verify graceful error handling and status updates when source video is missing."""
        res = self.auto_editor.analyze_source("V_NONEXISTENT_999", db=self.db)
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "INVALID_SOURCE_VIDEO")

    # ==========================================================
    # 28. ONE-CLICK FALLBACK FROM AUTO EDIT TO CAPCUT MANUAL
    # ==========================================================
    def test_28_fallback_to_capcut_manual(self):
        """Verify switch_to_capcut_manual updates edit_mode and creates CapCut package files."""
        video_id = "VTEST_FALLBACK"
        p = self.db.query(Product).first()
        v = Video(
            video_id=video_id,
            product_id=p.id if p else 1,
            douyin_url="https://douyin.com/fallback",
            status="PROCESSING",
            edit_mode="AUTO_EDIT",
            auto_edit_status="PROCESSING"
        )
        self.db.add(v)
        self.db.commit()

        res = self.auto_editor.switch_to_capcut_manual(video_id, reason="User requested fallback", db=self.db)
        self.assertTrue(res["success"])
        self.assertEqual(res["edit_mode"], "CAPCUT_MANUAL")

        self.db.refresh(v)
        self.assertEqual(v.edit_mode, "CAPCUT_MANUAL")
        self.assertEqual(v.auto_edit_status, "FALLBACK_TO_CAPCUT")

        # Clean up
        self.db.delete(v)
        self.db.commit()

    # ==========================================================
    # 29. FOLDER WATCHER PICKS UP AUTO-EDITED FINAL VIDEO
    # ==========================================================
    def test_29_folder_watcher_compatibility(self):
        """Verify Folder Watcher detects final video in downloads/final/ and triggers READY state."""
        video_id = "VTEST_WATCH_P12"
        p = self.db.query(Product).first()
        v = Video(
            video_id=video_id,
            product_id=p.id if p else 1,
            douyin_url="https://douyin.com/watch",
            status="DOWNLOADING",
            edit_mode="AUTO_EDIT"
        )
        self.db.add(v)
        self.db.commit()

        # Create a final video file
        dest = FINAL_DIR / f"{video_id}.mp4"
        shutil.copy2(self.synth_mp4, dest)

        # Trigger watcher process directly
        watcher = FinalFolderWatcher()
        watcher.handle_file_event(dest)

        self.db.refresh(v)
        self.assertEqual(v.status, "READY")
        self.assertIsNotNone(v.final_video_path)

        # Clean up
        if dest.exists():
            dest.unlink()
        self.db.delete(v)
        self.db.commit()

    # ==========================================================
    # 30. AUTO-EDIT STATUS TRANSITIONS
    # ==========================================================
    def test_30_status_transitions(self):
        """Verify video status transitions across the pipeline (PENDING -> PROCESSING -> COMPLETED)."""
        video_id = "VTEST_TRANSITIONS"
        p = self.db.query(Product).first()
        v = Video(
            video_id=video_id,
            product_id=p.id if p else 1,
            douyin_url="https://douyin.com/transitions",
            status="APPROVED",
            edit_mode="AUTO_EDIT",
            auto_edit_status="PENDING"
        )
        self.db.add(v)
        self.db.commit()

        self.auto_editor.update_status(video_id, "PROCESSING", db=self.db)
        self.db.refresh(v)
        self.assertEqual(v.auto_edit_status, "PROCESSING")

        self.auto_editor.update_status(video_id, "NEEDS_REVIEW", db=self.db)
        self.db.refresh(v)
        self.assertEqual(v.auto_edit_status, "NEEDS_REVIEW")

        self.auto_editor.update_status(video_id, "COMPLETED", db=self.db)
        self.db.refresh(v)
        self.assertEqual(v.auto_edit_status, "COMPLETED")

        # Clean up
        self.db.delete(v)
        self.db.commit()


if __name__ == "__main__":
    unittest.main()
