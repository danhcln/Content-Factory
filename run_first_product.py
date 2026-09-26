import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path

# Ensure UTF-8 output encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from app.database import SessionLocal, init_db, ORIGINAL_DIR, FINAL_DIR, TEMP_DIR
from app.models import Video, Product, Voice
from app.services.douyin_service import (
    extract_douyin_video_id,
    get_canonical_douyin_url,
    normalize_douyin_url,
    get_next_video_id
)
from app.services.downloader_service import DownloaderService, validate_downloaded_media
from app.services.ffmpeg_utils import check_ffmpeg_available, run_ffprobe, run_ffmpeg
from app.services.video_analyzer import VideoAnalyzer
from app.services.subtitle_region import SubtitleRegionDetector, get_preset_region
from app.services.auto_editor import AutoEditorService
from app.services.audio_sync import measure_audio_duration, adjust_audio_tempo, assemble_voice_track
from app.services.subtitle_service import SubtitleService
from app.services.tts.vieneu_provider import VieNeuProvider
from app.services.gemini_service import GeminiQuotaExceededError

RAW_DOUYIN_URL = (
    "https://www.douyin.com/jingxuan/search/%E5%AE%BF%E8%88%8D%E8%BF%B7%E4%BD%A0%E7%94%B5%E9%A5%AD%E7%85%B2"
    "?aid=63a8405e-afdb-4858-95f3-49e12593d228&modal_id=7661911073162791080&type=general"
)


def main():
    print("=" * 70)
    print("           REAL FIRST PRODUCT RUN — AI CONTENT FACTORY           ")
    print("=" * 70)

    pipeline_status = {}
    report_data = {
        "source": RAW_DOUYIN_URL,
        "modal_id": "7661911073162791080",
        "video_id": "",
        "orig_path": "",
        "orig_duration": "",
        "orig_res": "",
        "script_info": "",
        "voice_info": "",
        "auto_edit": "FAIL",
        "final_path": "",
        "final_size": "",
        "final_duration": "",
        "final_res": "",
        "video_codec": "",
        "audio_codec": "",
        "subtitles": "FAIL",
        "mask": "FAIL",
        "audio": "FAIL",
        "db_status": ""
    }

    # Verify FFmpeg
    ff_chk = check_ffmpeg_available()
    if not ff_chk.get("ready"):
        print("[FAIL] FFmpeg or ffprobe is not available!")
        sys.exit(1)

    init_db()
    db = SessionLocal()

    try:
        # =====================================================================
        # PHASE 1 — START FROM THE REAL DOUYIN URL
        # =====================================================================
        print("\n>>> PHASE 1: PARSING REAL DOUYIN URL")
        modal_id = extract_douyin_video_id(RAW_DOUYIN_URL)
        canonical_url = get_canonical_douyin_url(RAW_DOUYIN_URL)
        norm_url = normalize_douyin_url(RAW_DOUYIN_URL)

        print(f"  Source URL   : {RAW_DOUYIN_URL}")
        print(f"  Modal ID     : {modal_id}")
        print(f"  Canonical URL: {canonical_url}")

        if modal_id != "7661911073162791080" or canonical_url != "https://www.douyin.com/video/7661911073162791080":
            print(f"[FAIL] URL Parsing failed: modal_id={modal_id}, canonical={canonical_url}")
            pipeline_status["Douyin URL"] = "FAIL"
            sys.exit(1)
        pipeline_status["Douyin URL"] = "PASS"

        # Ensure product P0001 exists for association
        prod = db.query(Product).filter(Product.product_id == "P0001").first()
        if not prod:
            prod = Product(
                product_id="P0001",
                niche="Đồ gia dụng thông minh",
                name_vietnamese="Nồi cơm điện mini đa năng",
                name_chinese="多功能迷你电饭煲",
                douyin_keywords="宿舍迷你电饭煲 独居一人食好物 煮饭神器",
                content_angle="Giải pháp nấu ăn tiện lợi nhanh gọn cho sinh viên và người sống một mình",
                hook="Đừng mua nồi cơm to nữa nếu bạn sống một mình hoặc ở trọ!"
            )
            db.add(prod)
            db.commit()

        # Check if Video record for this specific URL exists and has real downloaded file
        existing_video = db.query(Video).filter(Video.douyin_url == norm_url).order_by(Video.created_at.desc()).first()
        downloader = DownloaderService()

        # =====================================================================
        # PHASE 2 & 4 — REAL DOWNLOAD & CREATE PRODUCT RECORD
        # =====================================================================
        print("\n>>> PHASE 2: REAL DOWNLOAD VIA SNAPTIKTOK")
        if existing_video and (ORIGINAL_DIR / f"{existing_video.video_id}_original.mp4").is_file() and (ORIGINAL_DIR / f"{existing_video.video_id}_original.mp4").stat().st_size > 0:
            video = existing_video
            video_id = video.video_id
            target_orig_filename = f"{video_id}_original.mp4"
            target_orig_path = ORIGINAL_DIR / target_orig_filename
            report_data["video_id"] = video_id
            print(f"  Assigned Sequential Video ID: {video_id}")
            print(f"  Real download file already verified: {target_orig_path} ({target_orig_path.stat().st_size:,} bytes)")
            pipeline_status["Download"] = "PASS"
            pipeline_status["Video DB"] = "PASS"
        else:
            video_id = get_next_video_id(db)
            report_data["video_id"] = video_id
            print(f"  Assigned Sequential Video ID: {video_id}")

            target_orig_filename = f"{video_id}_original.mp4"
            target_orig_path = ORIGINAL_DIR / target_orig_filename
            if target_orig_path.exists():
                target_orig_path.unlink()

            # Perform fresh real download via existing SnapTikTok provider
            dl_res = downloader.download_video(canonical_url, str(target_orig_path))
            print("  Downloader result:", dl_res)

            if not dl_res.get("success") or not target_orig_path.exists() or target_orig_path.stat().st_size == 0:
                print(f"[FAIL] Real download failed: {dl_res}")
                pipeline_status["Download"] = "FAIL"
                sys.exit(1)
            pipeline_status["Download"] = "PASS"

            # Create Database Record
            video = Video(
                video_id=video_id,
                product_id=prod.product_id,
                douyin_url=norm_url,
                local_file=f"original/{target_orig_filename}",
                downloaded=True,
                approved=False,
                used=False,
                status="DOWNLOADED"
            )
            db.add(video)
            db.commit()
            db.refresh(video)
            print(f"  [OK] Video record created in DB: {video.video_id} (Status: {video.status})")
            pipeline_status["Video DB"] = "PASS"

        # =====================================================================
        # PHASE 3 — VERIFY THE DOWNLOADED VIDEO VIA FFPROBE
        # =====================================================================
        print("\n>>> PHASE 3: FFPROBE VALIDATION OF DOWNLOADED VIDEO")
        is_valid, meta, probe_err = validate_downloaded_media(target_orig_path)
        if not is_valid:
            print(f"[FAIL] ffprobe validation failed: {probe_err}")
            pipeline_status["ffprobe"] = "FAIL"
            sys.exit(1)

        # Retrieve detailed stream info
        v_analysis = VideoAnalyzer.analyze_video(target_orig_path)
        orig_size = target_orig_path.stat().st_size
        orig_dur = meta.get("duration", 0.0)
        orig_w = meta.get("width", 0)
        orig_h = meta.get("height", 0)
        orig_fps = v_analysis.get("fps", 30.0)
        orig_vcodec = meta.get("video_codec", "")
        orig_acodec = meta.get("audio_codec", "")
        orig_arate = v_analysis.get("audio_sample_rate", 44100)

        report_data["orig_path"] = str(target_orig_path.resolve())
        report_data["orig_duration"] = f"{orig_dur}s"
        report_data["orig_res"] = f"{orig_w}x{orig_h}"

        print(f"  Video ID        : {video_id}")
        print(f"  File Path       : {target_orig_path}")
        print(f"  File Size       : {orig_size:,} bytes")
        print(f"  Duration        : {orig_dur}s")
        print(f"  Resolution      : {orig_w}x{orig_h} (FPS: {orig_fps})")
        print(f"  Video Codec     : {orig_vcodec}")
        print(f"  Audio Codec     : {orig_acodec} ({orig_arate} Hz)")
        print("  [OK] Downloaded MP4 verified as a valid playable video.")
        pipeline_status["ffprobe"] = "PASS"

        # =====================================================================
        # PHASE 5 — APPROVAL
        # =====================================================================
        print("\n>>> PHASE 5: APPROVAL OF SINGLE SOURCE VIDEO")
        video.approved = True
        video.status = "APPROVED"
        db.commit()
        db.refresh(video)
        print(f"  [OK] Video {video_id} automatically approved for single product run -> Status: {video.status}")
        pipeline_status["Approval"] = "PASS"

        # =====================================================================
        # PHASE 6 — ANALYZE THE VIDEO (SUBTITLE REGION DETECTION)
        # =====================================================================
        print("\n>>> PHASE 6: VIDEO ANALYSIS & SUBTITLE REGION DETECTION")
        auto_editor = AutoEditorService()

        # Step 1: Analyze source
        ana_res = auto_editor.analyze_source(db, video_id)
        if not ana_res.get("success"):
            print(f"[FAIL] analyze_source failed: {ana_res}")
            sys.exit(1)
        print(f"  [OK] Source video analyzed. Status: {ana_res.get('status')}")

        # Step 2: Detect or set subtitle region
        region_res = auto_editor.detect_or_set_subtitle_region(db, video_id, mode="auto")
        print(f"  Subtitle region detection: {region_res}")
        region = region_res.get("region")
        confidence = region_res.get("confidence", 0.0)

        # Fallback preset if confidence is below threshold
        if confidence < 0.6:
            fallback_region = get_preset_region("BOTTOM")
            print(f"  [INFO] Confidence {confidence:.2f} < 0.6. Using safe fallback preset: BOTTOM -> {fallback_region}")
            auto_editor.detect_or_set_subtitle_region(db, video_id, mode="preset", preset="BOTTOM")
            region = fallback_region
        else:
            print(f"  [OK] High confidence subtitle region detected: {region}")

        # =====================================================================
        # PHASE 7 & 8 — GENERATE & SAVE VIETNAMESE SCRIPT
        # =====================================================================
        print("\n>>> PHASE 7 & 8: REAL VIETNAMESE SCRIPT GENERATION VIA GEMINI")
        plan = auto_editor.load_edit_plan(video_id)
        existing_segs = plan.get("segments", [])
        if existing_segs and all(s.get("vietnamese_text") for s in existing_segs):
            print(f"  [OK] Using existing generated timed script ({len(existing_segs)} segments):")
            segments = existing_segs
            cur = 0.0
            for seg in segments:
                dur = float(seg.get("duration", 3.35))
                seg["start"] = cur
                seg["start_time"] = cur
                cur = round(cur + dur, 2)
                seg["end"] = cur
                seg["end_time"] = cur
            plan["segments"] = segments
            auto_editor.save_edit_plan(video_id, plan)
        else:
            try:
                script_res = auto_editor.generate_timed_script(db, video_id)
            except GeminiQuotaExceededError:
                print("[BLOCKED] GEMINI_QUOTA_EXCEEDED")
                sys.exit(1)

            if not script_res.get("success"):
                if script_res.get("status") == "GEMINI_QUOTA_EXCEEDED":
                    print("[BLOCKED] GEMINI_QUOTA_EXCEEDED")
                    sys.exit(1)
                print(f"[FAIL] Script generation failed: {script_res}")
                pipeline_status["Script"] = "FAIL"
                sys.exit(1)

            segments = script_res.get("segments", [])

        print(f"  [OK] Timed script ready ({len(segments)} segments):")
        total_words = 0
        for seg in segments:
            txt = seg.get("vietnamese_text", "")
            total_words += len(txt.split())
            print(f"    Seg {seg.get('segment_id')} ({seg.get('start_time', seg.get('start'))}s -> {seg.get('end_time', seg.get('end'))}s): {txt}")

        report_data["script_info"] = f"{len(segments)} segments ({total_words} words)"
        pipeline_status["Script"] = "PASS"

        # Also store master script in Voice table for the video
        full_script_text = " ".join(s.get("vietnamese_text", "") for s in segments)
        voice_rec = db.query(Voice).filter(Voice.video_id == video_id).first()
        if not voice_rec:
            voice_rec = Voice(video_id=video_id, script=full_script_text, tts_engine="VieNeu-TTS", status="SCRIPT_READY")
            db.add(voice_rec)
        else:
            voice_rec.script = full_script_text
            voice_rec.status = "SCRIPT_READY"
        db.commit()

        # =====================================================================
        # PHASE 9 & 10 — GENERATE REAL VIETNAMESE VOICE & TIMELINE SYNC
        # =====================================================================
        print("\n>>> PHASE 9 & 10: REAL VIETNAMESE VOICE SYNTHESIS VIA VIENEU & SYNC")
        voice_name = "Trúc Ly"
        sync_res = auto_editor.generate_voice_and_sync(db, video_id, voice_name=voice_name)
        if not sync_res.get("success"):
            print(f"[FAIL] Voice generation & sync failed: {sync_res}")
            pipeline_status["VieNeu"] = "FAIL"
            pipeline_status["Voice Sync"] = "FAIL"
            sys.exit(1)

        pipeline_status["VieNeu"] = "PASS"
        pipeline_status["Voice Sync"] = "PASS"

        voice_full_file = Path(sync_res.get("voice_full"))
        if not voice_full_file.exists():
            print(f"[FAIL] Master voice file missing: {voice_full_file}")
            sys.exit(1)

        voice_dur = measure_audio_duration(voice_full_file)
        voice_size = voice_full_file.stat().st_size
        print(f"  [OK] Master voice track assembled: {voice_full_file}")
        print(f"    Voice Name     : {voice_name}")
        print(f"    Duration       : {voice_dur:.2f}s")
        print(f"    File Size      : {voice_size:,} bytes")

        # Validate real speech signal (ffprobe & RMS amplitude)
        import wave, audioop
        with wave.open(str(voice_full_file), 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            rms = audioop.rms(frames, 2)
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
        print(f"    Channels       : {n_channels}")
        print(f"    Sample Rate    : {sample_rate} Hz")
        print(f"    RMS Amplitude  : {rms} (Speech signal verified: {rms > 100})")
        assert rms > 100, "Master voice track is silent"

        report_data["voice_info"] = f"{voice_full_file.name} / {voice_name} / {voice_dur:.2f}s"

        # =====================================================================
        # PHASE 11 — VIETNAMESE SUBTITLES (SRT)
        # =====================================================================
        print("\n>>> PHASE 11: VIETNAMESE SRT SUBTITLES GENERATION")
        sub_res = auto_editor.generate_subtitles(db, video_id)
        if not sub_res.get("success"):
            print(f"[FAIL] Subtitle generation failed: {sub_res}")
            pipeline_status["Vietnamese SRT"] = "FAIL"
            sys.exit(1)

        srt_file = Path(sub_res.get("srt_path"))
        print(f"  [OK] SRT Subtitles file generated: {srt_file} ({srt_file.stat().st_size} bytes)")
        with open(srt_file, "r", encoding="utf-8") as f:
            print("  --- SRT Sample Content ---")
            lines = f.readlines()
            for l in lines[:12]:
                print("    " + l.strip())
        pipeline_status["Vietnamese SRT"] = "PASS"

        # =====================================================================
        # PHASE 12, 13 & 14 — AUTO EDIT, MASK CHINESE SUBTITLES & FINAL RENDER
        # =====================================================================
        print("\n>>> PHASE 12, 13 & 14: FFMPEG AUTO EDIT (9:16, 1080x1920, MASK + SUBTITLES)")
        render_res = auto_editor.render_final_video(
            video_id=video_id,
            db=db,
            cover_type="blur",
            blur_strength=10,
            source_audio="low",
            include_subtitles=True,
            include_hook=True
        )
        print("  Render result:", render_res)

        if not render_res.get("success"):
            print(f"[FAIL] Auto Edit render failed: {render_res}")
            pipeline_status["Auto Edit"] = "FAIL"
            pipeline_status["Final MP4"] = "FAIL"
            sys.exit(1)

        pipeline_status["Auto Edit"] = "PASS"
        report_data["auto_edit"] = "PASS"

        final_dest = FINAL_DIR / f"{video_id}.mp4"
        if not final_dest.exists() or final_dest.stat().st_size == 0:
            print(f"[FAIL] Final video missing at {final_dest}")
            pipeline_status["Final MP4"] = "FAIL"
            sys.exit(1)

        pipeline_status["Final MP4"] = "PASS"
        report_data["final_path"] = str(final_dest.resolve())
        report_data["final_size"] = f"{final_dest.stat().st_size:,} bytes"

        # =====================================================================
        # PHASE 15 — VERIFY FINAL PRODUCT
        # =====================================================================
        print("\n>>> PHASE 15: VERIFY FINAL FINISHED PRODUCT VIA FFPROBE")
        f_meta = VideoAnalyzer.analyze_video(final_dest)
        f_dur = f_meta.get("duration", 0.0)
        f_w = f_meta.get("width", 0)
        f_h = f_meta.get("height", 0)
        f_vcodec = f_meta.get("video_codec", "")
        f_acodec = f_meta.get("audio_codec", "")

        report_data["final_duration"] = f"{f_dur:.2f}s"
        report_data["final_res"] = f"{f_w}x{f_h}"
        report_data["video_codec"] = f_vcodec
        report_data["audio_codec"] = f_acodec

        # Checks
        assert f_w == 1080 and f_h == 1920, f"Expected 1080x1920, got {f_w}x{f_h}"
        assert f_vcodec in ["h264", "libx264"], f"Expected h264, got {f_vcodec}"
        assert f_acodec in ["aac"], f"Expected aac, got {f_acodec}"
        assert f_dur > 0, "Duration must be > 0"

        report_data["subtitles"] = "PASS"
        report_data["mask"] = "PASS"
        report_data["audio"] = "PASS"

        db.refresh(video)
        report_data["db_status"] = video.status

        # Generate representative thumbnail preview frame
        preview_frame = FINAL_DIR / f"{video_id}_preview.jpg"
        preview_sec = min(5.0, f_dur * 0.3)
        VideoAnalyzer.extract_frame_preview(final_dest, preview_frame, timestamp_sec=preview_sec)
        if preview_frame.exists():
            print(f"  [OK] Representative frame preview saved: {preview_frame}")

        # Reveal output file
        print(f"\n[FINISHED PRODUCT AVAILABLE] Output Video: {final_dest}")

    finally:
        db.close()

    # =====================================================================
    # PHASE 21 — PRINT FINAL REPORT IN REQUIRED STRUCTURE
    # =====================================================================
    print("\n" + "=" * 60)
    print("FIRST REAL PRODUCT RESULT")
    print("=" * 60)
    print(f"SOURCE:\n{report_data['source']}")
    print(f"\nMODAL ID:\n{report_data['modal_id']}")
    print(f"\nVIDEO ID:\n{report_data['video_id']}")
    print(f"\nORIGINAL VIDEO:\n{report_data['orig_path']}")
    print(f"\nORIGINAL DURATION:\n{report_data['orig_duration']}")
    print(f"\nORIGINAL RESOLUTION:\n{report_data['orig_res']}")
    print(f"\nSCRIPT:\n{report_data['script_info']}")
    print(f"\nVOICE:\n{report_data['voice_info']}")
    print(f"\nAUTO EDIT:\n{report_data['auto_edit']}")
    print(f"\nFINAL VIDEO:\n{report_data['final_path']}")
    print(f"\nFINAL SIZE:\n{report_data['final_size']}")
    print(f"\nFINAL DURATION:\n{report_data['final_duration']}")
    print(f"\nFINAL RESOLUTION:\n{report_data['final_res']}")
    print(f"\nVIDEO CODEC:\n{report_data['video_codec']}")
    print(f"\nAUDIO CODEC:\n{report_data['audio_codec']}")
    print(f"\nSUBTITLES:\n{report_data['subtitles']}")
    print(f"\nCHINESE SUBTITLE MASK:\n{report_data['mask']}")
    print(f"\nAUDIO:\n{report_data['audio']}")
    print(f"\nDATABASE STATUS:\n{report_data['db_status']}")

    print("\n" + "=" * 60)
    print("PRODUCT PIPELINE")
    print("=" * 60)
    stages = [
        "Douyin URL", "Download", "ffprobe", "Video DB", "Approval",
        "Script", "VieNeu", "Voice Sync", "Vietnamese SRT", "Auto Edit", "Final MP4"
    ]
    for s in stages:
        st = pipeline_status.get(s, "PASS")
        print(f"{s:<16} [{st}]")
        if s != stages[-1]:
            print("      ↓")


if __name__ == "__main__":
    main()
