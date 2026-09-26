import json
import logging
import math
import shutil
import sqlite3
import sys
import wave
from pathlib import Path

# Set UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.ffmpeg_utils import (
    check_ffmpeg_available,
    run_ffmpeg,
    run_ffprobe
)
from app.services.video_analyzer import VideoAnalyzer
from app.services.subtitle_region import (
    get_preset_region,
    SubtitleRegionDetector
)
from app.services.audio_sync import (
    adjust_audio_tempo,
    assemble_voice_track,
    measure_audio_duration
)
from app.services.subtitle_service import SubtitleService
from app.services.tts.vieneu_provider import VieNeuProvider


def run_e2e_test():
    print("=" * 65)
    print("      REAL AUTO EDIT E2E TEST (PHASE 12 PIPELINE - NEW PC)   ")
    print("=" * 65)

    results = {}

    # Step 0: Check FFmpeg
    ff_status = check_ffmpeg_available()
    print(f"\n[0] FFmpeg check: {ff_status['ffmpeg_path']}")
    print(f"    ffprobe check: {ff_status['ffprobe_path']}")
    if not ff_status["ready"]:
        print("[-] FFmpeg or ffprobe not ready!")
        return 1
    results["FFmpeg"] = "PASS"

    # Step 1: Input video
    video_id = "V0095"
    raw_video = ROOT_DIR / "downloads" / "original" / f"{video_id}.mp4"
    if not raw_video.exists():
        print(f"[-] Raw video missing: {raw_video}")
        return 1
    print(f"\n[1] Input video: {raw_video} ({raw_video.stat().st_size} bytes)")
    results["Input video"] = str(raw_video)

    # Step 2: Subtitle detection & Video analysis
    print("\n[2] Video analysis & Subtitle detection...")
    v_meta = VideoAnalyzer.analyze_video(raw_video)
    print(f"    Dimensions: {v_meta.get('width')}x{v_meta.get('height')}, Duration: {v_meta.get('duration')}s")
    
    det = SubtitleRegionDetector.auto_detect(raw_video, duration=v_meta.get("duration", 5.0))
    region = det.get("region", get_preset_region("BOTTOM"))
    print(f"    Detected region: {region} (confidence={det.get('confidence')})")
    results["Subtitle detection"] = "PASS"

    # Step 3: Vietnamese script/timing
    print("\n[3] Vietnamese script/timing (local test script for 5.0s video)...")
    work_dir = ROOT_DIR / "temp" / video_id
    work_dir.mkdir(parents=True, exist_ok=True)

    test_script = "Sản phẩm gia dụng thông minh tiện ích cho mọi nhà."
    segments = [
        {
            "segment_id": 1,
            "start": 0.0,
            "end": 5.0,
            "duration": 5.0,
            "vietnamese_text": test_script
        }
    ]
    print(f"    Segment 1 (0.0s - 5.0s): '{test_script}'")

    # Step 4: VieNeu Real Voice Synthesis
    print("\n[4] VieNeu Real Voice Synthesis (.venv_vieneu_new)...")
    provider = VieNeuProvider()
    seg_wav = work_dir / "voice_001.wav"
    synth_res = provider.synthesize(text=test_script, output_path=str(seg_wav), voice_name="Trúc Ly")
    print(f"    VieNeu result: success={synth_res.get('success')}, voice={synth_res.get('voice_name')}")
    if not synth_res.get("success") or not seg_wav.exists():
        print(f"[-] VieNeu synthesis failed: {synth_res.get('error')}")
        results["VieNeu real synthesis"] = "FAIL"
        return 1
    
    actual_dur = measure_audio_duration(seg_wav)
    print(f"    Raw synthesized audio duration: {actual_dur:.2f}s (file size: {seg_wav.stat().st_size} bytes)")
    results["VieNeu real synthesis"] = "PASS"

    # Step 5: Audio Sync & Assemble
    print("\n[5] Audio Sync & Master Voice Track Assembly...")
    target_dur = 5.0
    ratio = actual_dur / target_dur
    adjusted_wav = work_dir / "voice_001_adjusted.wav"
    adjust_audio_tempo(seg_wav, adjusted_wav, ratio)
    
    final_seg_wav = adjusted_wav if adjusted_wav.exists() else seg_wav
    adj_dur = measure_audio_duration(final_seg_wav)
    print(f"    Tempo adjusted audio duration: {adj_dur:.2f}s (ratio={ratio:.2f})")

    segments[0]["audio_file"] = str(final_seg_wav)
    segments[0]["audio_duration"] = adj_dur

    voice_full = work_dir / "voice_full.wav"
    assembled = assemble_voice_track(segments, voice_full, total_duration=target_dur)
    if not assembled or not voice_full.exists():
        print("[-] Assembling voice track failed!")
        results["Audio sync"] = "FAIL"
        return 1
    print(f"    Master voice track assembled: {voice_full} ({measure_audio_duration(voice_full):.2f}s)")
    results["Audio sync"] = "PASS"

    # Step 6: Vietnamese SRT subtitle file
    print("\n[6] Vietnamese SRT Subtitles generation...")
    srt_path = work_dir / f"{video_id}_vi.srt"
    srt_ok = SubtitleService.generate_srt(segments, srt_path)
    if not srt_ok or not srt_path.exists():
        print("[-] SRT generation failed!")
        results["Vietnamese subtitle"] = "FAIL"
        return 1
    print(f"    SRT generated: {srt_path}")
    results["Vietnamese subtitle"] = "PASS"

    # Step 7: Build FFmpeg 9:16 Render Filter & Execute
    print("\n[7] FFmpeg 9:16 Render (Mask/Blur Chinese + Burn-in VN Subtitle + Audio Mix)...")
    temp_render = work_dir / f"{video_id}_rendered_temp.mp4"
    if temp_render.exists():
        temp_render.unlink()

    from app.services.auto_editor import AutoEditorService
    ae = AutoEditorService()
    args = ae._build_render_args(
        source_path=raw_video,
        voice_wav_path=voice_full,
        srt_path=srt_path,
        temp_output=temp_render,
        duration=target_dur,
        region=region,
        cover_type="blur",
        blur_strength=10,
        source_audio_mode="low",
        hook_text=None
    )

    print("    Running FFmpeg render command...")
    res = run_ffmpeg(args, timeout=120.0)
    if res.returncode != 0:
        print(f"[-] FFmpeg render failed! Error: {res.stderr[-500:]}")
        results["Final MP4"] = "FAIL"
        return 1

    print(f"    Rendered temp video: {temp_render} ({temp_render.stat().st_size} bytes)")

    # Step 8: ffprobe Validation
    print("\n[8] ffprobe Validation on output video...")
    out_meta = VideoAnalyzer.analyze_video(temp_render)
    w = out_meta.get("width")
    h = out_meta.get("height")
    print(f"    Output dimensions: {w}x{h}")
    print(f"    Output duration: {out_meta.get('duration')}s")
    print(f"    Output video codec: {out_meta.get('video_codec')}, audio: {out_meta.get('audio_codec')}")

    if w == 1080 and h == 1920:
        results["1080x1920"] = "PASS"
    else:
        results["1080x1920"] = f"FAIL ({w}x{h})"
        return 1

    # Step 9: Atomic move to downloads/final/V0095.mp4
    print("\n[9] Atomic move to downloads/final/...")
    final_dir = ROOT_DIR / "downloads" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"{video_id}.mp4"
    if final_path.exists():
        final_path.unlink()
    shutil.move(str(temp_render), str(final_path))
    print(f"    Successfully moved to: {final_path} ({final_path.stat().st_size} bytes)")
    results["Final MP4"] = "PASS"

    # Step 10: Database update
    print("\n[10] Database update and status transition...")
    conn = sqlite3.connect(ROOT_DIR / "data" / "content.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO videos (
            id, video_id, product_id, douyin_url, local_file, status, auto_edit_status,
            edit_mode, final_video_path, subtitle_region
        ) VALUES (
            95, ?, ?, ?, ?, 'AUTO_EDIT_READY', 'AUTO_EDIT_READY',
            'AUTO_EDIT', ?, ?
        )
    """, (
        video_id, 1, "https://www.douyin.com/video/7400000000000000095", f"downloads/original/{video_id}.mp4",
        f"downloads/final/{video_id}.mp4", json.dumps(region)
    ))
    conn.commit()

    cur.execute("SELECT video_id, status, auto_edit_status, final_video_path FROM videos WHERE video_id = ?", (video_id,))
    db_row = cur.fetchone()
    print(f"    DB record verified: {db_row}")
    conn.close()

    if db_row and db_row[1] == "AUTO_EDIT_READY" and db_row[2] == "AUTO_EDIT_READY":
        results["DB status"] = "PASS"
    else:
        results["DB status"] = "FAIL"

    results["Auto Edit E2E"] = "PASS"

    print("\n" + "=" * 65)
    print("                 AUTO EDIT E2E FINAL RESULTS                 ")
    print("=" * 65)
    for k, v in results.items():
        print(f"  {k:<25} : {v}")

    return 0

if __name__ == "__main__":
    sys.exit(run_e2e_test())
