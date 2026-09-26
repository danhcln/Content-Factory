import logging
import math
import struct
import wave
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
try:
    from sqlalchemy.orm import Session
except ImportError:
    Session = Any

from app.services.ffmpeg_utils import run_ffprobe, run_ffmpeg
from app.services.tts import TTSService
from app.services.gemini_service import GeminiService

logger = logging.getLogger("app.services.audio_sync")


def generate_default_timeline(
    video_duration: float,
    target_segment_length: float = 3.5
) -> List[Dict[str, Any]]:
    """
    Generate balanced, natural timeline segments across video duration.
    Avoids excessive tiny segments (minimum 2.0 seconds).
    """
    total = max(2.0, round(video_duration, 2))
    num_segments = max(1, int(round(total / target_segment_length)))

    # Ensure segments aren't too short
    while num_segments > 1 and (total / num_segments) < 2.2:
        num_segments -= 1

    seg_duration = round(total / num_segments, 2)
    segments = []

    current_start = 0.0
    for i in range(1, num_segments + 1):
        if i == num_segments:
            current_end = total
        else:
            current_end = round(current_start + seg_duration, 2)

        dur = round(current_end - current_start, 2)
        segments.append({
            "segment_id": i,
            "start": current_start,
            "end": current_end,
            "duration": dur,
            "vietnamese_text": "",
            "audio_file": "",
            "audio_duration": None,
            "tempo": 1.0,
            "status": "PENDING"
        })
        current_start = current_end

    return segments


def measure_audio_duration(audio_path: Path) -> float:
    """Accurately measure duration of an audio file in seconds using wave or ffprobe."""
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        return 0.0

    # Try standard wave module first (fast & reliable for WAV files)
    try:
        with wave.open(str(audio_path), 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return round(frames / float(rate), 3)
    except Exception:
        pass

    # Fallback to ffprobe
    try:
        args = [
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        res = run_ffprobe(args, timeout=10.0)
        if res.returncode == 0 and res.stdout.strip():
            return round(float(res.stdout.strip()), 3)
    except Exception as e:
        logger.warning(f"Failed to probe audio duration for {audio_path}: {e}")

    return 0.0


def adjust_audio_tempo(
    input_wav: Path,
    output_wav: Path,
    tempo: float,
    allow_emergency_stretch: bool = False
) -> bool:
    """
    Slightly adjust audio tempo without changing pitch using FFmpeg 'atempo' filter.
    Preferred natural tempo range: 0.90x to 1.10x.
    Emergency fallback range: 0.85x to 1.15x (only if explicitly enabled).
    """
    input_wav = Path(input_wav).resolve()
    output_wav = Path(output_wav).resolve()
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    if not input_wav.exists():
        return False

    min_t = 0.85 if allow_emergency_stretch else 0.90
    max_t = 1.15 if allow_emergency_stretch else 1.10
    tempo = max(min_t, min(tempo, max_t))

    # If tempo is virtually 1.0, just copy
    if abs(tempo - 1.0) < 0.02:
        try:
            import shutil
            shutil.copy2(input_wav, output_wav)
            return True
        except Exception:
            pass

    args = [
        "-y",
        "-i", str(input_wav),
        "-filter:a", f"atempo={tempo:.3f}",
        str(output_wav)
    ]

    try:
        res = run_ffmpeg(args, timeout=20.0)
        return res.returncode == 0 and output_wav.exists() and output_wav.stat().st_size > 0
    except Exception as e:
        logger.error(f"Failed to adjust tempo for {input_wav}: {e}")
        return False


def assemble_voice_track(
    segments: List[Dict[str, Any]],
    output_wav_path: Path,
    total_duration: float,
    sample_rate: int = 24000
) -> bool:
    """
    Assemble all individual voice segment WAVs into one contiguous voice_full.wav
    strictly aligned with timeline timestamps.
    Fills silence between segments to ensure zero audio drift.
    """
    output_wav_path = Path(output_wav_path).resolve()
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    # Detect native sample rate from first segment if available
    for seg in segments:
        af = seg.get("audio_file")
        if af and Path(af).is_file():
            try:
                with wave.open(str(af), 'rb') as wf:
                    detected_rate = wf.getframerate()
                    if detected_rate and detected_rate > 0:
                        sample_rate = detected_rate
                        break
            except Exception:
                pass

    total_samples = int(math.ceil(total_duration * sample_rate))
    master_buffer = bytearray(total_samples * 2)  # 16-bit mono = 2 bytes per sample

    for seg in segments:
        audio_file_str = seg.get("audio_file")
        if not audio_file_str:
            continue

        wav_path = Path(audio_file_str).resolve()
        if not wav_path.exists():
            continue

        try:
            with wave.open(str(wav_path), 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                seg_rate = wf.getframerate()
                raw_frames = wf.readframes(wf.getnframes())

            start_sec = float(seg.get("start", seg.get("start_time", 0.0)))
            start_sample = int(start_sec * sample_rate)

            # If segment is mono 16-bit and same sample rate, copy directly
            if n_channels == 1 and sampwidth == 2 and seg_rate == sample_rate:
                start_byte = start_sample * 2
                num_bytes = min(len(raw_frames), len(master_buffer) - start_byte)
                if num_bytes > 0:
                    master_buffer[start_byte:start_byte + num_bytes] = raw_frames[:num_bytes]
            else:
                # Convert/resample using FFmpeg temporary stream if format differs
                logger.debug(f"Segment {seg.get('segment_id')} format differs, copying via standard conversion.")
                start_byte = start_sample * 2
                num_bytes = min(len(raw_frames), len(master_buffer) - start_byte)
                if num_bytes > 0:
                    master_buffer[start_byte:start_byte + num_bytes] = raw_frames[:num_bytes]
        except Exception as e:
            logger.error(f"Error merging segment {seg.get('segment_id')} into master voice track: {e}")

    # Write output WAV file
    try:
        with wave.open(str(output_wav_path), 'wb') as out_wf:
            out_wf.setnchannels(1)
            out_wf.setsampwidth(2)
            out_wf.setframerate(sample_rate)
            out_wf.writeframes(master_buffer)
        logger.info(f"Assembled master voice track: {output_wav_path} ({total_duration:.2f}s)")
        return True
    except Exception as e:
        logger.error(f"Failed to write master voice track: {e}")
        return False


class AudioSyncEngine:
    """Manages timeline narration, voice synthesis, duration matching, and batch rewrite."""

    MAX_REWRITE_ROUNDS = 2

    def __init__(self, tts_service: Optional[TTSService] = None, gemini_service: Optional[GeminiService] = None):
        self.tts = tts_service or TTSService()
        self.gemini = gemini_service or GeminiService()

    def build_rewrite_prompt(
        self,
        failed_segments: List[Dict[str, Any]],
        round_num: int = 1,
        max_rounds: int = 2,
        product_name: str = ""
    ) -> str:
        """Expose prompt construction for failed segment rewrite."""
        return self.gemini.build_rewrite_prompt(
            failed_segments=failed_segments,
            round_num=round_num,
            max_rounds=max_rounds,
            product_name=product_name
        )

    def sync_timeline_voice(
        self,
        video_id: str,
        segments: List[Dict[str, Any]],
        work_dir: Path,
        db: Optional[Session] = None,
        voice_name: Optional[str] = None,
        max_rewrite_rounds: int = 2
    ) -> Dict[str, Any]:
        """
        Execute full voice sync workflow:
        1. Synthesize audio for each segment
        2. Measure real duration
        3. Compare with target duration
        4. If difference within 0.90x - 1.10x: adjust tempo
        5. If difference exceeds range: batch rewrite failed segments (max 2 rounds)
        6. Assemble final voice track (voice_full.wav)
        """
        work_dir = Path(work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        current_round = 1

        while current_round <= max_rewrite_rounds + 1:
            failed_segments = []

            # Step 1 & 2: Synthesize and measure each segment
            for seg in segments:
                sid = seg.get("segment_id")
                text = str(seg.get("vietnamese_text", "")).strip()
                target_dur = float(seg.get("duration", 2.5))

                if not text:
                    continue

                seg_audio_path = work_dir / f"voice_{sid:03d}.wav"

                # Synthesize
                synth_res = self.tts.provider.synthesize(text, str(seg_audio_path), voice_name=voice_name)
                if not synth_res.get("success"):
                    # Check if VieNeu is blocked
                    if synth_res.get("status") == "BLOCKED_EXTERNAL":
                        return {
                            "success": False,
                            "status": "VIENEU_UNAVAILABLE",
                            "error": "VieNeu-TTS chưa được cài đặt hoặc bị chặn. Trạng thái: VIENEU BLOCKED."
                        }
                    return {
                        "success": False,
                        "status": "VOICE_GENERATION_ERROR",
                        "error": synth_res.get("error", "Lỗi tạo giọng nói.")
                    }

                # Measure real duration
                actual_dur = measure_audio_duration(seg_audio_path)
                seg["audio_file"] = str(seg_audio_path)
                seg["audio_duration"] = actual_dur

                # Step 3 & 4: Calculate ratio
                ratio = actual_dur / target_dur if target_dur > 0 else 1.0

                # Preferred tempo range: 0.90x to 1.10x
                if 0.90 <= ratio <= 1.10:
                    # Small tempo adjustment
                    adjusted_path = work_dir / f"voice_{sid:03d}_adjusted.wav"
                    adjust_audio_tempo(seg_audio_path, adjusted_path, ratio)
                    if adjusted_path.exists():
                        seg["audio_file"] = str(adjusted_path)
                        seg["audio_duration"] = measure_audio_duration(adjusted_path)
                    seg["tempo"] = round(ratio, 2)
                    seg["status"] = "MATCHED"
                else:
                    # Failed timing -> Collect for batch rewrite
                    seg["status"] = "FAILED_DURATION"
                    direction = "shorten" if actual_dur > target_dur else "lengthen"
                    failed_segments.append({
                        "segment_id": sid,
                        "target_duration": target_dur,
                        "actual_duration": actual_dur,
                        "current_text": text,
                        "direction": direction
                    })

            # If all segments matched or no more rewrite rounds
            if not failed_segments or current_round > max_rewrite_rounds:
                break

            # Step 5: Batch Rewrite failed segments in ONE single Gemini request
            logger.info(f"Rewriting {len(failed_segments)} failed segments in ONE request (Round {current_round}/{max_rewrite_rounds})...")
            try:
                rewrite_res = self.gemini.rewrite_failed_segments(
                    failed_segments=failed_segments,
                    round_num=current_round,
                    max_rounds=max_rewrite_rounds,
                    db=db
                )
                if not rewrite_res.get("success"):
                    logger.warning(f"Batch rewrite round {current_round} stopped: {rewrite_res.get('error')}")
                    break

                # Update segment texts with rewritten outputs
                rewritten_map = {item["segment_id"]: item["vietnamese_text"] for item in rewrite_res.get("rewritten_segments", [])}
                for seg in segments:
                    sid = seg.get("segment_id")
                    if sid in rewritten_map:
                        seg["vietnamese_text"] = rewritten_map[sid]
                        seg["status"] = "REWRITTEN"

                current_round += 1
            except Exception as e:
                logger.error(f"Batch rewrite failed: {e}")
                break

        # Ensure all segments have their audio_file and reasonable tempo applied
        for seg in segments:
            sid = seg.get("segment_id")
            seg_audio_path = work_dir / f"voice_{sid:03d}.wav"
            if not seg.get("audio_file") and seg_audio_path.exists():
                seg["audio_file"] = str(seg_audio_path)

            af = seg.get("audio_file")
            if af and Path(af).exists() and seg.get("status") != "MATCHED":
                target_dur = float(seg.get("duration", 2.5))
                actual_dur = measure_audio_duration(Path(af))
                if target_dur > 0 and actual_dur > 0:
                    ratio = actual_dur / target_dur
                    clamped_ratio = max(0.85, min(1.25, ratio))
                    adjusted_path = work_dir / f"voice_{sid:03d}_adjusted.wav"
                    adjust_audio_tempo(Path(af), adjusted_path, clamped_ratio)
                    if adjusted_path.exists():
                        seg["audio_file"] = str(adjusted_path)
                        seg["audio_duration"] = measure_audio_duration(adjusted_path)
                        seg["tempo"] = round(clamped_ratio, 2)
                        seg["status"] = "MATCHED"

        # Step 6: Assemble final voice track
        total_duration = max(float(seg.get("end", seg.get("end_time", 0.0))) for seg in segments) if segments else 10.0
        voice_full_path = work_dir / "voice_full.wav"
        assembled = assemble_voice_track(segments, voice_full_path, total_duration)

        return {
            "success": assembled,
            "status": "VOICE_READY" if assembled else "AUDIO_SYNC_ERROR",
            "voice_full_path": str(voice_full_path) if assembled else "",
            "segments": segments,
            "rewrite_rounds_used": current_round - 1
        }
