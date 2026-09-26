import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
try:
    from sqlalchemy.orm import Session
except ImportError:
    Session = Any

try:
    from app.database import BASE_DIR, TEMP_DIR, FINAL_DIR, ORIGINAL_DIR
    from app.models import Video, Product, Setting
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    TEMP_DIR = BASE_DIR / "temp"
    FINAL_DIR = BASE_DIR / "downloads" / "final"
    ORIGINAL_DIR = BASE_DIR / "downloads" / "original"
    Video = None
    Product = None
    Setting = None
from app.services.ffmpeg_utils import run_ffmpeg, run_ffprobe, check_ffmpeg_available
from app.services.video_analyzer import VideoAnalyzer
from app.services.subtitle_region import (
    get_preset_region,
    validate_coordinates,
    build_blur_filter,
    build_cover_filter,
    SubtitleRegionDetector
)
from app.services.audio_sync import (
    generate_default_timeline,
    AudioSyncEngine,
    measure_audio_duration
)
from app.services.subtitle_service import SubtitleService
from app.services.gemini_service import GeminiService, GeminiQuotaExceededError

logger = logging.getLogger("app.services.auto_editor")


def validate_render_filter_graph(filter_complex: str) -> bool:
    """
    Validate that the FFmpeg filter complex string strictly enforces a single authoritative text path:
    Each subtitle/text phrase must be rendered exactly once.
    Rejects any configuration with duplicate subtitle filters or drawtext + subtitle conflicts.
    """
    sub_count = filter_complex.count("subtitles=") + filter_complex.count("ass=")
    if sub_count > 1:
        raise ValueError(f"Duplicate subtitle filters detected ({sub_count}) in filter_complex: {filter_complex}")
    if sub_count >= 1 and "drawtext=" in filter_complex:
        raise ValueError("Conflicting text overlay filters detected: both subtitle and drawtext filters are present")
    return True


class AutoEditorService:
    def __init__(self):
        self.gemini = GeminiService()
        self.sync_engine = AudioSyncEngine()

    def get_work_dir(self, video_id: str) -> Path:
        """Retrieve dedicated temporary working folder for video_id."""
        work_dir = TEMP_DIR / video_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def get_edit_plan_path(self, video_id: str) -> Path:
        return self.get_work_dir(video_id) / "edit_plan.json"

    def load_edit_plan(self, video_id: str) -> Dict[str, Any]:
        """Load edit_plan.json or initialize empty structure."""
        plan_path = self.get_edit_plan_path(video_id)
        if plan_path.exists():
            try:
                with open(plan_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read edit_plan.json for {video_id}: {e}")

        # Initialize default plan
        return {
            "video_id": video_id,
            "source": "",
            "duration": 0.0,
            "subtitle_region": get_preset_region("BOTTOM"),
            "cover": {
                "type": "blur",
                "strength": 10
            },
            "source_audio": "low",
            "output": {
                "width": 1080,
                "height": 1920
            },
            "segments": []
        }

    def save_edit_plan(self, video_id: str, plan: Dict[str, Any]) -> None:
        """Safely write edit_plan.json."""
        plan_path = self.get_edit_plan_path(video_id)
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

    def analyze_source(self, db_or_id: Any, id_or_db: Any = None, db: Optional[Session] = None) -> Dict[str, Any]:
        """Stage 1: Verify source video integrity and extract metadata + frame preview."""
        if isinstance(db_or_id, str):
            video_id = db_or_id
            session = id_or_db or db
        else:
            session = db_or_id or db
            video_id = id_or_db

        video = session.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            return {"success": False, "status": "INVALID_SOURCE_VIDEO", "error": f"Không tìm thấy video {video_id}"}

        # Resolve source video file
        src_path = None
        if video.local_file and Path(video.local_file).exists():
            src_path = Path(video.local_file)
        else:
            candidates = [
                ORIGINAL_DIR / f"{video_id}_original.mp4",
                ORIGINAL_DIR / f"{video_id}.mp4"
            ]
            for c in candidates:
                if c.exists():
                    src_path = c
                    break

        if not src_path or not src_path.exists():
            video.auto_edit_status = "INVALID_SOURCE_VIDEO"
            session.commit()
            return {"success": False, "error": f"Không tìm thấy tệp video gốc cho {video_id}."}

        # Analyze video
        analysis = VideoAnalyzer.analyze_video(src_path)
        if not analysis.get("valid"):
            video.auto_edit_status = analysis.get("status", "INVALID_SOURCE_VIDEO")
            video.render_error = analysis.get("error")
            session.commit()
            return {"success": False, "error": analysis.get("error")}

        # Extract representative frame preview
        work_dir = self.get_work_dir(video_id)
        preview_path = work_dir / "frame_preview.jpg"
        timestamp = min(2.0, max(0.5, analysis["duration"] * 0.2))
        VideoAnalyzer.extract_frame_preview(src_path, preview_path, timestamp_sec=timestamp)

        # Update edit plan
        plan = self.load_edit_plan(video_id)
        plan["source"] = str(src_path.resolve())
        plan["duration"] = analysis["duration"]
        plan["source_metadata"] = analysis
        self.save_edit_plan(video_id, plan)

        video.auto_edit_status = "ANALYZED"
        video.edit_plan_path = str(self.get_edit_plan_path(video_id))
        session.commit()

        return {
            "success": True,
            "status": "ANALYZED",
            "metadata": analysis,
            "preview_url": f"/temp/{video_id}/frame_preview.jpg" if preview_path.exists() else None
        }

    def detect_or_set_subtitle_region(
        self,
        db: Session,
        video_id: str,
        mode: str = "auto",
        custom_region: Optional[Dict[str, float]] = None,
        preset: str = "BOTTOM"
    ) -> Dict[str, Any]:
        """Stage 2: Configure Chinese subtitle region using Manual Preset, Custom, or Best-effort Auto Detect."""
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            return {"success": False, "error": f"Không tìm thấy video {video_id}"}

        plan = self.load_edit_plan(video_id)
        src_path = Path(plan.get("source", ""))
        if not src_path.exists():
            return {"success": False, "error": "Chưa hoàn thành phân tích video gốc."}

        if mode == "auto":
            det_res = SubtitleRegionDetector.auto_detect(src_path, duration=plan.get("duration", 10.0))
            region = det_res["region"]
            confidence = det_res["confidence"]

            plan["subtitle_region"] = {
                "mode": "auto",
                "x": region["x"],
                "y": region["y"],
                "width": region["width"],
                "height": region["height"],
                "confidence": confidence
            }
            self.save_edit_plan(video_id, plan)

            video.subtitle_region = json.dumps(plan["subtitle_region"])
            video.auto_edit_status = det_res["status"]
            db.commit()

            return {
                "success": det_res["detected"],
                "status": det_res["status"],
                "region": region,
                "confidence": confidence,
                "reason": det_res.get("reason")
            }
        else:
            # Manual / Preset
            if custom_region:
                val = validate_coordinates(custom_region)
                if not val["valid"]:
                    return {"success": False, "error": val["error"]}
                region = val["region"]
                chosen_mode = "custom"
            else:
                region = get_preset_region(preset)
                chosen_mode = preset.lower()

            plan["subtitle_region"] = {
                "mode": chosen_mode,
                "x": region["x"],
                "y": region["y"],
                "width": region["width"],
                "height": region["height"],
                "confidence": 1.0
            }
            self.save_edit_plan(video_id, plan)

            video.subtitle_region = json.dumps(plan["subtitle_region"])
            video.auto_edit_status = "SUBTITLE_REGION_CONFIRMED"
            db.commit()

            return {
                "success": True,
                "status": "SUBTITLE_REGION_CONFIRMED",
                "region": region,
                "confidence": 1.0
            }

    def generate_timed_script(self, db: Session, video_id: str) -> Dict[str, Any]:
        """Stage 3: Generate duration-aware Vietnamese narration for all timeline segments in ONE Gemini call."""
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            return {"success": False, "error": f"Không tìm thấy video {video_id}"}

        plan = self.load_edit_plan(video_id)
        duration = float(plan.get("duration", 0.0))
        if duration <= 0:
            return {"success": False, "error": "Thời lượng video không hợp lệ. Vui lòng phân tích lại video."}

        # Generate timeline if not present
        if not plan.get("segments"):
            plan["segments"] = generate_default_timeline(duration)

        product = video.product
        prod_info = {
            "name_vietnamese": product.name_vietnamese if product else "Sản phẩm",
            "content_angle": product.content_angle if product else "",
            "hook": product.hook if product else ""
        }

        try:
            res = self.gemini.generate_timed_script(
                video_duration=duration,
                segments=plan["segments"],
                product_info=prod_info,
                db=db
            )
            if not res.get("valid"):
                video.auto_edit_status = "AUTO_EDIT_SCRIPT_ERROR"
                db.commit()
                return {"success": False, "error": res.get("error", "Lỗi tạo kịch bản")}

            # Update segment texts
            plan["segments"] = res["segments"]
            self.save_edit_plan(video_id, plan)

            video.auto_edit_status = "TIMED_SCRIPT_READY"
            db.commit()

            return {
                "success": True,
                "status": "TIMED_SCRIPT_READY",
                "segments": plan["segments"]
            }
        except GeminiQuotaExceededError:
            video.auto_edit_status = "GEMINI_QUOTA_EXCEEDED"
            video.render_error = "Gemini daily quota has been reached. Try again after quota reset or use a project with sufficient quota."
            db.commit()
            return {
                "success": False,
                "status": "GEMINI_QUOTA_EXCEEDED",
                "error": "Gemini daily quota has been reached. Try again after quota reset or use a project with sufficient quota."
            }
        except Exception as e:
            video.auto_edit_status = "AUTO_EDIT_SCRIPT_ERROR"
            video.render_error = str(e)
            db.commit()
            return {"success": False, "error": f"Lỗi tạo kịch bản phân đoạn: {str(e)}"}

    def generate_voice_and_sync(
        self,
        db: Session,
        video_id: str,
        voice_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Stage 4: Synthesize voice per segment, match duration, batch rewrite, assemble full voice track."""
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            return {"success": False, "error": f"Không tìm thấy video {video_id}"}

        plan = self.load_edit_plan(video_id)
        segments = plan.get("segments", [])
        if not segments:
            return {"success": False, "error": "Chưa có kịch bản phân đoạn. Vui lòng tạo kịch bản trước."}

        work_dir = self.get_work_dir(video_id)

        user_prefs = SubtitleService.get_user_settings(db)
        target_voice = voice_name or plan.get("voice_name") or user_prefs.get("voice_name") or "Trúc Ly"

        sync_res = self.sync_engine.sync_timeline_voice(
            video_id=video_id,
            segments=segments,
            work_dir=work_dir,
            db=db,
            voice_name=target_voice,
            max_rewrite_rounds=2
        )

        if not sync_res.get("success"):
            video.auto_edit_status = sync_res.get("status", "AUTO_EDIT_VOICE_ERROR")
            video.render_error = sync_res.get("error")
            db.commit()
            return sync_res

        # Update edit plan
        plan["segments"] = sync_res.get("segments", segments)
        plan["voice_full"] = sync_res.get("voice_full_path")
        plan["voice_name"] = target_voice
        self.save_edit_plan(video_id, plan)

        video.auto_edit_status = "VOICE_READY"
        db.commit()

        return {
            "success": True,
            "status": "VOICE_READY",
            "voice_name": target_voice,
            "voice_full": sync_res.get("voice_full_path"),
            "segments": plan["segments"]
        }

    def generate_subtitles(
        self,
        db: Session,
        video_id: str,
        font_name: Optional[str] = None,
        font_size: Optional[Any] = None,
        primary_color: Optional[str] = None,
        outline_color: Optional[str] = None,
        outline_width: Optional[float] = None,
        position: Optional[str] = None,
        custom_pos_y: Optional[float] = None,
        animation: Optional[str] = None
    ) -> Dict[str, Any]:
        """Stage 5: Generate speech-synchronized ASS and SRT subtitles tied to timeline segments."""
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            return {"success": False, "error": f"Không tìm thấy video {video_id}"}

        plan = self.load_edit_plan(video_id)
        segments = plan.get("segments", [])
        if not segments:
            return {"success": False, "error": "Không có phân đoạn kịch bản nào để tạo phụ đề."}

        work_dir = self.get_work_dir(video_id)
        srt_path = work_dir / f"{video_id}_vi.srt"
        ass_path = work_dir / f"{video_id}_vi.ass"

        user_prefs = SubtitleService.get_user_settings(db)
        resolved_font = font_name or plan.get("subtitle_font") or user_prefs.get("font", "Arial Bold")
        resolved_size = font_size if font_size is not None else plan.get("subtitle_size") or user_prefs.get("size", "medium")
        resolved_color = primary_color or plan.get("subtitle_color") or user_prefs.get("color", "#FFFFFF")
        resolved_outline = outline_color or plan.get("subtitle_outline_color") or user_prefs.get("outline_color", "#000000")
        resolved_width = outline_width if outline_width is not None else plan.get("subtitle_outline_width") or user_prefs.get("outline_width", 2.2)
        resolved_pos = position or plan.get("subtitle_position") or user_prefs.get("position", "bottom")
        resolved_custom_y = custom_pos_y if custom_pos_y is not None else plan.get("subtitle_custom_y") or user_prefs.get("custom_y", 0.84)
        resolved_anim = animation or plan.get("subtitle_animation") or user_prefs.get("animation", "fade")

        hook_text = video.product.hook if video.product else None

        # Generate styled ASS file
        ok_ass = SubtitleService.generate_ass(
            segments=segments,
            output_ass_path=ass_path,
            font_name=resolved_font,
            font_size=resolved_size,
            primary_color=resolved_color,
            outline_color=resolved_outline,
            outline_width=float(resolved_width),
            position=resolved_pos,
            custom_pos_y=float(resolved_custom_y) if resolved_custom_y is not None else None,
            animation=resolved_anim,
            hook_text=hook_text
        )

        # Generate synchronized SRT file as well
        ok_srt = SubtitleService.generate_srt(segments, srt_path, hook_text=hook_text, split_phrases=True)

        if not (ok_ass or ok_srt):
            video.auto_edit_status = "SUBTITLE_GENERATION_ERROR"
            db.commit()
            return {"success": False, "error": "Không thể tạo tệp phụ đề."}

        plan["srt_file"] = str(srt_path.resolve())
        plan["ass_file"] = str(ass_path.resolve())
        plan["subtitle_settings"] = {
            "font": resolved_font,
            "size": resolved_size,
            "color": resolved_color,
            "outline_color": resolved_outline,
            "outline_width": resolved_width,
            "position": resolved_pos,
            "custom_y": resolved_custom_y,
            "animation": resolved_anim
        }
        self.save_edit_plan(video_id, plan)

        video.auto_edit_status = "SUBTITLES_READY"
        db.commit()

        return {
            "success": True,
            "status": "SUBTITLES_READY",
            "srt_path": str(srt_path.resolve()),
            "ass_path": str(ass_path.resolve()),
            "settings": plan["subtitle_settings"]
        }

    def _build_render_args(
        self,
        source_path: Path,
        voice_wav_path: Optional[Path],
        srt_path: Optional[Path],
        temp_output: Path,
        duration: float,
        region: Dict[str, float],
        cover_type: str = "blur",
        blur_strength: int = 10,
        source_audio_mode: str = "low",
        hook_text: Optional[str] = None
    ) -> List[str]:
        """Construct the FFmpeg command line arguments for vertical 9:16 rendering."""
        target_w = 1080
        target_h = 1920

        # Video filter chain
        scale_pad = (
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
        )
        filter_steps = [f"[0:v]{scale_pad}[scaled]"]

        crop_x = int(target_w * region.get("x", 0.05))
        crop_y = int(target_h * region.get("y", 0.72))
        crop_w = int(target_w * region.get("width", 0.90))
        crop_h = int(target_h * region.get("height", 0.20))
        if crop_w % 2 != 0: crop_w += 1
        if crop_h % 2 != 0: crop_h += 1

        if cover_type == "cover":
            cover_f = build_cover_filter(region, target_w, target_h, opacity=0.85)
            filter_steps.append(f"[scaled]{cover_f}[v_cleaned]")
        else:
            filter_steps.extend([
                f"[scaled]split[base][subzone]",
                f"[subzone]crop=w={crop_w}:h={crop_h}:x={crop_x}:y={crop_y},boxblur=luma_radius={blur_strength}:luma_power=2[blurred_sub]",
                f"[base][blurred_sub]overlay=x={crop_x}:y={crop_y}[v_cleaned]"
            ])

        last_v = "[v_cleaned]"

        # Single authoritative text overlay path:
        # If subtitles exist, burn them in once. Do NOT append secondary drawtext overlays.
        if srt_path and Path(srt_path).exists() and Path(srt_path).stat().st_size > 0:
            sub_path_obj = Path(srt_path)
            if sub_path_obj.suffix.lower() == ".ass":
                sub_filter = SubtitleService.build_subtitle_filter(sub_path_obj)
            else:
                sub_filter = SubtitleService.build_subtitle_filter(sub_path_obj, font_size=42, margin_v=260)
            filter_steps.append(f"{last_v}{sub_filter}[v_subbed]")
            last_v = "[v_subbed]"
        elif hook_text:
            # Standalone hook text only when no subtitles are present
            hook_filter = SubtitleService.build_hook_filter(hook_text, target_w, target_h, duration=2.5)
            if hook_filter:
                filter_steps.append(f"{last_v}{hook_filter}[v_final]")
                last_v = "[v_final]"

        video_filter_complex = ";".join(filter_steps)
        # Pre-render verification: assert no duplicate text/subtitle filters exist
        validate_render_filter_graph(video_filter_complex)

        cmd_inputs = ["-y", "-i", str(source_path)]
        audio_map = []
        has_voice = voice_wav_path and Path(voice_wav_path).exists() and Path(voice_wav_path).stat().st_size > 100

        if has_voice:
            cmd_inputs.extend(["-i", str(voice_wav_path)])
            if source_audio_mode != "off":
                vol = 0.15 if source_audio_mode == "low" else 0.40
                audio_filter = f"[0:a]volume={vol}[bg];[1:a]volume=1.0[vc];[bg][vc]amix=inputs=2:duration=first[aout]"
                full_filter_complex = f"{video_filter_complex};{audio_filter}"
                audio_map = ["-map", "[aout]"]
            else:
                audio_filter = "[1:a]aresample=44100[aout]"
                full_filter_complex = f"{video_filter_complex};{audio_filter}"
                audio_map = ["-map", "[aout]"]
        else:
            full_filter_complex = video_filter_complex
            if source_audio_mode != "off":
                audio_map = ["-map", "0:a"]
            else:
                audio_map = ["-an"]

        args = cmd_inputs + [
            "-filter_complex", full_filter_complex,
            "-map", last_v.strip("[]") if "[" not in last_v else last_v,
        ] + audio_map + [
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(temp_output)
        ]
        return args

    def render_final_video(
        self,
        db_or_id: Any = None,
        id_or_db: Any = None,
        video_id: Optional[str] = None,
        db: Optional[Session] = None,
        cover_type: str = "blur",
        blur_strength: int = 10,
        source_audio: str = "low",
        include_subtitles: bool = True,
        include_hook: bool = True
    ) -> Dict[str, Any]:
        """
        Stage 6 & 7: Construct FFmpeg filter graph, render 9:16 vertical video to temp,
        validate output with ffprobe, and atomically move to downloads/final/{video_id}.mp4.
        """
        if isinstance(db_or_id, str):
            v_id = db_or_id
            session = id_or_db or db
        else:
            session = db_or_id or db
            v_id = id_or_db or video_id

        video = session.query(Video).filter(Video.video_id == v_id).first()
        if not video:
            return {"success": False, "error": f"Không tìm thấy video {v_id}"}

        plan = self.load_edit_plan(v_id)
        src_str = plan.get("source", "")
        src_path = None
        if src_str and Path(src_str).is_file():
            src_path = Path(src_str)
        elif video.local_file and Path(video.local_file).is_file():
            src_path = Path(video.local_file)
        else:
            candidates = [
                ORIGINAL_DIR / f"{v_id}.mp4",
                ORIGINAL_DIR / f"{v_id}_original.mp4",
            ]
            for c in candidates:
                if c.is_file():
                    src_path = c
                    break

        if not src_path or not src_path.is_file():
            return {"success": False, "error": f"Tệp video gốc không tồn tại: {src_path or v_id}"}

        work_dir = self.get_work_dir(v_id)
        temp_render_file = work_dir / "rendering.mp4"
        final_dest_file = FINAL_DIR / f"{v_id}.mp4"

        # Check FFmpeg availability
        ff_chk = check_ffmpeg_available()
        if not ff_chk.get("ready"):
            return {
                "success": False,
                "status": "FFMPEG_NOT_FOUND",
                "error": "FFmpeg hoặc ffprobe không sẵn sàng trên hệ thống."
            }

        target_w = 1080
        target_h = 1920

        # Subtitle region
        region_data = plan.get("subtitle_region", get_preset_region("BOTTOM"))

        # Find subtitle file (prefer ASS for pixel-perfect styled vertical subtitles, fallback to SRT)
        sub_file = None
        if include_subtitles:
            ass_cand = work_dir / f"{v_id}_vi.ass"
            srt_cand = work_dir / f"{v_id}_vi.srt"
            if ass_cand.exists() and ass_cand.stat().st_size > 0:
                sub_file = ass_cand
            elif srt_cand.exists() and srt_cand.stat().st_size > 0:
                sub_file = srt_cand
            elif (work_dir / "subtitles.srt").exists():
                sub_file = work_dir / "subtitles.srt"

        # Voice file
        voice_file = work_dir / "voice_full.wav"

        # Hook text
        hook_text = video.product.hook if video.product else None

        args = self._build_render_args(
            source_path=src_path,
            voice_wav_path=voice_file if voice_file.exists() else None,
            srt_path=sub_file,
            temp_output=temp_render_file,
            duration=plan.get("duration", 0.0),
            region=region_data,
            cover_type=cover_type,
            blur_strength=blur_strength,
            source_audio_mode=source_audio,
            hook_text=hook_text if include_hook else None
        )

        logger.info(f"Starting FFmpeg final render for {v_id} to temporary file {temp_render_file}...")
        video.auto_edit_status = "AUTO_EDIT_RENDERING"
        session.commit()

        try:
            render_res = run_ffmpeg(args, timeout=300.0)
            if render_res.returncode != 0:
                err = render_res.stderr[-400:] if render_res.stderr else "Lỗi render FFmpeg không xác định"
                video.auto_edit_status = "AUTO_EDIT_RENDER_ERROR"
                video.render_error = err
                session.commit()
                return {"success": False, "error": f"Lỗi render FFmpeg: {err}"}
        except Exception as e:
            video.auto_edit_status = "AUTO_EDIT_RENDER_ERROR"
            video.render_error = str(e)
            session.commit()
            return {"success": False, "error": f"FFmpeg render thất bại: {str(e)}"}

        # Stage 7: Final Video Validation
        validation = VideoAnalyzer.analyze_video(temp_render_file)
        if not validation.get("valid"):
            video.auto_edit_status = "AUTO_EDIT_VALIDATION_ERROR"
            video.render_error = validation.get("error", "Kiểm định tệp xuất thất bại")
            session.commit()
            return {"success": False, "error": validation.get("error")}

        # Check required dimensions
        if validation["width"] != target_w or validation["height"] != target_h:
            video.auto_edit_status = "AUTO_EDIT_VALIDATION_ERROR"
            video.render_error = f"Kích thước video ({validation['width']}x{validation['height']}) không khớp mục tiêu 1080x1920."
            session.commit()
            return {"success": False, "error": video.render_error}

        # Atomic move to downloads/final/{video_id}.mp4
        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if final_dest_file.exists():
                final_dest_file.unlink()
            shutil.move(str(temp_render_file), str(final_dest_file))
        except Exception as me:
            logger.error(f"Failed to move final video: {me}")
            return {"success": False, "error": f"Không thể di chuyển video vào downloads/final/: {me}"}

        # Update database record
        video.status = "AUTO_EDIT_READY"
        video.auto_edit_status = "AUTO_EDIT_READY"
        video.final_video_path = f"downloads/final/{v_id}.mp4"
        video.render_error = None
        session.commit()

        logger.info(f"Video {v_id} rendered and validated successfully: {final_dest_file}")

        return {
            "success": True,
            "status": "AUTO_EDIT_READY",
            "video_id": v_id,
            "final_path": str(final_dest_file.resolve()),
            "validation": validation
        }

    def update_status(self, video_id: str, status: str, db: Optional[Session] = None) -> None:
        """Update auto_edit_status on Video model."""
        session = db or SessionLocal()
        close_session = db is None
        try:
            v = session.query(Video).filter(Video.video_id == video_id).first()
            if v:
                v.auto_edit_status = status
                session.commit()
        finally:
            if close_session:
                session.close()

    def switch_to_capcut_manual(
        self,
        db_or_id: Any,
        id_or_db: Any = None,
        reason: Optional[str] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """Switch edit mode to CapCut Manual as a guaranteed fallback."""
        if isinstance(db_or_id, str):
            v_id = db_or_id
            session = id_or_db or db
        else:
            session = db_or_id or db
            v_id = id_or_db

        video = session.query(Video).filter(Video.video_id == v_id).first()
        if not video:
            return {"success": False, "error": f"Không tìm thấy video {v_id}"}

        video.edit_mode = "CAPCUT_MANUAL"
        video.auto_edit_status = "FALLBACK_TO_CAPCUT"
        if reason:
            video.render_error = f"Chuyển CapCut: {reason}"
        session.commit()

        return {
            "success": True,
            "video_id": v_id,
            "edit_mode": "CAPCUT_MANUAL",
            "message": "Đã chuyển sang chế độ biên tập thủ công bằng CapCut."
        }

    def generate_subtitle_preview_frame(
        self,
        db: Session,
        video_id: str,
        font_name: Optional[str] = None,
        font_size: Optional[Any] = None,
        primary_color: Optional[str] = None,
        outline_color: Optional[str] = None,
        outline_width: Optional[float] = None,
        position: Optional[str] = None,
        custom_pos_y: Optional[float] = None,
        animation: Optional[str] = None,
        sample_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Render a 1080x1920 preview frame with the exact ASS subtitle styling burned onto it via FFmpeg.
        Allows instant visual verification of font, size, color, outline and position.
        """
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            return {"success": False, "error": f"Không tìm thấy video {video_id}"}

        work_dir = self.get_work_dir(video_id)
        plan = self.load_edit_plan(video_id)

        # Source frame
        frame_jpg = work_dir / "frame_preview.jpg"
        if not frame_jpg.exists():
            src_str = plan.get("source", "")
            src_path = Path(src_str) if src_str and Path(src_str).is_file() else (ORIGINAL_DIR / f"{video_id}_original.mp4")
            if src_path.is_file():
                VideoAnalyzer.extract_frame_preview(src_path, frame_jpg, timestamp_sec=2.5)

        user_prefs = SubtitleService.get_user_settings(db)
        resolved_font = font_name or plan.get("subtitle_font") or user_prefs.get("font", "Arial Bold")
        resolved_size = font_size if font_size is not None else plan.get("subtitle_size") or user_prefs.get("size", "medium")
        resolved_color = primary_color or plan.get("subtitle_color") or user_prefs.get("color", "#FFFFFF")
        resolved_outline = outline_color or plan.get("subtitle_outline_color") or user_prefs.get("outline_color", "#000000")
        resolved_width = outline_width if outline_width is not None else plan.get("subtitle_outline_width") or user_prefs.get("outline_width", 2.2)
        resolved_pos = position or plan.get("subtitle_position") or user_prefs.get("position", "bottom")
        resolved_custom_y = custom_pos_y if custom_pos_y is not None else plan.get("subtitle_custom_y") or user_prefs.get("custom_y", 0.84)
        resolved_anim = animation or plan.get("subtitle_animation") or user_prefs.get("animation", "fade")

        preview_text = sample_text or "Hôm nay giá đang có ưu đãi"
        preview_segments = [{
            "segment_id": 1,
            "start": 0.0,
            "end": 3.0,
            "vietnamese_text": preview_text
        }]
        preview_ass = work_dir / "preview_sub.ass"
        SubtitleService.generate_ass(
            segments=preview_segments,
            output_ass_path=preview_ass,
            font_name=resolved_font,
            font_size=resolved_size,
            primary_color=resolved_color,
            outline_color=resolved_outline,
            outline_width=float(resolved_width),
            position=resolved_pos,
            custom_pos_y=float(resolved_custom_y) if resolved_custom_y is not None else None,
            animation=resolved_anim
        )

        out_preview = work_dir / "sub_preview.jpg"
        escaped_ass = preview_ass.as_posix().replace(":", r"\:")

        if frame_jpg.exists():
            input_arg = ["-i", str(frame_jpg)]
            vf = f"scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,subtitles='{escaped_ass}'"
        else:
            input_arg = ["-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1"]
            vf = f"subtitles='{escaped_ass}'"

        cmd = ["-y"] + input_arg + ["-vf", vf, "-vframes", "1", str(out_preview)]
        res = run_ffmpeg(cmd, timeout=30.0)
        if res.returncode == 0 and out_preview.exists():
            return {
                "success": True,
                "preview_url": f"/temp/{video_id}/sub_preview.jpg?t={int(time.time())}"
            }
        else:
            return {"success": False, "error": f"FFmpeg error: {res.stderr[-200:] if res.stderr else 'unknown'}"}


