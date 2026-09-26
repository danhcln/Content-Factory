import logging
import math
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from sqlalchemy.orm import Session
    from app.models import Setting
except ImportError:
    Session = Any
    Setting = None

logger = logging.getLogger("app.services.subtitle")


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp: HH:MM:SS,mmm"""
    total_ms = max(0, int(round(seconds * 1000)))
    hrs = total_ms // 3600000
    mins = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def format_ass_timestamp(seconds: float) -> str:
    """Format seconds into ASS timestamp: H:MM:SS.cc (centiseconds)"""
    total_cs = max(0, int(round(seconds * 100)))
    hrs = total_cs // 360000
    mins = (total_cs % 360000) // 6000
    secs = (total_cs % 6000) // 100
    cs = total_cs % 100
    return f"{hrs}:{mins:02d}:{secs:02d}.{cs:02d}"


def hex_to_ass_color(hex_str: str, alpha: int = 0) -> str:
    """
    Convert HTML color hex (e.g. #FFFFFF or #000000) to ASS &HAABBGGRR format.
    Alpha 0 = opaque, 255 = completely transparent.
    """
    clean_hex = hex_str.strip().lstrip("#").upper()
    if len(clean_hex) == 3:
        clean_hex = "".join([c * 2 for c in clean_hex])
    if len(clean_hex) != 6:
        clean_hex = "FFFFFF"

    r = clean_hex[0:2]
    g = clean_hex[2:4]
    b = clean_hex[4:6]
    a = f"{max(0, min(255, alpha)):02X}"
    return f"&H{a}{b}{g}{r}"


def split_segment_into_phrases(text: str, min_words: int = 2, max_words: int = 5) -> List[str]:
    """
    Split a segment sentence into short, natural phrase chunks suitable for mobile vertical short-form video.
    Breaks cleanly at natural punctuation and Vietnamese conjunctions/prepositions, ensuring 1 short line
    per subtitle event (3-5 words typically, max 6 words).
    """
    text = text.strip()
    if not text:
        return []

    # 1. Split by major punctuation marks
    punct_parts = re.split(r'([,;!?\.\n]+)', text)
    clauses: List[str] = []
    curr = ""
    for p in punct_parts:
        if not p:
            continue
        if re.match(r'^[,;!?\.\n]+$', p):
            curr += p.strip()
            if curr.strip():
                clauses.append(curr.strip())
            curr = ""
        else:
            if curr.strip():
                clauses.append(curr.strip())
            curr = p.strip()
    if curr.strip():
        clauses.append(curr.strip())

    # Prepositions, connectives and phrase boundaries where Vietnamese naturally pauses
    BREAK_WORDS = {
        'để', 'hãy', 'bấm ngay', 'nhấn ngay', 'vào giỏ hàng', 'vào giỏ', 'ở góc', 'bên góc',
        'siêu rẻ', 'cực kỳ', 'quá rẻ', 'bởi vì', 'vì thế', 'nếu bạn', 'hoặc ở',
        'giúp bạn', 'ăn xong', 'chỉ cần', 'để gọn', 'hay mang', 'mới thực sự',
        'nấu lẩu', 'hay nấu', 'vừa', 'luôn nóng', 'được tráng', 'lại siêu', 'chốt đơn'
    }

    final_phrases: List[str] = []

    for c in clauses:
        words = c.split()
        if len(words) <= max_words:
            final_phrases.append(c)
            continue

        sub_chunks: List[str] = []
        cur_words: List[str] = []
        i = 0
        while i < len(words):
            w = words[i]
            two_word = f"{words[i]} {words[i+1]}".lower() if i + 1 < len(words) else ""

            # Check if we should break before this point
            if len(cur_words) >= 3:
                if two_word in BREAK_WORDS:
                    sub_chunks.append(" ".join(cur_words))
                    cur_words = []
                elif w.lower() in BREAK_WORDS:
                    sub_chunks.append(" ".join(cur_words))
                    cur_words = []

            cur_words.append(w)
            if len(cur_words) >= max_words:
                sub_chunks.append(" ".join(cur_words))
                cur_words = []
            i += 1

        if cur_words:
            if len(cur_words) <= 2 and sub_chunks:
                # Merge short trailing words into previous chunk if not excessively long
                prev_words = sub_chunks[-1].split()
                if len(prev_words) + len(cur_words) <= max_words + 1:
                    sub_chunks[-1] = sub_chunks[-1] + " " + " ".join(cur_words)
                else:
                    sub_chunks.append(" ".join(cur_words))
            else:
                sub_chunks.append(" ".join(cur_words))

        final_phrases.extend(sub_chunks)

    return final_phrases


def generate_timed_phrases(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert timeline audio segments into short speech-synchronized subtitle phrases.
    Proportionally allocates the segment's synchronized duration according to word count,
    guaranteeing zero drift while showing only one short line at a time.
    """
    all_phrases: List[Dict[str, Any]] = []

    for seg in segments:
        text = str(seg.get("vietnamese_text", "")).strip()
        if not text:
            continue

        start_t = float(seg.get("start", seg.get("start_time", 0.0)))
        dur = float(seg.get("duration", 2.0))
        end_t = float(seg.get("end", seg.get("end_time", start_t + dur)))

        phrases = split_segment_into_phrases(text, min_words=2, max_words=5)
        if not phrases:
            continue

        # Count words per phrase for time weighting
        phrase_word_counts = [max(1, len(p.split())) for p in phrases]
        total_words = sum(phrase_word_counts)

        cur_t = start_t
        for p_text, w_count in zip(phrases, phrase_word_counts):
            p_dur = round((w_count / total_words) * (end_t - start_t), 3)
            p_end = round(cur_t + p_dur, 3)
            if p_text == phrases[-1]:
                p_end = end_t

            all_phrases.append({
                "text": p_text,
                "start": cur_t,
                "end": p_end,
                "duration": round(p_end - cur_t, 3),
                "words": p_text.split()
            })
            cur_t = p_end

    return all_phrases


class SubtitleService:
    """
    Generates synchronized, modern short-form vertical subtitles (SRT and ASS).
    Designed for 9:16 vertical video (1080x1920) with customizable fonts, colors,
    compact 1-line phrase segmentation, safe lower positioning (78%-88%), and subtle animations.
    """

    DEFAULT_FONT = "Arial Bold"
    DEFAULT_FONT_SIZE = 40  # Clean, compact mobile size on 1080x1920 (~1 line)
    DEFAULT_COLOR = "#FFFFFF"
    DEFAULT_OUTLINE_COLOR = "#000000"
    DEFAULT_OUTLINE_WIDTH = 2.2
    DEFAULT_SHADOW_DEPTH = 1.2
    DEFAULT_POSITION = "bottom"
    DEFAULT_CUSTOM_Y = 0.84
    DEFAULT_ANIMATION = "fade"

    SIZE_MAP = {
        "small": 34,
        "medium": 40,
        "large": 48
    }

    @classmethod
    def resolve_font_size(cls, size_val: Any) -> int:
        if isinstance(size_val, int) and size_val > 10:
            return size_val
        if isinstance(size_val, str):
            clean = size_val.strip().lower()
            if clean in cls.SIZE_MAP:
                return cls.SIZE_MAP[clean]
            if clean.isdigit():
                return int(clean)
        return cls.DEFAULT_FONT_SIZE

    @classmethod
    def resolve_margin_v(cls, position: str = "bottom", custom_pos_y: Optional[float] = None) -> int:
        pos_key = (position or "bottom").lower()
        if pos_key == "custom" and custom_pos_y is not None:
            clamped = max(0.2, min(0.95, float(custom_pos_y)))
            return int(round(1920 * (1.0 - clamped)))
        elif pos_key in ["lower_center", "lower-center", "lower"]:
            return 380  # ~80.2% height
        elif pos_key in ["center", "middle"]:
            return 880  # ~54.2% height
        else:
            # "bottom": safe lower area ~85.4% from top (avoids TikTok/Reels bottom caption bar)
            return 280

    @classmethod
    def get_verified_fonts(cls) -> List[Dict[str, Any]]:
        """
        Return verified installed fonts with full Vietnamese unicode diacritics support.
        Detects presence in Windows Fonts directory with safe fallbacks.
        """
        font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        installed_files = set()
        if font_dir.exists():
            try:
                installed_files = {f.name.lower() for f in font_dir.iterdir()}
            except Exception:
                pass

        candidate_fonts = [
            {"id": "Arial Bold", "name": "Arial Bold (Nổi bật, rõ nét)", "file": "arialbd.ttf"},
            {"id": "Arial", "name": "Arial (Chuẩn, dễ đọc)", "file": "arial.ttf"},
            {"id": "Segoe UI", "name": "Segoe UI (Hiện đại, tinh tế)", "file": "segoeui.ttf"},
            {"id": "Tahoma", "name": "Tahoma (Gọn gàng, thanh thoát)", "file": "tahoma.ttf"},
            {"id": "Calibri", "name": "Calibri (Mềm mại, tròn trịa)", "file": "calibri.ttf"},
            {"id": "Verdana", "name": "Verdana (Khoảng cách rộng)", "file": "verdana.ttf"},
        ]

        verified = []
        for cf in candidate_fonts:
            is_present = not installed_files or cf["file"] in installed_files
            verified.append({
                "id": cf["id"],
                "name": cf["name"],
                "available": is_present
            })

        return verified

    @classmethod
    def get_user_settings(cls, db: Optional[Session] = None) -> Dict[str, Any]:
        """Fetch saved subtitle preferences from DB Setting or environment defaults."""
        settings = {
            "font": cls.DEFAULT_FONT,
            "size": "medium",
            "font_size": cls.DEFAULT_FONT_SIZE,
            "color": cls.DEFAULT_COLOR,
            "outline_color": cls.DEFAULT_OUTLINE_COLOR,
            "outline_width": cls.DEFAULT_OUTLINE_WIDTH,
            "shadow_depth": cls.DEFAULT_SHADOW_DEPTH,
            "position": cls.DEFAULT_POSITION,
            "custom_y": cls.DEFAULT_CUSTOM_Y,
            "animation": cls.DEFAULT_ANIMATION,
            "voice_name": "Trúc Ly"
        }

        if db and Setting:
            try:
                keys = {
                    "subtitle_font": "font",
                    "subtitle_size": "size",
                    "subtitle_color": "color",
                    "subtitle_outline_color": "outline_color",
                    "subtitle_outline_width": "outline_width",
                    "subtitle_position": "position",
                    "subtitle_custom_y": "custom_y",
                    "subtitle_animation": "animation",
                    "voice_name": "voice_name"
                }
                records = db.query(Setting).filter(Setting.key.in_(list(keys.keys()))).all()
                for r in records:
                    field = keys.get(r.key)
                    if field and r.value:
                        if field in ["outline_width", "custom_y"]:
                            try:
                                settings[field] = float(r.value)
                            except ValueError:
                                pass
                        else:
                            settings[field] = r.value
                settings["font_size"] = cls.resolve_font_size(settings.get("size"))
            except Exception as e:
                logger.warning(f"Failed to read subtitle settings from DB: {e}")

        return settings

    @classmethod
    def save_user_settings(cls, db: Session, settings_data: Dict[str, Any]) -> bool:
        """Persist subtitle preferences into DB Setting table."""
        if not db or not Setting:
            return False
        try:
            key_map = {
                "font": ("subtitle_font", "Subtitle font family"),
                "size": ("subtitle_size", "Subtitle size preset (small/medium/large)"),
                "color": ("subtitle_color", "Subtitle text color hex"),
                "outline_color": ("subtitle_outline_color", "Subtitle outline color hex"),
                "outline_width": ("subtitle_outline_width", "Subtitle outline thickness"),
                "position": ("subtitle_position", "Subtitle position (bottom/lower_center/center/custom)"),
                "custom_y": ("subtitle_custom_y", "Subtitle custom vertical position (0.2-0.95)"),
                "animation": ("subtitle_animation", "Subtitle animation style (fade/karaoke/none)"),
                "voice_name": ("voice_name", "Selected VieNeu voice name")
            }
            for k, (db_key, desc) in key_map.items():
                if k in settings_data and settings_data[k] is not None:
                    val_str = str(settings_data[k]).strip()
                    rec = db.query(Setting).filter(Setting.key == db_key).first()
                    if not rec:
                        db.add(Setting(key=db_key, value=val_str, description=desc))
                    else:
                        rec.value = val_str
            db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save subtitle settings to DB: {e}")
            db.rollback()
            return False

    @classmethod
    def generate_srt(
        cls,
        segments: List[Dict[str, Any]],
        output_srt_path: Path,
        hook_text: Optional[str] = None,
        split_phrases: bool = False
    ) -> bool:
        """
        Generate a clean, UTF-8 encoded SRT subtitle file.
        When split_phrases=True, breaks narration into short speech-synchronized phrases.
        """
        output_srt_path = Path(output_srt_path)
        output_srt_path.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        counter = 1

        if split_phrases:
            timed_items = generate_timed_phrases(segments)
        else:
            timed_items = []
            for s in segments:
                txt = str(s.get("vietnamese_text", "")).strip()
                if txt:
                    st = float(s.get("start", s.get("start_time", 0.0)))
                    dur = float(s.get("duration", 2.0))
                    et = float(s.get("end", s.get("end_time", st + dur)))
                    timed_items.append({"text": txt, "start": st, "end": et})

        for item in timed_items:
            text = str(item.get("text", "")).strip()
            if not text:
                continue

            start_t = float(item.get("start", 0.0))
            end_t = float(item.get("end", start_t + 2.0))

            start_str = format_srt_timestamp(start_t)
            end_str = format_srt_timestamp(end_t)

            lines.append(str(counter))
            lines.append(f"{start_str} --> {end_str}")
            lines.append(text)
            lines.append("")
            counter += 1

        content = "\n".join(lines).strip() + "\n"
        try:
            with open(output_srt_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Generated SRT subtitles ({counter - 1} entries) at {output_srt_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write SRT subtitles: {e}")
            return False

    @classmethod
    def generate_ass(
        cls,
        segments: List[Dict[str, Any]],
        output_ass_path: Path,
        font_name: str = "Arial Bold",
        font_size: Any = 40,
        primary_color: str = "#FFFFFF",
        outline_color: str = "#000000",
        outline_width: float = 2.2,
        shadow_depth: float = 1.2,
        position: str = "bottom",
        custom_pos_y: Optional[float] = None,
        animation: str = "fade",
        hook_text: Optional[str] = None
    ) -> bool:
        """
        Generate Advanced SubStation Alpha (.ass) subtitle file for vertical 1080x1920 video.
        Pixel-perfect styling, safe lower margins (78%-88%), subtle fade animation,
        and natural short phrase pacing (1 short line at a time).
        """
        output_ass_path = Path(output_ass_path)
        output_ass_path.parent.mkdir(parents=True, exist_ok=True)

        is_bold = 1 if "bold" in font_name.lower() else 0
        clean_font = font_name.replace(" Bold", "").strip() or "Arial"

        resolved_size = cls.resolve_font_size(font_size)
        margin_v = cls.resolve_margin_v(position, custom_pos_y)

        pri_ass = hex_to_ass_color(primary_color, alpha=0)
        out_ass = hex_to_ass_color(outline_color, alpha=0)
        back_ass = hex_to_ass_color("#000000", alpha=140)

        ass_header = f"""[Script Info]
Title: AI Content Factory Vertical Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{clean_font},{resolved_size},{pri_ass},&H000000FF,{out_ass},{back_ass},{is_bold},0,0,0,100,100,0,0,1,{outline_width:.1f},{shadow_depth:.1f},2,50,50,{margin_v},1
Style: Hook,{clean_font},52,&H0000FFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3.0,2.0,8,60,60,320,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        events: List[str] = []

        if hook_text and hook_text.strip():
            clean_hook = hook_text.replace("\n", " ").replace("\\", "").strip()
            hook_start = format_ass_timestamp(0.0)
            hook_end = format_ass_timestamp(2.5)
            events.append(f"Dialogue: 1,{hook_start},{hook_end},Hook,,0,0,0,,{{\\fad(150,150)}}{clean_hook}")

        timed_phrases = generate_timed_phrases(segments)

        for item in timed_phrases:
            phrase_text = item["text"].replace("\n", " ").strip()
            if not phrase_text:
                continue

            start_str = format_ass_timestamp(item["start"])
            end_str = format_ass_timestamp(item["end"])

            anim_tag = ""
            anim_key = (animation or "fade").lower()
            if anim_key in ["fade", "subtle_fade"]:
                anim_tag = "{\\fad(80,80)}"
            elif anim_key == "karaoke":
                words = item.get("words", phrase_text.split())
                if words and item["duration"] > 0:
                    per_word_cs = max(10, int(round((item["duration"] * 100) / len(words))))
                    k_text = "".join([f"{{\\k{per_word_cs}}}{w} " for w in words]).strip()
                    anim_tag = "{\\fad(80,80)}"
                    phrase_text = k_text

            dialogue_line = f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{anim_tag}{phrase_text}"
            events.append(dialogue_line)

        full_content = ass_header + "\n".join(events) + "\n"

        try:
            with open(output_ass_path, "w", encoding="utf-8") as f:
                f.write(full_content)
            logger.info(f"Generated styled ASS subtitles ({len(events)} events) at {output_ass_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write ASS subtitles: {e}")
            return False

    @classmethod
    def build_subtitle_filter(
        cls,
        file_path: Path,
        font_size: int = 40,
        margin_v: int = 280,
        font_color: str = "&H00FFFFFF",
        outline_color: str = "&H00000000",
        outline_width: float = 2.2
    ) -> str:
        """
        Build an FFmpeg subtitles filter string.
        Supports both .ass and .srt files, using robust path escaping for Windows.
        """
        path_obj = Path(file_path).resolve()
        escaped_path = path_obj.as_posix().replace(":", r"\:")

        if path_obj.suffix.lower() == ".ass":
            return f"subtitles='{escaped_path}'"

        style_parts = [
            f"FontSize={font_size}",
            f"PrimaryColour={font_color}",
            f"OutlineColour={outline_color}",
            f"Outline={outline_width}",
            f"MarginV={margin_v}",
            "Alignment=2",
            "Bold=1"
        ]
        force_style = ",".join(style_parts)
        return f"subtitles='{escaped_path}':force_style='{force_style}'"

    @classmethod
    def build_hook_filter(
        cls,
        hook_text: str,
        video_width: int = 1080,
        video_height: int = 1920,
        duration: float = 2.5
    ) -> str:
        """Construct an FFmpeg drawtext filter string to overlay hook text during first seconds."""
        clean_text = hook_text.replace("'", "").replace(":", " ").replace('"', "").strip()
        if not clean_text:
            return ""

        y_pos = int(video_height * 0.18)
        font_size = 46

        return (
            f"drawtext=text='{clean_text}':"
            f"fontsize={font_size}:fontcolor=yellow:"
            f"borderw=3:bordercolor=black:"
            f"x=(w-text_w)/2:y={y_pos}:"
            f"enable='between(t,0,{duration:.1f})'"
        )

    @classmethod
    def build_hook_overlay_filter(
        cls,
        hook_text: str,
        video_width: int = 1080,
        video_height: int = 1920,
        duration_sec: float = 2.5
    ) -> str:
        return cls.build_hook_filter(hook_text, video_width=video_width, video_height=video_height, duration=duration_sec)
