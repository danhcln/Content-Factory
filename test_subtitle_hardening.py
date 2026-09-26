import sys
import unittest
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.services.auto_editor import validate_render_filter_graph
from app.services.subtitle_service import (
    SubtitleService,
    split_segment_into_phrases,
    generate_timed_phrases,
    hex_to_ass_color
)


class TestSubtitleHardening(unittest.TestCase):
    def test_01_single_subtitle_filter_passes(self):
        """Verify that a single subtitle filter passes validation."""
        filt = "[v_cleaned]subtitles='temp/sub.ass'[v_subbed]"
        self.assertTrue(validate_render_filter_graph(filt))

    def test_02_duplicate_subtitle_filters_rejected(self):
        """Verify that duplicate subtitle filters are rejected."""
        filt = "[v_cleaned]subtitles='sub1.ass'[v1];[v1]subtitles='sub2.ass'[v2]"
        with self.assertRaises(ValueError) as ctx:
            validate_render_filter_graph(filt)
        self.assertIn("Duplicate subtitle filters", str(ctx.exception))

    def test_03_drawtext_and_subtitles_conflict_rejected(self):
        """Verify that both drawtext and subtitle filter together are rejected."""
        filt = "[v_cleaned]subtitles='sub.ass'[v1];[v1]drawtext=text='Hook'[v2]"
        with self.assertRaises(ValueError) as ctx:
            validate_render_filter_graph(filt)
        self.assertIn("Conflicting text overlay filters", str(ctx.exception))

    def test_04_standalone_drawtext_without_subtitles_allowed(self):
        """Verify that standalone drawtext without subtitles passes validation."""
        filt = "[v_cleaned]drawtext=text='Title'[v_final]"
        self.assertTrue(validate_render_filter_graph(filt))

    def test_05_generate_ass_no_yellow_hook(self):
        """Verify generated ASS file contains no yellow Hook style and renders white dialogue."""
        segments = [
            {"start": 0.0, "end": 2.5, "vietnamese_text": "Nồi cơm điện mini đa năng dành cho sinh viên."}
        ]
        out_ass = Path("temp/test_no_yellow.ass")
        out_ass.parent.mkdir(parents=True, exist_ok=True)

        ok = SubtitleService.generate_ass(
            segments=segments,
            output_ass_path=out_ass,
            hook_text="Hook caption that should not be duplicated"
        )
        self.assertTrue(ok)
        self.assertTrue(out_ass.exists())

        content = out_ass.read_text(encoding="utf-8")
        # Assert NO Style: Hook with yellow hex (&H0000FFFF)
        self.assertNotIn("Style: Hook", content)
        self.assertNotIn("&H0000FFFF", content)
        # Assert Default style is white text (&H00FFFFFF)
        self.assertIn("&H00FFFFFF", content)
        # Clean up
        if out_ass.exists():
            out_ass.unlink()

    def test_06_vietnamese_diacritics_in_ass(self):
        """Verify full Vietnamese Unicode diacritics are properly preserved in ASS output."""
        special_chars = ["ă", "â", "ê", "ô", "ơ", "ư", "đ", "Á", "À", "Ả", "Ã", "Ạ", "ắ", "ằ", "ẳ", "ẵ", "ặ", "Nồi", "cơm", "điện"]
        test_text = " ".join(special_chars)
        segments = [{"start": 0.0, "end": 3.0, "vietnamese_text": test_text}]
        out_ass = Path("temp/test_vietnamese_diacritics.ass")
        out_ass.parent.mkdir(parents=True, exist_ok=True)

        ok = SubtitleService.generate_ass(segments=segments, output_ass_path=out_ass)
        self.assertTrue(ok)

        content = out_ass.read_text(encoding="utf-8")
        for ch in special_chars:
            self.assertIn(ch, content)
        if out_ass.exists():
            out_ass.unlink()

    def test_07_phrase_segmentation_avoids_paragraphs(self):
        """Verify long narration is broken into short, readable phrases."""
        long_sentence = (
            "Hôm nay giá đang có ưu đãi siêu rẻ luôn, "
            "hãy bấm ngay vào giỏ hàng bên góc trái để chốt đơn nhé!"
        )
        phrases = split_segment_into_phrases(long_sentence, min_words=3, max_words=6)
        self.assertGreaterEqual(len(phrases), 3)
        for p in phrases:
            # Each phrase should be compact (< 8 words)
            self.assertLessEqual(len(p.split()), 7)


if __name__ == "__main__":
    unittest.main()
