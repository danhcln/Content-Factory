import json
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.database import SessionLocal, init_db
from app.models import Product, Video, Voice, Content, Setting
from app.services.gemini_service import (
    GeminiService,
    GeminiQuotaExceededError,
    GeminiRateLimitError,
    is_daily_quota_error,
    validate_timed_script,
    validate_rewritten_segments,
    clean_json_response
)
from app.services.script_service import ScriptService
from app.services.content_service import ContentService
from app.services.usage_tracker import GeminiUsageTracker


class TestGeminiHardening(unittest.TestCase):
    def setUp(self):
        init_db()
        self.db = SessionLocal()
        self.gemini = GeminiService()
        self.script_service = ScriptService()
        self.content_service = ContentService()

    def tearDown(self):
        self.db.close()

    # ==========================================================
    # 1. BATCH PRODUCT PARSER & RESEARCH
    # ==========================================================
    def test_batch_product_research_one_call(self):
        """Verify product research generates N products in exactly ONE Gemini request."""
        mock_response = json.dumps([
            {
                "name_vietnamese": f"Sản phẩm {i}",
                "name_chinese": f"产品 {i}",
                "douyin_keywords": f"抖音好物 {i}",
                "content_angle": f"Góc quay {i}",
                "hook": f"Hook mở đầu {i}"
            }
            for i in range(1, 11)
        ])

        with patch.object(self.gemini, "call_gemini", return_value=mock_response) as mock_call:
            products = self.gemini.generate_products(niche="Gia dụng", count=10, db=self.db)
            self.assertEqual(len(products), 10)
            self.assertEqual(mock_call.call_count, 1, "Must make exactly ONE Gemini call for all 10 products")
            self.assertEqual(products[0]["name_vietnamese"], "Sản phẩm 1")
            self.assertEqual(products[9]["name_vietnamese"], "Sản phẩm 10")

    # ==========================================================
    # 2. BATCH TIMED SCRIPT ARCHITECTURE (PHASE 12 PREPARATION)
    # ==========================================================
    def test_batch_timed_script_architecture_one_call(self):
        """Verify timed script sends the complete timeline in ONE Gemini call and validates locally."""
        input_segments = [
            {"segment_id": 1, "start_time": 0.0, "end_time": 2.5, "duration": 2.5, "description": "Cận cảnh mở hộp"},
            {"segment_id": 2, "start_time": 2.5, "end_time": 5.8, "duration": 3.3, "description": "Nhấn nút bật tính năng"},
            {"segment_id": 3, "start_time": 5.8, "end_time": 9.0, "duration": 3.2, "description": "Hiệu quả tức thì"},
            {"segment_id": 4, "start_time": 9.0, "end_time": 13.0, "duration": 4.0, "description": "So sánh với đồ cũ"},
            {"segment_id": 5, "start_time": 13.0, "end_time": 20.0, "duration": 7.0, "description": "Kêu gọi bấm vào giỏ hàng"}
        ]

        mock_ai_json = {
            "segments": [
                {"segment_id": 1, "vietnamese_text": "Đừng vội mua máy xay nếu chưa biết chiếc máy này!"},
                {"segment_id": 2, "vietnamese_text": "Chỉ cần nhấn một nút, lưỡi dao sáu cánh tự động xay nhuyễn."},
                {"segment_id": 3, "vietnamese_text": "Chưa đến mười giây là đã có ngay ly sinh tố thơm ngon."},
                {"segment_id": 4, "vietnamese_text": "Không còn cảnh dây cắm lằng nhằng hay cối xay cồng kềnh khó rửa nữa."},
                {"segment_id": 5, "vietnamese_text": "Mang đi làm hay đi tập cực tiện. Xem ngay góc trái màn hình nhé cả nhà!"}
            ]
        }

        with patch.object(self.gemini, "call_gemini", return_value=json.dumps(mock_ai_json)) as mock_call:
            res = self.gemini.generate_timed_script(
                video_duration=20.0,
                segments=input_segments,
                product_info={"name_vietnamese": "Máy xay mini cầm tay", "content_angle": "Tiện lợi", "hook": "Đừng vội mua"},
                db=self.db
            )
            self.assertEqual(mock_call.call_count, 1, "Must make exactly ONE Gemini call for entire timeline")
            self.assertTrue(res["valid"])
            self.assertEqual(len(res["segments"]), 5)
            self.assertEqual(res["segments"][0]["segment_id"], 1)
            self.assertEqual(res["segments"][4]["segment_id"], 5)

    def test_batch_timed_script_local_validation_catches_missing_segment(self):
        """Local Python validator must catch missing segments without calling Gemini again."""
        input_segments = [
            {"segment_id": 1, "start_time": 0.0, "end_time": 2.5, "duration": 2.5},
            {"segment_id": 2, "start_time": 2.5, "end_time": 5.0, "duration": 2.5}
        ]
        incomplete_ai_output = {
            "segments": [
                {"segment_id": 1, "vietnamese_text": "Lời thoại phân đoạn 1"}
                # segment 2 is missing!
            ]
        }
        val = validate_timed_script(input_segments, incomplete_ai_output)
        self.assertFalse(val["valid"])
        self.assertIn("segment_id=2", val["error"])

    # ==========================================================
    # 3. BATCH FAILED-SEGMENT REWRITE (MAX 2 ROUNDS)
    # ==========================================================
    def test_batch_failed_segment_rewrite_one_call(self):
        """Verify failed segments are sent together in ONE request."""
        failed_segments = [
            {"segment_id": 1, "target_duration": 2.5, "actual_duration": 3.8, "current_text": "Câu quá dài cần rút ngắn", "direction": "shorten"},
            {"segment_id": 3, "target_duration": 3.2, "actual_duration": 4.5, "current_text": "Câu thứ hai cũng dài", "direction": "shorten"}
        ]

        mock_rewrite_json = {
            "rewritten_segments": [
                {"segment_id": 1, "vietnamese_text": "Rút gọn câu một."},
                {"segment_id": 3, "vietnamese_text": "Rút gọn câu ba."}
            ]
        }

        with patch.object(self.gemini, "call_gemini", return_value=json.dumps(mock_rewrite_json)) as mock_call:
            res = self.gemini.rewrite_failed_segments(failed_segments, round_num=1, max_rounds=2, db=self.db)
            self.assertEqual(mock_call.call_count, 1, "Must send all failed segments in ONE request")
            self.assertTrue(res["success"])
            self.assertEqual(res["round"], 1)
            self.assertEqual(len(res["rewritten_segments"]), 2)

    def test_batch_rewrite_max_rounds_enforced(self):
        """Verify rewrite is strictly rejected when exceeding max 2 rounds."""
        failed_segments = [{"segment_id": 1, "target_duration": 2.0, "actual_duration": 3.0, "current_text": "Text"}]
        # Round 3 exceeds max_rounds=2
        res = self.gemini.rewrite_failed_segments(failed_segments, round_num=3, max_rounds=2, db=self.db)
        self.assertFalse(res["success"])
        self.assertTrue(res["max_rounds_exceeded"])
        self.assertIn("vượt quá số lần viết lại tối đa", res["error"])

    # ==========================================================
    # 4. ONE-CALL 7-PLATFORM CONTENT GENERATION
    # ==========================================================
    def test_one_call_7_platform_content(self):
        """Verify caption generation produces all 7 platforms in ONE single request."""
        # Create test video and voice record
        prod = Product(product_id="PTEST_7P", niche="Tech", name_vietnamese="Tai nghe Bluetooth", status="RESEARCHED")
        self.db.add(prod)
        self.db.flush()

        vid = Video(video_id="VTEST_7P", product_id="PTEST_7P", douyin_url="https://v.douyin.com/test7p/", status="READY")
        self.db.add(vid)
        self.db.flush()

        voice = Voice(video_id="VTEST_7P", script="Tai nghe chống ồn cực đỉnh cho dân văn phòng.", status="SCRIPT_READY")
        self.db.add(voice)
        self.db.commit()

        mock_content = {
            "facebook_personal": {"caption": "Cap FB cá nhân", "hashtags": "#fb"},
            "facebook_page": {"caption": "Cap FB page", "hashtags": "#page"},
            "tiktok": {"caption": "Cap TikTok", "hashtags": "#tiktok"},
            "threads": {"caption": "Cap Threads", "hashtags": "#threads"},
            "instagram": {"caption": "Cap Insta", "hashtags": "#insta"},
            "shopee": {"caption": "Cap Shopee", "hashtags": "#shopee"},
            "youtube": {"title": "Title Shorts", "description": "Mô tả", "hashtags": "#shorts"}
        }

        with patch.object(GeminiService, "call_gemini", return_value=json.dumps(mock_content)) as mock_call:
            res = self.content_service.generate_content_for_video(self.db, "VTEST_7P", regenerate=True)
            self.assertTrue(res["success"])
            self.assertEqual(mock_call.call_count, 1, "Must generate all 7 platforms in ONE single call")
            self.assertEqual(res["status"], "CONTENT_READY")

        # Cleanup
        self.db.query(Content).filter(Content.video_id == "VTEST_7P").delete()
        self.db.query(Voice).filter(Voice.video_id == "VTEST_7P").delete()
        self.db.query(Video).filter(Video.video_id == "VTEST_7P").delete()
        self.db.query(Product).filter(Product.product_id == "PTEST_7P").delete()
        self.db.commit()

    # ==========================================================
    # 5. DAILY QUOTA DETECTION & NO RETRY
    # ==========================================================
    def test_daily_quota_error_detection(self):
        """Verify is_daily_quota_error accurately identifies Google Free Tier daily quota exhaustion."""
        err_daily = {
            "error": {
                "code": 429,
                "message": "Resource has been exhausted (e.g. check quota).",
                "status": "RESOURCE_EXHAUSTED",
                "details": [{
                    "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                    "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                    "quotaValue": "20"
                }]
            }
        }
        self.assertTrue(is_daily_quota_error(429, "Resource has been exhausted", err_daily))

        # Compare with temporary rate limit (RPM)
        err_temp = {
            "error": {
                "code": 429,
                "message": "GenerateRequestsPerMinute limit exceeded. Please retry in 5s.",
                "status": "RESOURCE_EXHAUSTED"
            }
        }
        self.assertFalse(is_daily_quota_error(429, "GenerateRequestsPerMinute limit exceeded", err_temp))

    def test_no_retry_on_daily_quota_exhaustion(self):
        """Verify that when 429 Daily Quota is encountered, call_gemini fails immediately with 0 retries."""
        daily_quota_resp = MagicMock()
        daily_quota_resp.status_code = 429
        daily_quota_resp.text = json.dumps({
            "error": {
                "code": 429,
                "message": "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests",
                "status": "RESOURCE_EXHAUSTED",
                "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]
            }
        })
        daily_quota_resp.json.return_value = json.loads(daily_quota_resp.text)

        with patch("httpx.Client.post", return_value=daily_quota_resp) as mock_post:
            with self.assertRaises(GeminiQuotaExceededError) as ctx:
                self.gemini.call_gemini("Test prompt", db=self.db, max_retries=3)

            # CRITICAL: Must be called EXACTLY 1 time. Zero retries allowed!
            self.assertEqual(mock_post.call_count, 1, "Must NOT retry daily quota errors!")
            self.assertIn("GEMINI_QUOTA_EXCEEDED", str(ctx.exception))

    def test_limited_retry_on_temporary_burst_rate_limit(self):
        """Verify temporary burst rate limit (non-daily) allows limited retry."""
        temp_resp = MagicMock()
        temp_resp.status_code = 429
        temp_resp.text = json.dumps({
            "error": {
                "code": 429,
                "message": "Rate limit exceeded (per minute). retry in 0.1s",
                "status": "RESOURCE_EXHAUSTED"
            }
        })
        temp_resp.json.return_value = json.loads(temp_resp.text)

        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "OK"}]}}]
        }

        # First call: temporary 429, Second call: 200 OK
        with patch("httpx.Client.post", side_effect=[temp_resp, success_resp]) as mock_post:
            with patch("time.sleep", return_value=None):
                res = self.gemini.call_gemini("Test prompt", db=self.db, max_retries=1)
                self.assertEqual(res, "OK")
                self.assertEqual(mock_post.call_count, 2, "Should retry once on temporary rate limit")

    # ==========================================================
    # 6. CACHING / NO DUPLICATE AI CALLS
    # ==========================================================
    def test_caching_avoids_duplicate_script_generation(self):
        """Verify SCRIPT_READY does not call Gemini again when regenerate=False."""
        vid = Video(video_id="VTEST_CACHE_SCRIPT", douyin_url="https://v.douyin.com/test_cache1/", status="SCRIPT_READY")
        self.db.add(vid)
        self.db.flush()

        voice = Voice(
            video_id="VTEST_CACHE_SCRIPT",
            script="Đây là kịch bản đã tạo hợp lệ từ trước, không cần gọi lại Gemini.",
            status="SCRIPT_READY"
        )
        self.db.add(voice)
        self.db.commit()

        with patch.object(GeminiService, "call_gemini") as mock_gemini:
            res = self.script_service.generate_script_for_video(self.db, "VTEST_CACHE_SCRIPT", regenerate=False)
            self.assertTrue(res["success"])
            self.assertTrue(res.get("cached"), "Result must be marked as cached")
            self.assertEqual(mock_gemini.call_count, 0, "Must NOT call Gemini when script already exists!")

        # Cleanup
        self.db.query(Voice).filter(Voice.video_id == "VTEST_CACHE_SCRIPT").delete()
        self.db.query(Video).filter(Video.video_id == "VTEST_CACHE_SCRIPT").delete()
        self.db.commit()

    def test_caching_avoids_duplicate_content_generation(self):
        """Verify CONTENT_READY does not call Gemini again when regenerate=False."""
        vid = Video(video_id="VTEST_CACHE_CONTENT", douyin_url="https://v.douyin.com/test_cache2/", status="CONTENT_READY")
        self.db.add(vid)
        self.db.flush()

        voice = Voice(video_id="VTEST_CACHE_CONTENT", script="Kịch bản có sẵn.", status="SCRIPT_READY")
        self.db.add(voice)
        self.db.flush()

        content = Content(
            video_id="VTEST_CACHE_CONTENT",
            facebook_personal_caption="Caption đã có",
            facebook_page_caption="Caption page đã có",
            status="CONTENT_READY"
        )
        self.db.add(content)
        self.db.commit()

        with patch.object(GeminiService, "call_gemini") as mock_gemini:
            res = self.content_service.generate_content_for_video(self.db, "VTEST_CACHE_CONTENT", regenerate=False)
            self.assertTrue(res["success"])
            self.assertTrue(res.get("cached"), "Result must be marked as cached")
            self.assertEqual(mock_gemini.call_count, 0, "Must NOT call Gemini when content is already ready!")

        # Cleanup
        self.db.query(Content).filter(Content.video_id == "VTEST_CACHE_CONTENT").delete()
        self.db.query(Voice).filter(Voice.video_id == "VTEST_CACHE_CONTENT").delete()
        self.db.query(Video).filter(Video.video_id == "VTEST_CACHE_CONTENT").delete()
        self.db.commit()

    # ==========================================================
    # 7. LOCAL REQUEST COUNTER & DAILY RESET
    # ==========================================================
    def test_local_request_counter_tracking(self):
        """Verify local usage tracker increments counters accurately."""
        # Reset counters for clean test
        for k in ["gemini_requests_today", "gemini_successful_requests", "gemini_failed_requests", "gemini_quota_errors"]:
            rec = self.db.query(Setting).filter(Setting.key == k).first()
            if rec:
                rec.value = "0"
        self.db.commit()

        initial_stats = GeminiUsageTracker.get_usage(self.db)
        self.assertEqual(initial_stats["requests_today"], 0)

        # Simulate 1 request start + 1 success
        GeminiUsageTracker.record_request_start(self.db)
        GeminiUsageTracker.record_success(self.db)

        # Simulate 1 request start + 1 quota failure
        GeminiUsageTracker.record_request_start(self.db)
        GeminiUsageTracker.record_failure(self.db, is_quota=True)

        stats = GeminiUsageTracker.get_usage(self.db)
        self.assertEqual(stats["requests_today"], 2)
        self.assertEqual(stats["successful_requests"], 1)
        self.assertEqual(stats["failed_requests"], 1)
        self.assertEqual(stats["quota_errors"], 1)
        self.assertIn("tham khảo", stats["disclaimer"])

    def test_local_request_counter_daily_reset(self):
        """Verify that when the date changes, the counter automatically resets to 0."""
        # Set usage date to yesterday
        rec = self.db.query(Setting).filter(Setting.key == "gemini_usage_date").first()
        if rec:
            rec.value = "2020-01-01"
        req_rec = self.db.query(Setting).filter(Setting.key == "gemini_requests_today").first()
        if req_rec:
            req_rec.value = "99"
        self.db.commit()

        # Calling get_usage on today's date must reset requests_today to 0
        stats = GeminiUsageTracker.get_usage(self.db)
        self.assertEqual(stats["requests_today"], 0)
        self.assertNotEqual(stats["date"], "2020-01-01")

    # ==========================================================
    # 8. LOCAL PROCESSING NEVER USES GEMINI
    # ==========================================================
    def test_local_operations_do_not_call_gemini(self):
        """Assert local operations (deduplication, validation, ID generation) never call Gemini."""
        with patch.object(GeminiService, "call_gemini") as mock_gemini:
            # 1. Product ID generation
            from app.services.gemini_service import get_next_product_id
            pid = get_next_product_id(self.db)
            self.assertTrue(pid.startswith("P"))

            # 2. Script validation
            v_res = self.script_service.validate_script("Đây là một kịch bản tiếng Việt hợp lệ trên 40 ký tự và không có lỗi.")
            self.assertTrue(v_res["valid"])

            # 3. Content JSON validation
            from app.services.content_service import validate_content_json
            val_json = validate_content_json({
                "facebook_personal": {"caption": "a"}, "facebook_page": {"caption": "b"},
                "tiktok": {"caption": "c"}, "threads": {"caption": "d"},
                "instagram": {"caption": "e"}, "shopee": {"caption": "f"},
                "youtube": {"title": "g", "description": "h"}
            })
            self.assertTrue(val_json["valid"])

            # 4. Clean JSON response
            cleaned = clean_json_response("```json\n{\"test\": 1}\n```")
            self.assertEqual(cleaned, '{"test": 1}')

            # Assert Gemini was NEVER called for any of these
            self.assertEqual(mock_gemini.call_count, 0)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGeminiHardening)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
