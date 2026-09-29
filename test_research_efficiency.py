"""
Unit & Integration Tests for Research Low-Consumption Mode
Proves:
1. Successful Research = exactly 1 LLM call.
2. Research HTTP 503 = exactly 1 LLM attempt.
3. Research 429 = exactly 1 LLM attempt.
4. Research timeout = exactly 1 LLM attempt.
5. Invalid JSON = exactly 1 LLM attempt.
6. Research never invokes Gemini fallback.
7. Research never performs quality retry.
8. Local validation does not call AI.
9. Cache hit = 0 LLM calls, if safe cache is implemented.
10. Explicit fresh Research = exactly 1 LLM call.
11. Request counter does not increase on cache hit.
12. Output token budget is bounded.
13. Output token budget scales appropriately with product count.
14. Existing Gemini reliability behavior OUTSIDE Research remains intact.
15. API key is never exposed.
16. Research error UX displays concise, helpful Vietnamese messages.

All tests use mocked HTTP responses — ZERO real Gemini quota consumed.
"""
import unittest
from unittest.mock import patch, MagicMock, ANY
import json
import httpx
from starlette.testclient import TestClient

from app.main import app
from app.database import get_db
from app.models import Product, Setting
from app.services.gemini_service import (
    GeminiService,
    get_research_max_output_tokens,
    GeminiServiceUnavailableError,
    GeminiQuotaExceededError,
    GeminiRateLimitError,
    GeminiTimeoutError,
    sanitize_error_message,
)
from app.routes.research import normalize_niche, get_cached_products_for_niche


def _make_mock_response(status_code: int, data: dict = None, text: str = ""):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = {}
    if data is not None:
        resp.json.return_value = data
        resp.text = json.dumps(data)
    else:
        resp.json.side_effect = Exception("Not JSON")
        resp.text = text
    return resp


SAMPLE_PRODUCTS_JSON = json.dumps([
    {
        "name_vietnamese": "Máy hút bụi giường nệm UV",
        "name_chinese": "除螨仪家用小型吸尘器",
        "douyin_keywords": "除螨仪实测 居家好物 开箱测评",
        "content_angle": "Giải pháp bảo vệ sức khỏe gia đình",
        "hook": "Nếu bạn hay bị ngứa ngáy khi ngủ thì xem ngay!"
    },
    {
        "name_vietnamese": "Bình giữ nhiệt hiển thị nhiệt độ",
        "name_chinese": "智能保温杯显温水杯",
        "douyin_keywords": "智能显温保温杯 学生党便携水杯",
        "content_angle": "Tiện ích văn phòng và học tập",
        "hook": "Chiếc bình giữ nhiệt thông minh nhất mình từng dùng!"
    }
])


class TestResearchLowConsumption(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_db.query.return_value.filter.return_value.first.return_value = None

    # ------------------------------------------------------------------
    # 1. Successful Research = exactly 1 LLM call
    # ------------------------------------------------------------------
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("httpx.Client.post")
    def test_01_successful_research_exactly_one_llm_call(self, mock_post, mock_model, mock_key):
        resp_200 = _make_mock_response(200, {
            "candidates": [{"content": {"parts": [{"text": SAMPLE_PRODUCTS_JSON}]}}]
        })
        mock_post.return_value = resp_200

        service = GeminiService()
        result = service.generate_products("Đồ gia dụng", count=2, db=self.mock_db)

        self.assertEqual(mock_post.call_count, 1, "Successful research must make exactly 1 LLM HTTP call")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name_vietnamese"], "Máy hút bụi giường nệm UV")

    # ------------------------------------------------------------------
    # 2. Research HTTP 503 = exactly 1 LLM attempt
    # ------------------------------------------------------------------
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("httpx.Client.post")
    def test_02_research_503_exactly_one_llm_attempt(self, mock_post, mock_model, mock_key):
        resp_503 = _make_mock_response(503, {"error": {"message": "High demand"}})
        mock_post.return_value = resp_503

        service = GeminiService()
        with self.assertRaises(GeminiServiceUnavailableError):
            service.generate_products("Đồ gia dụng", count=5, db=self.mock_db)

        self.assertEqual(mock_post.call_count, 1, "HTTP 503 in research must stop immediately with 0 retries")

    # ------------------------------------------------------------------
    # 3. Research 429 = exactly 1 LLM attempt
    # ------------------------------------------------------------------
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("httpx.Client.post")
    def test_03_research_429_exactly_one_llm_attempt(self, mock_post, mock_model, mock_key):
        resp_429 = _make_mock_response(429, {"error": {"message": "Resource has been exhausted"}})
        mock_post.return_value = resp_429

        service = GeminiService()
        with self.assertRaises((GeminiQuotaExceededError, GeminiRateLimitError)):
            service.generate_products("Đồ gia dụng", count=5, db=self.mock_db)

        self.assertEqual(mock_post.call_count, 1, "HTTP 429 in research must stop immediately with 0 retries")

    # ------------------------------------------------------------------
    # 4. Research timeout = exactly 1 LLM attempt
    # ------------------------------------------------------------------
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("httpx.Client.post", side_effect=httpx.TimeoutException("Connection timed out"))
    def test_04_research_timeout_exactly_one_llm_attempt(self, mock_post, mock_model, mock_key):
        service = GeminiService()
        with self.assertRaises(GeminiTimeoutError):
            service.generate_products("Đồ gia dụng", count=5, db=self.mock_db)

        self.assertEqual(mock_post.call_count, 1, "Timeout in research must make exactly 1 LLM attempt")

    # ------------------------------------------------------------------
    # 5. Invalid JSON = exactly 1 LLM attempt (no repair, no re-call)
    # ------------------------------------------------------------------
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("httpx.Client.post")
    def test_05_invalid_json_exactly_one_llm_attempt(self, mock_post, mock_model, mock_key):
        resp_200 = _make_mock_response(200, {
            "candidates": [{"content": {"parts": [{"text": "Xin lỗi, tôi không thể trả về JSON lúc này."}]}}]
        })
        mock_post.return_value = resp_200

        service = GeminiService()
        with self.assertRaises(ValueError):
            service.generate_products("Đồ gia dụng", count=5, db=self.mock_db)

        self.assertEqual(mock_post.call_count, 1, "Invalid JSON must fail fast locally with 0 additional LLM calls")

    # ------------------------------------------------------------------
    # 6. Research never invokes Gemini fallback
    # ------------------------------------------------------------------
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("httpx.Client.post")
    def test_06_research_never_invokes_gemini_fallback(self, mock_post, mock_model, mock_key):
        resp_503 = _make_mock_response(503, {"error": {"message": "High demand"}})
        mock_post.return_value = resp_503

        service = GeminiService()
        with self.assertRaises(GeminiServiceUnavailableError):
            service.generate_products("Đồ gia dụng", count=5, db=self.mock_db)

        # Confirm post was called once and with gemini-3.8-flash only (never called fallback)
        self.assertEqual(mock_post.call_count, 1)
        called_url = mock_post.call_args[0][0]
        self.assertIn("gemini-3.8-flash", called_url)
        self.assertNotIn("gemini-3.7-flash", called_url)

    # ------------------------------------------------------------------
    # 7. Research never performs quality retry
    # ------------------------------------------------------------------
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("httpx.Client.post")
    def test_07_research_never_performs_quality_retry(self, mock_post, mock_model, mock_key):
        # Requested 10 products, AI only returned 2
        resp_200 = _make_mock_response(200, {
            "candidates": [{"content": {"parts": [{"text": SAMPLE_PRODUCTS_JSON}]}}]
        })
        mock_post.return_value = resp_200

        service = GeminiService()
        result = service.generate_products("Đồ gia dụng", count=10, db=self.mock_db)

        self.assertEqual(mock_post.call_count, 1, "Must not retry to fill remaining product quota")
        self.assertEqual(len(result), 2)

    # ------------------------------------------------------------------
    # 8. Local validation does not call AI
    # ------------------------------------------------------------------
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("httpx.Client.post")
    def test_08_local_validation_does_not_call_ai(self, mock_post, mock_model, mock_key):
        dirty_json = json.dumps([
            {"name_vietnamese": "Sản phẩm A", "name_chinese": "A"},
            {"non_dict_item": 123},
            {"name_vietnamese": "", "name_chinese": "Invalid empty name"},
            {"name_vietnamese": "Sản phẩm B", "content_angle": "Angle B"}
        ])
        resp_200 = _make_mock_response(200, {
            "candidates": [{"content": {"parts": [{"text": dirty_json}]}}]
        })
        mock_post.return_value = resp_200

        service = GeminiService()
        result = service.generate_products("Đồ gia dụng", count=5, db=self.mock_db)

        self.assertEqual(mock_post.call_count, 1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name_vietnamese"], "Sản phẩm A")
        self.assertEqual(result[1]["name_vietnamese"], "Sản phẩm B")

    # ------------------------------------------------------------------
    # 9. Cache hit = 0 LLM calls (deterministic local normalization)
    # ------------------------------------------------------------------
    def test_09_cache_normalization_logic(self):
        self.assertEqual(normalize_niche("Đồ gia dụng"), "đồ gia dụng")
        self.assertEqual(normalize_niche("  đồ   gia   dụng  "), "đồ gia dụng")
        self.assertEqual(normalize_niche("ĐỒ GIA DỤNG"), "đồ gia dụng")
        self.assertEqual(normalize_niche(""), "")

    @patch("app.routes.research.GeminiService.generate_products")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start")
    def test_09_cache_hit_zero_llm_calls_via_route(self, mock_record, mock_generate):
        mock_session = MagicMock()
        cached_p1 = MagicMock(spec=Product, niche="Đồ gia dụng", name_vietnamese="Sản phẩm 1")
        cached_p2 = MagicMock(spec=Product, niche="đồ gia dụng", name_vietnamese="Sản phẩm 2")
        mock_session.query.return_value.order_by.return_value.all.return_value = [cached_p1, cached_p2]

        app.dependency_overrides[get_db] = lambda: mock_session
        try:
            client = TestClient(app)
            response = client.post(
                "/research",
                data={"niche": "  ĐỒ GIA DỤNG  ", "product_count": 2, "fresh": False},
                follow_redirects=False
            )
            self.assertEqual(response.status_code, 303)
            self.assertIn("cached=1", response.headers["location"])
            self.assertEqual(mock_generate.call_count, 0, "Cache hit must make exactly 0 LLM calls")
            self.assertEqual(mock_record.call_count, 0, "Cache hit must not increment usage tracker")
        finally:
            app.dependency_overrides.pop(get_db, None)

    # ------------------------------------------------------------------
    # 10. Explicit fresh Research = exactly 1 LLM call
    # ------------------------------------------------------------------
    @patch("app.routes.research.GeminiService.generate_products")
    @patch("app.routes.research.get_next_product_id", return_value="P0010")
    def test_10_explicit_fresh_research_exactly_one_llm_call(self, mock_pid, mock_generate):
        mock_generate.return_value = [
            {"name_vietnamese": "Sản phẩm Mới 1", "name_chinese": "", "douyin_keywords": "", "content_angle": "", "hook": ""}
        ]
        mock_session = MagicMock()
        # 10 products already cached, but fresh=True
        mock_prods = [MagicMock(spec=Product, niche="Đồ gia dụng", name_vietnamese=f"Cũ {i}") for i in range(10)]
        mock_session.query.return_value.order_by.return_value.all.return_value = mock_prods
        mock_session.query.return_value.all.return_value = mock_prods

        app.dependency_overrides[get_db] = lambda: mock_session
        try:
            client = TestClient(app)
            response = client.post(
                "/research",
                data={"niche": "Đồ gia dụng", "product_count": 5, "fresh": True},
                follow_redirects=False
            )
            self.assertEqual(response.status_code, 303)
            self.assertNotIn("cached=1", response.headers["location"])
            self.assertEqual(mock_generate.call_count, 1, "Fresh research must trigger exactly 1 LLM call")
        finally:
            app.dependency_overrides.pop(get_db, None)

    # ------------------------------------------------------------------
    # 11. Request counter does not increase on cache hit
    # ------------------------------------------------------------------
    @patch("app.routes.research.GeminiService.generate_products")
    @patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start")
    def test_11_request_counter_does_not_increase_on_cache_hit(self, mock_record, mock_generate):
        mock_session = MagicMock()
        cached_prods = [MagicMock(spec=Product, niche="Gia dụng", name_vietnamese=f"SP {i}") for i in range(5)]
        mock_session.query.return_value.order_by.return_value.all.return_value = cached_prods

        app.dependency_overrides[get_db] = lambda: mock_session
        try:
            client = TestClient(app)
            client.post("/research", data={"niche": "Gia dụng", "product_count": 5, "fresh": False}, follow_redirects=False)
            self.assertEqual(mock_record.call_count, 0)
        finally:
            app.dependency_overrides.pop(get_db, None)

    # ------------------------------------------------------------------
    # 12 & 13. Output token budget bounded and scales appropriately
    # ------------------------------------------------------------------
    def test_12_and_13_output_token_budget_mapping(self):
        self.assertEqual(get_research_max_output_tokens(5), 800)
        self.assertEqual(get_research_max_output_tokens(10), 1500)
        self.assertEqual(get_research_max_output_tokens(20), 2800)
        self.assertEqual(get_research_max_output_tokens(30), 4000)
        self.assertEqual(get_research_max_output_tokens(50), 6000)
        # Boundedness
        self.assertLessEqual(get_research_max_output_tokens(50), 6000)

    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("httpx.Client.post")
    def test_13_output_token_passed_to_gemini_payload(self, mock_post, mock_model, mock_key):
        resp_200 = _make_mock_response(200, {
            "candidates": [{"content": {"parts": [{"text": SAMPLE_PRODUCTS_JSON}]}}]
        })
        mock_post.return_value = resp_200

        service = GeminiService()
        service.generate_products("Gia dụng", count=10, db=self.mock_db)

        # Inspect json sent in POST
        called_json = mock_post.call_args[1]["json"]
        self.assertIn("generationConfig", called_json)
        self.assertEqual(called_json["generationConfig"]["maxOutputTokens"], 1500)

    # ------------------------------------------------------------------
    # 14. Existing Gemini reliability behavior OUTSIDE Research intact
    # ------------------------------------------------------------------
    @patch("time.sleep", return_value=None)
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_TEST_KEY_FOR_UNIT_TESTS")
    @patch("app.config.get_gemini_model", return_value="gemini-3.8-flash")
    @patch("httpx.Client.post")
    def test_14_non_research_gemini_behavior_intact(self, mock_post, mock_model, mock_key, mock_sleep):
        resp_503 = _make_mock_response(503, {"error": {"message": "Busy"}})
        resp_200 = _make_mock_response(200, {"candidates": [{"content": {"parts": [{"text": "Outside Research OK"}]}}]})
        mock_post.side_effect = [resp_503, resp_200]

        service = GeminiService()
        # Outside research, max_retries > 0 and enable_fallback can be True
        result = service.call_gemini("Test prompt outside research", db=self.mock_db, max_retries=2, enable_fallback=True)

        self.assertEqual(result, "Outside Research OK")
        self.assertEqual(mock_post.call_count, 2, "Non-research calls must still retry on 503")

    # ------------------------------------------------------------------
    # 15. API key is never exposed
    # ------------------------------------------------------------------
    def test_15_api_key_never_exposed(self):
        raw_error = "Error calling https://generativelanguage.googleapis.com?key=AIzaSySecret123456789: 503 Overloaded"
        sanitized = sanitize_error_message(raw_error)
        self.assertNotIn("AIzaSySecret123456789", sanitized)
        self.assertIn("[REDACTED]", sanitized)

        direct_key_error = "Failed with key AIzaSy_SECRET_KEY_123"
        sanitized_direct = sanitize_error_message(direct_key_error, api_key="AIzaSy_SECRET_KEY_123")
        self.assertNotIn("AIzaSy_SECRET_KEY_123", sanitized_direct)
        self.assertIn("[REDACTED]", sanitized_direct)

    # ------------------------------------------------------------------
    # 16. Research error UX Vietnamese messages
    # ------------------------------------------------------------------
    @patch("app.routes.research.GeminiService.generate_products")
    def test_16_vietnamese_error_messages(self, mock_generate):
        mock_session = MagicMock()
        mock_session.query.return_value.order_by.return_value.all.return_value = []
        app.dependency_overrides[get_db] = lambda: mock_session

        client = TestClient(app)
        try:
            # 503
            mock_generate.side_effect = GeminiServiceUnavailableError("Model overloaded 503")
            r_503 = client.post("/research", data={"niche": "Gia dụng", "product_count": 5})
            self.assertEqual(r_503.status_code, 400)
            self.assertIn("tạm thời quá tải", r_503.text)

            # 429
            mock_generate.side_effect = GeminiQuotaExceededError("Quota exhausted 429")
            r_429 = client.post("/research", data={"niche": "Gia dụng", "product_count": 5})
            self.assertEqual(r_429.status_code, 429)
            self.assertIn("hạn mức Gemini", r_429.text)

            # Timeout
            mock_generate.side_effect = GeminiTimeoutError("Request timed out")
            r_to = client.post("/research", data={"niche": "Gia dụng", "product_count": 5})
            self.assertEqual(r_to.status_code, 400)
            self.assertIn("hết thời gian chờ", r_to.text)

            # Invalid JSON
            mock_generate.side_effect = ValueError("Malformed AI response format: unexpected char")
            r_json = client.post("/research", data={"niche": "Gia dụng", "product_count": 5})
            self.assertEqual(r_json.status_code, 400)
            self.assertIn("không hợp lệ", r_json.text)
        finally:
            app.dependency_overrides.pop(get_db, None)


if __name__ == "__main__":
    unittest.main()
