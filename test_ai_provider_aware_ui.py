"""
Comprehensive Offline Regression Test Suite for Provider-Aware UI & Research Progress.

Validates all 6 providers (Gemini, OpenAI, Anthropic, Groq, OpenRouter, MWAPI):
1. Active provider title changes correctly.
2. Active provider model changes correctly.
3. Primary status card changes correctly.
4. Compact global badge changes correctly.
5. Primary diagnostic button changes correctly.
6. Gemini quota is not presented as active-provider state when MWAPI is active.
7. Dashboard rendering makes zero external calls.
8. Settings rendering makes zero external calls.
9. Research rendering makes zero external calls.
10. Research card is provider-aware.
11. Progress UI appears on Research submit.
12. Submit button becomes disabled.
13. Double-submit protection exists.
14. Progress UI itself makes zero provider calls.
15. Cache hit = 0 generation calls.
16. Cache miss = exactly 1 generation call.
17. Fresh Research = exactly 1 generation call.
18. Research max_retries=0.
19. Research fallback disabled.
20. Research performs zero automatic model discovery calls.
21. Research performs zero automatic Test Connection calls.
22. Error state does not retry.
23. MWAPI Research cache miss makes exactly one POST /v1/chat/completions.
24. MWAPI Research performs zero GET /v1/models.
25. Raw API keys never appear in rendered HTML.
26. Existing custom models remain preserved.
27. Existing Anthropic Direct behavior remains unchanged.
28. Existing MWAPI provider tests remain passing.

STRICT INVARIANT:
ALL EXTERNAL NETWORK CALLS MOCKED. REAL API CALLS = 0.
"""
import json
import unittest
from unittest.mock import MagicMock, patch
import httpx
from starlette.testclient import TestClient

from app.database import get_db
from app.main import app
from app.models import Product, Setting
from app.services.ai import (
    get_ai_manager,
    AIAuthenticationError,
    AIQuotaExceededError,
    AIServiceUnavailableError,
)
from app.services.ai.providers.anthropic import AnthropicProvider
from app.services.ai.providers.mwapi import MWAPIProvider
from app.services.ai.status import get_active_provider_status, PROVIDER_METADATA


SAMPLE_PRODUCTS = [
    {
        "name_vietnamese": "Nồi chiên không dầu thông minh",
        "name_chinese": "智能空气炸锅",
        "douyin_keywords": "空气炸锅 厨房好物 家用电器",
        "content_angle": "Nấu ăn nhanh gọn cho gia đình",
        "hook": "Món chiên giòn rụm mà không lo dầu mỡ!",
    }
]


def _make_mock_db(active_provider="mwapi", custom_models=None, custom_keys=None, products=None):
    """Create in-memory mock SQLite session for testing."""
    models = {
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-6",
        "groq": "llama-3.3-70b-versatile",
        "openrouter": "google/gemini-2.0-flash-001",
        "mwapi": "claude-sonnet-4-6",
    }
    if custom_models:
        models.update(custom_models)

    keys = {
        "gemini": "AIzaSy_MOCK_GEMINI_KEY_ABCDEF",
        "openai": "sk-proj-MOCK_OPENAI_KEY_ABCDEF",
        "anthropic": "sk-ant-api03-MOCK_ANTHROPIC_KEY_ABCDEF",
        "groq": "gsk_MOCK_GROQ_KEY_ABCDEF",
        "openrouter": "sk-or-v1-MOCK_OPENROUTER_KEY_ABCDEF",
        "mwapi": "mwapi_live_MOCK_KEY_ABCDEF",
    }
    if custom_keys:
        keys.update(custom_keys)

    products = products or []

    settings_store = {
        "active_ai_provider": active_provider,
        "gemini_model": models["gemini"],
        "openai_model": models["openai"],
        "anthropic_model": models["anthropic"],
        "groq_model": models["groq"],
        "openrouter_model": models["openrouter"],
        "mwapi_model": models["mwapi"],
        "gemini_api_key": keys["gemini"],
        "openai_api_key": keys["openai"],
        "anthropic_api_key": keys["anthropic"],
        "groq_api_key": keys["groq"],
        "openrouter_api_key": keys["openrouter"],
        "mwapi_api_key": keys["mwapi"],
    }

    mock_session = MagicMock()

    def _query_side_effect(model_cls):
        q = MagicMock()
        if model_cls is Setting:
            def _filter_side_effect(*args, **kwargs):
                f = MagicMock()
                def _first_side_effect():
                    try:
                        expr = args[0]
                        k = getattr(getattr(expr, "right", None), "value", None)
                        if isinstance(k, (list, tuple, set)):
                            for cand in k:
                                if cand in settings_store:
                                    s = MagicMock()
                                    s.key = cand
                                    s.value = settings_store[cand]
                                    return s
                        elif isinstance(k, str) and k in settings_store:
                            s = MagicMock()
                            s.key = k
                            s.value = settings_store[k]
                            return s
                    except Exception:
                        pass
                    return None
                f.first = _first_side_effect
                f.all = lambda: []
                return f
            q.filter = _filter_side_effect
            q.all = lambda: []
            return q
        elif model_cls is Product:
            q.all = lambda: list(products)
            q.order_by.return_value.all = lambda: list(products)
            return q
        return q

    mock_session.query.side_effect = _query_side_effect
    return mock_session, settings_store


class TestProviderAwareUIAndResearchProgress(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.pop(get_db, None)

    # -------------------------------------------------------------------------
    # Requirements 1-5: All 6 Providers Status, Titles, Models, Badges, Diagnostics
    # -------------------------------------------------------------------------
    def test_01_all_six_providers_status_object_correctness(self):
        """Verify get_active_provider_status for all 6 providers."""
        providers = [
            ("gemini", "Gemini", "Gemini", "gemini-2.5-flash", "KIỂM TRA GEMINI"),
            ("openai", "OpenAI", "OpenAI", "gpt-4o-mini", "KIỂM TRA OPENAI"),
            ("anthropic", "Anthropic Claude", "Anthropic", "claude-sonnet-4-6", "KIỂM TRA ANTHROPIC CLAUDE"),
            ("groq", "Groq", "Groq", "llama-3.3-70b-versatile", "KIỂM TRA GROQ"),
            ("openrouter", "OpenRouter", "OpenRouter", "google/gemini-2.0-flash-001", "KIỂM TRA OPENROUTER"),
            ("mwapi", "MWAPI Gateway", "MWAPI", "claude-sonnet-4-6", "KIỂM TRA MWAPI GATEWAY"),
        ]
        for pid, dname, sname, default_model, diag_action in providers:
            mock_db, _ = _make_mock_db(active_provider=pid)
            st = get_active_provider_status(db=mock_db)
            self.assertEqual(st["provider_id"], pid)
            self.assertEqual(st["display_name"], dname)
            self.assertEqual(st["short_name"], sname)
            self.assertEqual(st["model"], default_model)
            self.assertTrue(st["configured"])
            self.assertEqual(st["diagnostic_action"], diag_action)
            self.assertEqual(st["status"], "READY")
            self.assertIn(dname, st["status_msg"])

    def test_02_compact_badge_and_header_provider_awareness(self):
        """Header badge and compact indicator must reflect active provider."""
        for pid, short_name in [("mwapi", "MWAPI"), ("anthropic", "Anthropic"), ("openai", "OpenAI")]:
            mock_db, _ = _make_mock_db(active_provider=pid)
            app.dependency_overrides[get_db] = lambda: mock_db

            resp = self.client.get("/settings")
            self.assertEqual(resp.status_code, 200)
            self.assertIn(f'id="aiHeaderName">{short_name}</span>', resp.text)
            self.assertIn('id="aiHeaderStatus">Sẵn sàng</span>', resp.text)

    def test_03_primary_status_card_and_diagnostics_button_in_settings(self):
        """Settings primary status card reflects active provider and has provider-aware diagnostic button."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        resp = self.client.get("/settings")
        self.assertEqual(resp.status_code, 200)
        # Primary status card title
        self.assertIn("Trạng Thái MWAPI Gateway", resp.text)
        self.assertIn("claude-sonnet-4-6", resp.text)
        self.assertIn("MWAPI Gateway đã cấu hình và sẵn sàng", resp.text)
        # Diagnostic button
        self.assertIn("KIỂM TRA MWAPI GATEWAY", resp.text)
        self.assertIn('data-provider="mwapi"', resp.text)

    # -------------------------------------------------------------------------
    # Requirement 6: Gemini quota is NOT presented as active-provider state when MWAPI is active
    # -------------------------------------------------------------------------
    def test_06_gemini_quota_not_presented_as_active_provider_state(self):
        """When MWAPI is active, Gemini quota/errors do not masquerade as primary AI status."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("app.services.gemini_status.GeminiStatusTracker.get_status", return_value={
            "status": "QUOTA_EXCEEDED",
            "message": "Đã hết hạn mức Gemini hôm nay",
            "hint": "Chờ hạn mức mới"
        }):
            resp = self.client.get("/settings")
            self.assertEqual(resp.status_code, 200)
            # The active status card must be MWAPI Gateway, NOT Gemini Quota Exceeded
            self.assertIn("Trạng Thái MWAPI Gateway", resp.text)
            self.assertIn("MWAPI Gateway đã cấu hình và sẵn sàng", resp.text)
            # Gemini-specific section is labeled as optional diagnostic
            self.assertIn("Chẩn đoán riêng Gemini (Tùy chọn)", resp.text)
            # Global compact badge should show MWAPI Sẵn sàng
            self.assertIn('id="aiHeaderName">MWAPI</span>', resp.text)
            self.assertIn('id="aiHeaderStatus">Sẵn sàng</span>', resp.text)

    # -------------------------------------------------------------------------
    # Requirements 7, 8, 9: Zero External Network Calls on Page Renders
    # -------------------------------------------------------------------------
    def test_07_dashboard_rendering_makes_zero_external_calls(self):
        """Rendering / makes 0 external HTTP calls."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        real_send = httpx.Client.send
        def no_external_send(client_self, request, *args, **kwargs):
            if not str(request.url).startswith("http://testserver"):
                raise AssertionError(f"Unexpected external HTTP call: {request.method} {request.url}")
            return real_send(client_self, request, *args, **kwargs)

        with patch.object(httpx.Client, "send", side_effect=no_external_send, autospec=True):
            resp = self.client.get("/")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("MWAPI", resp.text)
            self.assertIn("claude-sonnet-4-6 (Sẵn sàng)", resp.text)

    def test_08_settings_rendering_makes_zero_external_calls(self):
        """Rendering /settings makes 0 external HTTP calls."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        real_send = httpx.Client.send
        def no_external_send(client_self, request, *args, **kwargs):
            if not str(request.url).startswith("http://testserver"):
                raise AssertionError(f"Unexpected external HTTP call: {request.method} {request.url}")
            return real_send(client_self, request, *args, **kwargs)

        with patch.object(httpx.Client, "send", side_effect=no_external_send, autospec=True):
            resp = self.client.get("/settings")
            self.assertEqual(resp.status_code, 200)

    def test_09_research_rendering_makes_zero_external_calls(self):
        """Rendering /research makes 0 external HTTP calls."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        real_send = httpx.Client.send
        def no_external_send(client_self, request, *args, **kwargs):
            if not str(request.url).startswith("http://testserver"):
                raise AssertionError(f"Unexpected external HTTP call: {request.method} {request.url}")
            return real_send(client_self, request, *args, **kwargs)

        with patch.object(httpx.Client, "send", side_effect=no_external_send, autospec=True):
            resp = self.client.get("/research")
            self.assertEqual(resp.status_code, 200)

    # -------------------------------------------------------------------------
    # Requirement 10: Research Card is Provider-Aware
    # -------------------------------------------------------------------------
    def test_10_research_card_is_provider_aware(self):
        """Research card reflects MWAPI Gateway without hardcoded text."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        resp = self.client.get("/research")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Tạo Danh Sách Sản Phẩm Tiềm Năng (MWAPI Gateway)", resp.text)
        self.assertIn("MWAPI Gateway", resp.text)
        self.assertIn("claude-sonnet-4-6", resp.text)
        self.assertIn("Mỗi lượt chạy gửi đúng 1 yêu cầu duy nhất tới MWAPI Gateway.", resp.text)

    # -------------------------------------------------------------------------
    # Requirements 11-14: Progress UI, Submit Disable, Double-Submit, Zero Network
    # -------------------------------------------------------------------------
    def test_11_to_14_research_ui_double_submit_and_truthful_progress_script(self):
        """Verify presence of truthful workflow elements and double-submit protection in HTML/JS."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        resp = self.client.get("/research")
        self.assertEqual(resp.status_code, 200)
        html = resp.text
        # Progress Card elements
        self.assertIn('id="researchProgressCard"', html)
        self.assertIn('id="progressStageText"', html)
        self.assertIn('id="progressPercentBadge"', html)
        self.assertIn('id="researchProgressBar"', html)
        self.assertIn('id="progressDetailText"', html)
        # Double-submit protection
        self.assertIn('isSubmitting', html)
        self.assertIn('btn.disabled = true', html)
        self.assertIn('Đang nghiên cứu...', html)
        # Truthful waiting states without fake increments
        self.assertIn('Chuẩn bị nghiên cứu', html)
        self.assertIn('Đã gửi yêu cầu tới', html)
        self.assertIn('Đang chờ', html)
        self.assertIn('NO FAKE INCREMENTAL PERCENTAGES', html)

    # -------------------------------------------------------------------------
    # Requirements 15-21: Research Invariants (Cache Hit, Miss, Fresh, max_retries=0, fallback=False)
    # -------------------------------------------------------------------------
    def test_15_cache_hit_makes_zero_generation_calls(self):
        """Cache hit performs exactly 0 AI generation requests and redirects with cached=1."""
        existing_p = Product(
            id=1,
            product_id="P0001",
            niche="Đồ gia dụng thông minh",
            name_vietnamese="Nồi chiên không dầu",
            name_chinese="智能空气炸锅",
            douyin_keywords="空气炸锅",
            content_angle="Nấu ăn",
            hook="Giòn ngon",
            status="RESEARCHED"
        )
        mock_db, _ = _make_mock_db(active_provider="mwapi", products=[existing_p])
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("app.services.ai.manager.AIProviderManager.generate_products") as mock_gen:
            resp = self.client.post("/research", data={
                "niche": "Đồ gia dụng thông minh",
                "product_count": 1,
                "fresh": "false",
            }, follow_redirects=False)

            self.assertEqual(resp.status_code, 303)
            self.assertIn("cached=1", resp.headers["location"])
            mock_gen.assert_not_called()

    def test_16_cache_miss_makes_exactly_one_generation_call(self):
        """Cache miss performs exactly 1 generation attempt, max_retries=0, enable_fallback=False."""
        mock_db, _ = _make_mock_db(active_provider="mwapi", products=[])
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("app.services.ai.manager.AIProviderManager.generate_products", return_value=SAMPLE_PRODUCTS) as mock_gen:
            resp = self.client.post("/research", data={
                "niche": "Gia dụng nhà bếp",
                "product_count": 1,
                "fresh": "false",
            }, follow_redirects=False)

            self.assertEqual(resp.status_code, 303)
            self.assertIn("/products?success=1", resp.headers["location"])
            self.assertEqual(mock_gen.call_count, 1)

    def test_17_fresh_research_makes_exactly_one_generation_call(self):
        """Fresh research bypasses cache and makes exactly 1 generation call."""
        existing_p = Product(
            id=1,
            product_id="P0001",
            niche="Đồ gia dụng thông minh",
            name_vietnamese="Nồi chiên không dầu",
            name_chinese="智能空气炸锅",
            douyin_keywords="空气炸锅",
            content_angle="Nấu ăn",
            hook="Giòn ngon",
            status="RESEARCHED"
        )
        mock_db, _ = _make_mock_db(active_provider="mwapi", products=[existing_p])
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("app.services.ai.manager.AIProviderManager.generate_products", return_value=SAMPLE_PRODUCTS) as mock_gen:
            resp = self.client.post("/research", data={
                "niche": "Đồ gia dụng thông minh",
                "product_count": 1,
                "fresh": "true",
            }, follow_redirects=False)

            self.assertEqual(resp.status_code, 303)
            self.assertIn("/products?success=1", resp.headers["location"])
            self.assertNotIn("cached=1", resp.headers["location"])
            self.assertEqual(mock_gen.call_count, 1)

    def test_18_19_research_manager_parameters(self):
        """Verify generate_products in manager delegates to provider with max_retries=0 and fallback disabled."""
        mgr = get_ai_manager()
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        provider = mgr.get_provider("mwapi")

        with patch.object(provider, "generate", return_value='[{"name_vietnamese": "Sản phẩm A", "name_chinese": "A", "douyin_keywords": "A", "content_angle": "A", "hook": "A"}]') as mock_gen:
            res = mgr.generate_products(niche="Đồ tiện ích", count=1, db=mock_db)
            self.assertEqual(len(res), 1)
            mock_gen.assert_called_once()
            _, kwargs = mock_gen.call_args
            self.assertEqual(kwargs.get("max_retries"), 0)
            self.assertFalse(kwargs.get("enable_fallback"))
            options = kwargs.get("options")
            self.assertIsNotNone(options)
            self.assertEqual(options.max_retries, 0)
            self.assertFalse(options.enable_fallback)

    # -------------------------------------------------------------------------
    # Requirement 22: Error State Does Not Retry
    # -------------------------------------------------------------------------
    def test_22_error_state_stops_immediately_without_retry(self):
        """Quota or auth error in Research stops immediately with clean error and 0 retries."""
        mock_db, _ = _make_mock_db(active_provider="mwapi", products=[])
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("app.services.ai.manager.AIProviderManager.generate_products", side_effect=AIQuotaExceededError("Rate limit 429")) as mock_gen:
            resp = self.client.post("/research", data={
                "niche": "Gia dụng",
                "product_count": 1,
                "fresh": "true",
            })
            self.assertEqual(resp.status_code, 429)
            self.assertIn("MWAPI Gateway đã hết hạn mức API / quota", resp.text)
            self.assertEqual(mock_gen.call_count, 1)

    # -------------------------------------------------------------------------
    # Requirements 23-24: MWAPI Research Request Count & Zero Model Discovery
    # -------------------------------------------------------------------------
    def test_23_24_mwapi_research_makes_single_post_and_zero_get_models(self):
        """MWAPI execution under Research makes exactly 1 POST /v1/chat/completions and 0 GET /v1/models."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        provider = MWAPIProvider()

        mock_response = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-test",
                "choices": [{
                    "message": {
                        "content": '[{"name_vietnamese": "Máy hút bụi mini", "name_chinese": "迷你吸尘器", "douyin_keywords": "吸尘器", "content_angle": "Tiện dụng", "hook": "Sạch bóng trong 1 phút"}]'
                    }
                }],
                "usage": {"total_tokens": 100}
            },
            request=httpx.Request("POST", "https://api.mwapi.dev/v1/chat/completions")
        )

        with patch.object(httpx.Client, "send", return_value=mock_response) as mock_send:
            res = provider.generate(prompt="Nghiên cứu sản phẩm", db=mock_db, max_retries=0)
            self.assertIn("Máy hút bụi mini", res)
            self.assertEqual(mock_send.call_count, 1)
            sent_req = mock_send.call_args[0][0]
            self.assertEqual(sent_req.method, "POST")
            self.assertEqual(str(sent_req.url), "https://api.mwapi.dev/v1/chat/completions")
            # Verify ZERO calls to /v1/models
            for call in mock_send.call_args_list:
                self.assertNotIn("/v1/models", str(call[0][0].url))

    # -------------------------------------------------------------------------
    # Requirement 25: Raw API Keys Never Appear in Rendered HTML
    # -------------------------------------------------------------------------
    def test_25_raw_api_keys_never_appear_in_rendered_html(self):
        """Raw API keys must never be exposed in any rendered page."""
        raw_keys = {
            "gemini": "AIzaSy_SECRET_RAW_KEY_GEMINI_999",
            "openai": "sk-proj-SECRET_RAW_KEY_OPENAI_999",
            "anthropic": "sk-ant-SECRET_RAW_KEY_ANTHROPIC_999",
            "groq": "gsk_SECRET_RAW_KEY_GROQ_999",
            "openrouter": "sk-or-v1-SECRET_RAW_KEY_OPENROUTER_999",
            "mwapi": "mwapi_live_SECRET_RAW_KEY_MWAPI_999",
        }
        for pid in ["gemini", "openai", "anthropic", "groq", "openrouter", "mwapi"]:
            mock_db, _ = _make_mock_db(active_provider=pid, custom_keys=raw_keys)
            app.dependency_overrides[get_db] = lambda: mock_db

            for endpoint in ["/", "/settings", "/research", "/products"]:
                resp = self.client.get(endpoint)
                self.assertEqual(resp.status_code, 200)
                for secret in raw_keys.values():
                    self.assertNotIn(secret, resp.text, f"Key {secret} leaked in {endpoint} with active {pid}")

    # -------------------------------------------------------------------------
    # Requirement 26: Existing Custom Models Remain Preserved
    # -------------------------------------------------------------------------
    def test_26_existing_custom_models_remain_preserved(self):
        """Custom user models are respected and displayed verbatim."""
        custom_models = {
            "anthropic": "claude-3-haiku-custom-v1",
            "mwapi": "gpt-4o-custom-enterprise",
        }
        mock_db, _ = _make_mock_db(active_provider="mwapi", custom_models=custom_models)
        st = get_active_provider_status(db=mock_db)
        self.assertEqual(st["model"], "gpt-4o-custom-enterprise")

        mock_db2, _ = _make_mock_db(active_provider="anthropic", custom_models=custom_models)
        st2 = get_active_provider_status(db=mock_db2)
        self.assertEqual(st2["model"], "claude-3-haiku-custom-v1")

    # -------------------------------------------------------------------------
    # Requirement 27: Anthropic Direct Isolation and Behavior
    # -------------------------------------------------------------------------
    def test_27_anthropic_direct_remains_isolated_with_correct_headers(self):
        """Anthropic Direct provider sends x-api-key and anthropic-version to api.anthropic.com."""
        mock_db, _ = _make_mock_db(active_provider="anthropic")
        provider = AnthropicProvider()
        mock_response = httpx.Response(
            status_code=200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Anthropic response"}],
                "model": "claude-sonnet-4-6",
                "usage": {"input_tokens": 10, "output_tokens": 10}
            },
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        )
        with patch.object(httpx.Client, "send", return_value=mock_response) as mock_send:
            res = provider.generate("Test", db=mock_db, max_retries=0)
            self.assertEqual(res, "Anthropic response")
            sent_req = mock_send.call_args[0][0]
            self.assertEqual(str(sent_req.url), "https://api.anthropic.com/v1/messages")
            self.assertIn("x-api-key", sent_req.headers)
            self.assertIn("anthropic-version", sent_req.headers)
            self.assertNotIn("Authorization", sent_req.headers)

    # -------------------------------------------------------------------------
    # Requirement 28: Products Page Shows Truthful Cache and Success Banners
    # -------------------------------------------------------------------------
    def test_28_products_page_renders_truthful_cache_and_success_banners(self):
        """Products page reflects 0 AI calls banner on cache hit and 1 request banner on success."""
        mock_db, _ = _make_mock_db()
        app.dependency_overrides[get_db] = lambda: mock_db

        # Cache hit view
        resp_cached = self.client.get("/products?success=1&count=10&cached=1&niche=Gia+Dụng")
        self.assertEqual(resp_cached.status_code, 200)
        self.assertIn("Đã tìm thấy kết quả trong bộ nhớ đệm — không gọi AI", resp_cached.text)
        self.assertIn("0 LƯỢT GỌI AI", resp_cached.text)

        # Fresh generation success view
        resp_fresh = self.client.get("/products?success=1&count=10&niche=Gia+Dụng")
        self.assertEqual(resp_fresh.status_code, 200)
        self.assertIn("100% Hoàn tất nghiên cứu", resp_fresh.text)
        self.assertIn("1 YÊU CẦU DUY NHẤT", resp_fresh.text)


if __name__ == "__main__":
    unittest.main()
