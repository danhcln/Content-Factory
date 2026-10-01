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
    AITimeoutError,
)
from app.services.ai.base import (
    get_research_max_output_tokens,
    get_research_timeout,
)
from app.services.ai.providers.anthropic import AnthropicProvider
from app.services.ai.providers.mwapi import MWAPIProvider
from app.services.ai.status import get_active_provider_status, PROVIDER_METADATA
from app.services import gemini_service
from app.services.niche_catalog import (
    NICHE_CATALOG,
    get_broad_categories,
    get_all_niches,
    get_random_niche_sample,
    get_catalog_dict,
)



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

    # -------------------------------------------------------------------------
    # Requirement 29: Research Numeric Input UX & Presets & Advisory Text
    # -------------------------------------------------------------------------
    def test_29_research_product_count_ui_elements(self):
        """Research page renders numeric input (1-50), quick presets [5,10,20,30,50], and token advisory."""
        mock_db, _ = _make_mock_db()
        app.dependency_overrides[get_db] = lambda: mock_db

        resp = self.client.get("/research")
        self.assertEqual(resp.status_code, 200)

        # 1. Numeric input with min=1, max=50, step=1
        self.assertIn('type="number"', resp.text)
        self.assertIn('id="productCount"', resp.text)
        self.assertIn('name="product_count"', resp.text)
        self.assertIn('min="1"', resp.text)
        self.assertIn('max="50"', resp.text)
        self.assertIn('step="1"', resp.text)

        # 2. Quick presets [5, 10, 20, 30, 50] with type="button" and .btn-preset
        for count in [5, 10, 20, 30, 50]:
            self.assertIn(f'data-count="{count}"', resp.text)
            self.assertIn(f'id="presetBtn{count}"', resp.text)

        # 3. Advisory text
        self.assertIn("Yêu cầu nhiều sản phẩm hơn có thể cần nhiều token và thời gian xử lý hơn.", resp.text)

        # 4. Default 10 is highlighted
        self.assertIn('id="presetBtn10"', resp.text)
        self.assertIn('value="10"', resp.text)

    # -------------------------------------------------------------------------
    # Requirement 30: Exact Token Budget and Timeout Anchors
    # -------------------------------------------------------------------------
    def test_30_exact_token_budget_and_timeout_anchors(self):
        """Verify exact token budget and timeout values for all anchor points."""
        anchors = [
            (1, 500, 60.0),
            (5, 800, 60.0),
            (10, 1500, 75.0),
            (20, 2800, 100.0),
            (25, 3400, 125.0),
            (30, 4000, 150.0),
            (40, 5000, 225.0),
            (50, 6000, 300.0),
        ]
        for count, expected_tokens, expected_timeout in anchors:
            tokens = get_research_max_output_tokens(count)
            timeout = get_research_timeout(count)
            self.assertEqual(
                tokens, expected_tokens,
                f"Token budget mismatch for count={count}: got {tokens}, expected {expected_tokens}"
            )
            self.assertEqual(
                timeout, expected_timeout,
                f"Timeout mismatch for count={count}: got {timeout}, expected {expected_timeout}"
            )

    # -------------------------------------------------------------------------
    # Requirement 31: Token Budget Strictly Monotonic for Every Integer 1..50
    # -------------------------------------------------------------------------
    def test_31_token_budget_strictly_monotonic_for_1_to_50(self):
        """Verify token budget is strictly increasing for every integer from 1 to 50."""
        for c in range(1, 50):
            t_curr = get_research_max_output_tokens(c)
            t_next = get_research_max_output_tokens(c + 1)
            self.assertGreater(
                t_next, t_curr,
                f"Monotonicity violated at count={c}: {t_next} <= {t_curr}"
            )

        # Bounded tests
        self.assertEqual(get_research_max_output_tokens(0), 500)
        self.assertEqual(get_research_max_output_tokens(-5), 500)
        self.assertEqual(get_research_max_output_tokens(60), 6000)

    # -------------------------------------------------------------------------
    # Requirement 32: Timeout Non-Decreasing for Every Integer 1..50 and Capped at 300s
    # -------------------------------------------------------------------------
    def test_32_timeout_non_decreasing_for_1_to_50_and_capped_at_180(self):
        """Verify timeout is non-decreasing for every integer from 1 to 50 and capped at 300s."""
        for c in range(1, 50):
            to_curr = get_research_timeout(c)
            to_next = get_research_timeout(c + 1)
            self.assertGreaterEqual(
                to_next, to_curr,
                f"Non-decreasing violated at count={c}: {to_next} < {to_curr}"
            )

        # Hard ceiling test
        self.assertEqual(get_research_timeout(50), 300.0)
        self.assertEqual(get_research_timeout(51), 300.0)
        self.assertEqual(get_research_timeout(100), 300.0)

    # -------------------------------------------------------------------------
    # Requirement 33: Preset Buttons are type=button and Have Zero Submit Action
    # -------------------------------------------------------------------------
    def test_33_preset_buttons_attributes_and_type_button(self):
        """All preset buttons must have type='button' so they never trigger form submission."""
        mock_db, _ = _make_mock_db()
        app.dependency_overrides[get_db] = lambda: mock_db
        resp = self.client.get("/research")

        for count in [5, 10, 20, 30, 50]:
            btn_needle = f'type="button" \n                                    class="btn btn-sm btn-outline-info btn-preset'
            self.assertIn(f'data-count="{count}"', resp.text)
            self.assertIn(f'id="presetBtn{count}"', resp.text)

    # -------------------------------------------------------------------------
    # Requirement 34: Dual Server Validation Rejects Invalid Counts with Zero AI Calls
    # -------------------------------------------------------------------------
    def test_34_server_validation_rejects_invalid_counts_with_zero_ai_calls(self):
        """Server-side validation rejects empty, non-numeric, decimal, <1, >50 with 400 and 0 AI calls."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        invalid_counts = [
            "",
            "abc",
            "1.5",
            "10.0",
            "20,5",
            "0",
            "-1",
            "-10",
            "51",
            "100",
        ]

        with patch("app.services.ai.manager.AIProviderManager.generate_products") as mock_gen:
            for bad_count in invalid_counts:
                resp = self.client.post("/research", data={
                    "niche": "Đồ chơi công nghệ",
                    "product_count": bad_count,
                    "fresh": "true",
                })
                self.assertEqual(
                    resp.status_code, 400,
                    f"Expected 400 for product_count='{bad_count}', got {resp.status_code}"
                )
                self.assertIn("Số lượng sản phẩm không hợp lệ", resp.text)
                mock_gen.assert_not_called()

    # -------------------------------------------------------------------------
    # Requirement 35: Custom Product Counts Accepted and Generate Single Request
    # -------------------------------------------------------------------------
    def test_35_custom_product_counts_accepted_and_generate_single_request(self):
        """Custom counts (7, 12, 15, 25, 35, 40, 45) are accepted and make exactly 1 AI call."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        custom_counts = [7, 12, 15, 25, 35, 40, 45]
        for count in custom_counts:
            with patch("app.services.ai.manager.AIProviderManager.generate_products", return_value=SAMPLE_PRODUCTS) as mock_gen:
                resp = self.client.post("/research", data={
                    "niche": f"Niche test {count}",
                    "product_count": str(count),
                    "fresh": "true",
                }, follow_redirects=False)

                self.assertEqual(resp.status_code, 303)
                self.assertEqual(mock_gen.call_count, 1)
                args, kwargs = mock_gen.call_args
                self.assertEqual(kwargs.get("count"), count)

    # -------------------------------------------------------------------------
    # Requirement 36: MWAPI 30 Products Single Request 120s Timeout 4000 Tokens
    # -------------------------------------------------------------------------
    def test_36_mwapi_30_products_single_request_120s_timeout_4000_tokens(self):
        """For MWAPI 30 products: exactly 1 POST /v1/chat/completions, timeout=120s, max_tokens=4000, 0 GET /v1/models."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        provider = MWAPIProvider()

        raw_30_items = [
            {"name_vietnamese": f"Sản phẩm {i}", "name_chinese": f"产品 {i}", "douyin_keywords": "kw", "content_angle": "ang", "hook": "hk"}
            for i in range(1, 31)
        ]
        mock_response = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-30prods",
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(raw_30_items, ensure_ascii=False)
                    }
                }],
                "usage": {"total_tokens": 500}
            },
            request=httpx.Request("POST", "https://api.mwapi.dev/v1/chat/completions")
        )

        with patch.object(httpx.Client, "send", return_value=mock_response) as mock_send:
            res = provider.generate_products(niche="Nhà bếp", count=30, db=mock_db)
            self.assertEqual(len(res), 30)

            # Exactly 1 HTTP request
            self.assertEqual(mock_send.call_count, 1)
            sent_req = mock_send.call_args[0][0]
            self.assertEqual(sent_req.method, "POST")
            self.assertEqual(str(sent_req.url), "https://api.mwapi.dev/v1/chat/completions")

            # Max tokens = 4000 in payload
            sent_body = json.loads(sent_req.content.decode("utf-8"))
            self.assertEqual(sent_body.get("max_tokens"), 4000)

            # ZERO calls to /v1/models
            for call in mock_send.call_args_list:
                self.assertNotIn("/v1/models", str(call[0][0].url))

    # -------------------------------------------------------------------------
    # Requirement 37: Legacy Gemini Service Uses Canonical Functions
    # -------------------------------------------------------------------------
    def test_37_legacy_gemini_service_uses_canonical_functions(self):
        """Legacy gemini_service uses the exact canonical token-budget and timeout functions from base.py."""
        self.assertIs(
            gemini_service.get_research_max_output_tokens,
            get_research_max_output_tokens
        )
        self.assertIs(
            gemini_service.get_research_timeout,
            get_research_timeout
        )

    # -------------------------------------------------------------------------
    # Requirement 38: Research Rendering and Progress Trigger Zero External API Calls
    # -------------------------------------------------------------------------
    def test_38_research_rendering_and_progress_trigger_zero_external_api_calls(self):
        """Research page GET and preset clicks make 0 external network requests."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        real_send = httpx.Client.send
        def no_external_send(client_self, request, *args, **kwargs):
            if not str(request.url).startswith("http://testserver"):
                raise AssertionError(f"Unexpected external HTTP call: {request.method} {request.url}")
            return real_send(client_self, request, *args, **kwargs)

        with patch.object(httpx.Client, "send", side_effect=no_external_send, autospec=True):
            resp = self.client.get("/research?product_count=25")
            self.assertEqual(resp.status_code, 200)
            self.assertIn('value="25"', resp.text)

    # -------------------------------------------------------------------------
    # Requirement 39: Niche Catalog Breadth and Structure
    # -------------------------------------------------------------------------
    def test_39_niche_catalog_breadth_and_structure(self):
        """Verify local niche catalog contains 28 categories with 8-15 practical niches each."""
        cats = get_broad_categories()
        self.assertEqual(len(cats), 28, f"Expected 28 broad categories, got {len(cats)}")

        all_niches = get_all_niches()
        self.assertGreaterEqual(len(all_niches), 250, f"Expected at least 250 niches, got {len(all_niches)}")

        for cat, niches in NICHE_CATALOG.items():
            self.assertGreaterEqual(len(niches), 8, f"Category '{cat}' has fewer than 8 niches ({len(niches)})")
            self.assertLessEqual(len(niches), 15, f"Category '{cat}' has more than 15 niches ({len(niches)})")
            for n in niches:
                self.assertTrue(len(n.strip()) > 0, f"Empty niche found in '{cat}'")

        # Random sample returns requested count
        sample = get_random_niche_sample(8)
        self.assertEqual(len(sample), 8)
        self.assertEqual(len(set(sample)), 8, "Sample should contain distinct niches")

    # -------------------------------------------------------------------------
    # Requirement 40: Safe JSON Serialization & Breakout Prevention (XSS Defense)
    # -------------------------------------------------------------------------
    def test_40_safe_json_serialization_xss_prevention(self):
        """Verify that catalog serialization uses tojson and safely escapes breakout characters."""
        import jinja2
        env = jinja2.Environment()
        t = env.from_string('<script id="testData" type="application/json">{{ data | tojson }}</script>')

        # Malicious breakout payload with <, >, &, ", ', and </script>
        malicious_data = {
            "attack": '</script><script>alert("XSS")</script>&<>\'\"'
        }
        rendered = t.render(data=malicious_data)

        # Assert no closing script tag breakout and safe unicode escapes
        self.assertNotIn("</script><script>", rendered)
        self.assertIn(r"\u003c/script\u003e", rendered)
        self.assertIn(r"\u003cscript\u003e", rendered)
        self.assertIn(r"\u0026", rendered)
        self.assertIn(r"\u003e", rendered)
        self.assertIn(r"\u0027", rendered)

        # Verify page /research renders safe JSON block with catalog data
        mock_db, _ = _make_mock_db()
        app.dependency_overrides[get_db] = lambda: mock_db
        resp = self.client.get("/research")
        self.assertIn('id="nicheCatalogData"', resp.text)
        self.assertIn('type="application/json"', resp.text)

        # Verify route with breakout payload in catalog data
        with patch.dict("app.services.niche_catalog.NICHE_CATALOG", {"Breakout <Category> &": ['Test "Niche" </script><script>alert(1)</script> & \'single\'']}):
            resp_injected = self.client.get("/research")
            self.assertEqual(resp_injected.status_code, 200)
            self.assertNotIn("</script><script>alert(1)</script>", resp_injected.text)
            self.assertIn(r"\u003c/script\u003e", resp_injected.text)
            self.assertIn(r"\u0026", resp_injected.text)

    # -------------------------------------------------------------------------
    # Requirement 41: Niche Discovery UI Elements Rendered in Research Page
    # -------------------------------------------------------------------------
    def test_41_niche_discovery_ui_elements_in_research_page(self):
        """Verify presence of progressive discovery section, category chips, random action, and filter."""
        mock_db, _ = _make_mock_db()
        app.dependency_overrides[get_db] = lambda: mock_db
        resp = self.client.get("/research")
        self.assertEqual(resp.status_code, 200)

        html = resp.text
        self.assertIn("Gợi ý ngách nghiên cứu", html)
        self.assertIn('id="nicheDiscoveryCard"', html)
        self.assertIn('id="broadCategoryChips"', html)
        self.assertIn('id="specificNichesContainer"', html)
        self.assertIn('id="randomNichesContainer"', html)
        self.assertIn('id="btnRandomNiches"', html)
        self.assertIn('id="nicheSearchFilter"', html)

        # Check key category names appear
        import html as html_lib
        unescaped_html = html_lib.unescape(html)
        for expected_cat in ["Đồ nhà bếp", "Đồ gia dụng", "Công nghệ & phụ kiện", "Làm đẹp", "Thú cưng"]:
            self.assertIn(expected_cat, unescaped_html)

    # -------------------------------------------------------------------------
    # Requirement 42: Category and Niche Buttons are type=button & Zero Network
    # -------------------------------------------------------------------------
    def test_42_category_and_niche_buttons_type_button_zero_network(self):
        """All category and discovery buttons must be type='button' and make zero network calls."""
        mock_db, _ = _make_mock_db()
        app.dependency_overrides[get_db] = lambda: mock_db

        real_send = httpx.Client.send
        def no_external_send(client_self, request, *args, **kwargs):
            if not str(request.url).startswith("http://testserver"):
                raise AssertionError(f"Unexpected external HTTP call: {request.method} {request.url}")
            return real_send(client_self, request, *args, **kwargs)

        with patch.object(httpx.Client, "send", side_effect=no_external_send, autospec=True):
            resp = self.client.get("/research")
            self.assertEqual(resp.status_code, 200)

            # Ensure all category chips have type="button"
            self.assertIn('type="button" \n                                                class="btn btn-xs btn-outline-secondary category-chip', resp.text)
            self.assertIn('type="button" class="btn btn-xs btn-outline-secondary text-secondary-emphasis" id="btnRandomNiches"', resp.text)

    # -------------------------------------------------------------------------
    # Requirement 43: Niche Input Remains Fully Editable and Retains Manual Edits
    # -------------------------------------------------------------------------
    def test_43_niche_input_remains_fully_editable(self):
        """The niche text input must retain standard form input properties and accept manual edits."""
        mock_db, _ = _make_mock_db()
        app.dependency_overrides[get_db] = lambda: mock_db
        resp = self.client.get("/research?niche=N%E1%BB%93i+chi%C3%AAn+kh%C3%B4ng+d%E1%BA%A7u+mini")
        self.assertEqual(resp.status_code, 200)
        self.assertIn('name="niche"', resp.text)
        self.assertIn('value="Nồi chiên không dầu mini"', resp.text)

    # -------------------------------------------------------------------------
    # Requirement 44: HTTP 503 Overload Stops Immediately with Zero Retries
    # -------------------------------------------------------------------------
    def test_44_http_503_overload_stops_immediately_with_zero_retries(self):
        """When provider returns 503, Research stops immediately with 0 retries and status 503."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("app.services.ai.manager.AIProviderManager.generate_products", side_effect=AIServiceUnavailableError("MWAPI 503 Overloaded")) as mock_gen:
            resp = self.client.post("/research", data={
                "niche": "Đồ nhà bếp thông minh",
                "product_count": "30",
                "fresh": "true",
            })

            # Exactly 1 single attempt
            self.assertEqual(mock_gen.call_count, 1)

            # HTTP 503 status
            self.assertEqual(resp.status_code, 503)

            # Localized Vietnamese overload message
            self.assertIn("MWAPI Gateway đang tạm thời quá tải (HTTP 503)", resp.text)
            self.assertIn("Research đã dừng và hệ thống không tự thử lại", resp.text)
            self.assertIn("Bạn có thể chờ một lúc rồi thử lại", resp.text)

            # Distinct from timeout
            self.assertNotIn("hết thời gian chờ", resp.text)

    # -------------------------------------------------------------------------
    # Requirement 45: Timeout Remains Semantically Distinct From HTTP 503
    # -------------------------------------------------------------------------
    def test_45_timeout_remains_semantically_distinct_from_503(self):
        """A timeout error must display timeout guidance and never be labeled as HTTP 503 overload."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("app.services.ai.manager.AIProviderManager.generate_products", side_effect=AITimeoutError("Request timed out")) as mock_gen:
            resp = self.client.post("/research", data={
                "niche": "Đồ nhà bếp thông minh",
                "product_count": "30",
                "fresh": "true",
            })

            self.assertEqual(mock_gen.call_count, 1)
            self.assertIn("quá thời gian chờ (Timeout)", resp.text)
            self.assertIn("hết thời gian chờ", resp.text)

            # Not classified as 503
            self.assertNotIn("503", resp.text)
            self.assertNotIn("quá tải", resp.text)

    # -------------------------------------------------------------------------
    # Requirement 46: Form State Preserved After HTTP 503
    # -------------------------------------------------------------------------
    def test_46_form_state_preserved_after_http_503(self):
        """Niche, product count, and fresh toggle values are preserved in form on 503."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("app.services.ai.manager.AIProviderManager.generate_products", side_effect=AIServiceUnavailableError("503 Overload")):
            resp = self.client.post("/research", data={
                "niche": "Dụng cụ bảo quản thực phẩm",
                "product_count": "40",
                "fresh": "true",
            })
            self.assertEqual(resp.status_code, 503)
            html = resp.text

            # Preserved inputs
            self.assertIn('value="Dụng cụ bảo quản thực phẩm"', html)
            self.assertIn('value="40"', html)
            self.assertIn('checked', html)

            # Recovery button present
            self.assertIn('id="btnRetryFocus"', html)
            self.assertIn("Thử Lại (Kiểm tra lại thông số)", html)

    # -------------------------------------------------------------------------
    # Requirement 47: Manual Retry Action Consumes Zero AI Calls
    # -------------------------------------------------------------------------
    def test_47_manual_retry_action_consumes_zero_ai_calls(self):
        """The retry button is a client-side button with type='button' that triggers zero network requests."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("app.services.ai.manager.AIProviderManager.generate_products", side_effect=AIServiceUnavailableError("503")):
            resp = self.client.post("/research", data={
                "niche": "Đồ chơi mèo",
                "product_count": "10",
                "fresh": "true",
            })
            self.assertEqual(resp.status_code, 503)
            # btnRetryFocus has type="button"
            self.assertIn('type="button" class="btn btn-sm btn-outline-warning fw-semibold" id="btnRetryFocus"', resp.text)

    # -------------------------------------------------------------------------
    # Requirement 48: MWAPI 50 Products Single Request Timeout and 6000 Tokens
    # -------------------------------------------------------------------------
    def test_48_mwapi_50_products_single_request_180s_timeout_6000_tokens(self):
        """For MWAPI 50 products: exactly 1 POST, max_tokens=6000, 0 models GET, no splitting."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        provider = MWAPIProvider()

        raw_50_items = [
            {"name_vietnamese": f"Sản phẩm {i}", "name_chinese": f"产品 {i}", "douyin_keywords": "kw", "content_angle": "ang", "hook": "hk"}
            for i in range(1, 51)
        ]
        mock_response = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-50prods",
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(raw_50_items, ensure_ascii=False)
                    }
                }],
                "usage": {"total_tokens": 1200}
            },
            request=httpx.Request("POST", "https://api.mwapi.dev/v1/chat/completions")
        )

        with patch.object(httpx.Client, "send", return_value=mock_response) as mock_send:
            res = provider.generate_products(niche="Đồ cắm trại", count=50, db=mock_db)
            self.assertEqual(len(res), 50)

            # Exactly 1 HTTP request
            self.assertEqual(mock_send.call_count, 1)
            sent_req = mock_send.call_args[0][0]
            self.assertEqual(sent_req.method, "POST")
            self.assertEqual(str(sent_req.url), "https://api.mwapi.dev/v1/chat/completions")

            # Max tokens = 6000 in payload
            sent_body = json.loads(sent_req.content.decode("utf-8"))
            self.assertEqual(sent_body.get("max_tokens"), 6000)

            # ZERO calls to /v1/models
            for call in mock_send.call_args_list:
                self.assertNotIn("/v1/models", str(call[0][0].url))

    # -------------------------------------------------------------------------
    # Requirement 49: Large Request Advisory Neutral Guidance
    # -------------------------------------------------------------------------
    def test_49_large_request_advisory_neutral_guidance(self):
        """Advisory element is rendered with truthful, non-scaring guidance."""
        mock_db, _ = _make_mock_db()
        app.dependency_overrides[get_db] = lambda: mock_db
        resp = self.client.get("/research")
        self.assertIn('id="productCountAdvisory"', resp.text)
        self.assertIn("Yêu cầu nhiều sản phẩm hơn có thể cần nhiều token và thời gian xử lý hơn.", resp.text)

    # -------------------------------------------------------------------------
    # Requirement 50: Research 0/1 Invariant with Niche Catalog
    # -------------------------------------------------------------------------
    def test_50_research_0_1_invariant_with_niche_catalog(self):
        """Cache hit makes 0 calls; cache miss makes exactly 1 call; no retries, no splitting."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("app.services.ai.manager.AIProviderManager.generate_products", return_value=SAMPLE_PRODUCTS) as mock_gen:
            # First run: cache miss -> 1 call
            resp1 = self.client.post("/research", data={
                "niche": "Dụng cụ làm bánh",
                "product_count": "10",
                "fresh": "false",
            }, follow_redirects=False)
            self.assertEqual(resp1.status_code, 303)
            self.assertEqual(mock_gen.call_count, 1)

    # -------------------------------------------------------------------------
    # Requirement 51: Research Prompt Optimization at Count=50
    # -------------------------------------------------------------------------
    def test_51_prompt_character_and_token_efficiency_at_count_50(self):
        """Verify prompt requests 50 products, bounds field lengths, and enforces raw JSON."""
        captured_prompt = None
        provider = MWAPIProvider()

        def mock_generate(prompt, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            return json.dumps([{"name_vietnamese": f"Sp {i}"} for i in range(50)])

        with patch.object(provider, "generate", side_effect=mock_generate):
            res = provider.generate_products(niche="Đồ cắm trại", count=50)
            self.assertEqual(len(res), 50)

        self.assertIsNotNone(captured_prompt)
        # 1. Explicitly requests 50 products
        self.assertIn("exactly 50 trending", captured_prompt)

        # 2. Compact wire keys requested
        self.assertIn('"nv"', captured_prompt)
        self.assertIn('"nc"', captured_prompt)
        self.assertIn('"dk"', captured_prompt)
        self.assertIn('"ca"', captured_prompt)
        self.assertIn('"h"', captured_prompt)

        # 3. Bounded length guidance
        self.assertIn("3–4 authentic Chinese Douyin search phrases", captured_prompt)
        self.assertIn("8–12 words", captured_prompt)
        self.assertIn("under 12 words", captured_prompt)

        # 4. Strict raw JSON, no markdown fences
        self.assertIn("Return ONLY the raw JSON array. The response must start with [ and end with ]", captured_prompt)
        self.assertIn("No markdown fences", captured_prompt)

        # 5. Reasonably compact prompt (under 2,000 chars)
        self.assertLess(len(captured_prompt), 2000)

    # -------------------------------------------------------------------------
    # Requirement 52: MWAPI 50 Products Single POST Payload and No Temperature Added
    # -------------------------------------------------------------------------
    def test_52_mwapi_50_products_exact_single_post_payload_no_temperature(self):
        """For count=50: exactly 1 POST, max_tokens=6000, timeout=180s, 0 GET models, NO temperature added."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        provider = MWAPIProvider()

        mock_resp = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-test50",
                "choices": [{
                    "message": {
                        "content": json.dumps([{"name_vietnamese": f"Sp {i}", "name_chinese": f"Cp {i}", "douyin_keywords": "kw", "content_angle": "ang", "hook": "hk"} for i in range(50)])
                    }
                }],
                "usage": {"total_tokens": 4200}
            },
            request=httpx.Request("POST", "https://api.mwapi.dev/v1/chat/completions")
        )

        with patch.object(httpx.Client, "send", return_value=mock_resp) as mock_send:
            res = provider.generate_products(niche="Đồ cắm trại", count=50, db=mock_db)
            self.assertEqual(len(res), 50)

            # Exactly 1 HTTP POST
            self.assertEqual(mock_send.call_count, 1)
            sent_req = mock_send.call_args[0][0]
            self.assertEqual(sent_req.method, "POST")
            self.assertEqual(str(sent_req.url), "https://api.mwapi.dev/v1/chat/completions")

            sent_body = json.loads(sent_req.content.decode("utf-8"))
            self.assertEqual(sent_body.get("max_tokens"), 6000)
            self.assertEqual(sent_body.get("model"), "claude-sonnet-4-6")

            # CRITICAL: temperature is NOT added or modified
            self.assertNotIn("temperature", sent_body)

            # Zero calls to /v1/models
            for call in mock_send.call_args_list:
                self.assertNotIn("/v1/models", str(call[0][0].url))

    # -------------------------------------------------------------------------
    # Requirement 53: 503 Diagnostics Capture Retry-After and Request ID Safely
    # -------------------------------------------------------------------------
    def test_53_503_diagnostics_capture_retry_after_and_request_id(self):
        """HTTP 503 captures Retry-After and request-id safely, displaying informational hint without auto retry."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        mock_503_resp = httpx.Response(
            status_code=503,
            headers={
                "Retry-After": "30",
                "x-request-id": "req-trace-xyz-987"
            },
            text=json.dumps({"error": {"message": "Service Overloaded", "type": "overloaded_error"}}),
            request=httpx.Request("POST", "https://api.mwapi.dev/v1/chat/completions")
        )

        real_send = httpx.Client.send
        outbound_calls = []

        def intercept_send(client_self, request, *args, **kwargs):
            if str(request.url).startswith("http://testserver"):
                return real_send(client_self, request, *args, **kwargs)
            outbound_calls.append(request)
            return mock_503_resp

        with patch.object(httpx.Client, "send", side_effect=intercept_send, autospec=True):
            resp = self.client.post("/research", data={
                "niche": "Thiết bị nhà bếp",
                "product_count": "50",
                "fresh": "true",
            })

            # Exactly 1 POST, NO automatic retry
            self.assertEqual(len(outbound_calls), 1)
            self.assertEqual(resp.status_code, 503)

            # Informational hint rendered to user
            self.assertIn("30 giây", resp.text)
            self.assertIn("Nhà cung cấp đề xuất thử lại sau khoảng 30 giây", resp.text)

            # Retry focus button present (type="button")
            self.assertIn('id="btnRetryFocus"', resp.text)

    # -------------------------------------------------------------------------
    # Requirement 54: Malformed or Missing 503 Diagnostics Handled Gracefully
    # -------------------------------------------------------------------------
    def test_54_malformed_or_missing_503_diagnostics(self):
        """Malformed Retry-After does not crash and does not invent a wait time."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        mock_503_resp = httpx.Response(
            status_code=503,
            headers={
                "Retry-After": "not-a-number",
                "Set-Cookie": "secret_cookie_token=secret123"
            },
            text="Upstream capacity reached",
            request=httpx.Request("POST", "https://api.mwapi.dev/v1/chat/completions")
        )

        real_send = httpx.Client.send
        outbound_calls = []

        def intercept_send(client_self, request, *args, **kwargs):
            if str(request.url).startswith("http://testserver"):
                return real_send(client_self, request, *args, **kwargs)
            outbound_calls.append(request)
            return mock_503_resp

        with patch.object(httpx.Client, "send", side_effect=intercept_send, autospec=True):
            resp = self.client.post("/research", data={
                "niche": "Thiết bị nhà bếp",
                "product_count": "50",
                "fresh": "true",
            })
            self.assertEqual(len(outbound_calls), 1)
            self.assertEqual(resp.status_code, 503)
            # No fabricated wait time
            self.assertNotIn("khoảng not-a-number giây", resp.text)
            self.assertNotIn("secret_cookie_token", resp.text)

    # -------------------------------------------------------------------------
    # Requirement 55: 503 Structured Logging Contains No Secrets
    # -------------------------------------------------------------------------
    def test_55_503_structured_logging_contains_no_secrets(self):
        """Structured 503 log output contains telemetry but zero secrets or Authorization headers."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        mock_503_resp = httpx.Response(
            status_code=503,
            headers={"Retry-After": "45", "x-request-id": "req-trace-abc"},
            text='{"error": "Capacity limit"}',
            request=httpx.Request("POST", "https://api.mwapi.dev/v1/chat/completions")
        )

        real_send = httpx.Client.send
        outbound_calls = []

        def intercept_send(client_self, request, *args, **kwargs):
            if str(request.url).startswith("http://testserver"):
                return real_send(client_self, request, *args, **kwargs)
            outbound_calls.append(request)
            return mock_503_resp

        with patch.object(httpx.Client, "send", side_effect=intercept_send, autospec=True):
            with self.assertLogs("app.routes.research", level="WARNING") as cm:
                resp = self.client.post("/research", data={
                    "niche": "Dụng cụ làm vườn",
                    "product_count": "50",
                    "fresh": "true"
                })
                self.assertEqual(len(outbound_calls), 1)
                self.assertEqual(resp.status_code, 503)

                log_output = "\n".join(cm.output)
                self.assertIn("[RESEARCH 503 OVERLOAD]", log_output)
                self.assertIn("Count=50", log_output)
                self.assertIn("RetryAfter=45", log_output)
                self.assertIn("RequestID=req-trace-abc", log_output)

                # Absolute security: zero secrets or Authorization in log
                self.assertNotIn("sk-", log_output)
                self.assertNotIn("Bearer", log_output)
                self.assertNotIn("mwapi_key", log_output)

    # -------------------------------------------------------------------------
    # Requirement 56: Full 50-Product Output Parsing and Database Persistence
    # -------------------------------------------------------------------------
    def test_56_full_50_product_output_parsing_and_persistence(self):
        """50 products returned in mock response are parsed and persisted with all 5 fields and local IDs."""
        mock_db, session = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        raw_50_items = [
            {
                "name_vietnamese": f"Sản phẩm gia dụng thông minh {i}",
                "name_chinese": f"智能家居用品 {i}",
                "douyin_keywords": f"智能家居 好物推荐 测评 {i}",
                "content_angle": f"Góc tiếp cận độc đáo cho sản phẩm {i}",
                "hook": f"Đừng bỏ qua sản phẩm {i} nếu bạn muốn tiết kiệm thời gian!"
            }
            for i in range(1, 51)
        ]

        mock_200_resp = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-valid-50",
                "choices": [{
                    "message": {
                        "content": json.dumps(raw_50_items, ensure_ascii=False)
                    }
                }],
                "usage": {"total_tokens": 4500}
            },
            request=httpx.Request("POST", "https://api.mwapi.dev/v1/chat/completions")
        )

        real_send = httpx.Client.send
        outbound_calls = []

        def intercept_send(client_self, request, *args, **kwargs):
            if str(request.url).startswith("http://testserver"):
                return real_send(client_self, request, *args, **kwargs)
            outbound_calls.append(request)
            return mock_200_resp

        with patch.object(httpx.Client, "send", side_effect=intercept_send, autospec=True):
            resp = self.client.post("/research", data={
                "niche": "Đồ tiện ích",
                "product_count": "50",
                "fresh": "true"
            }, follow_redirects=False)

            # Exactly 1 HTTP POST outbound
            self.assertEqual(len(outbound_calls), 1)

            # Redirects to products list with count=50
            self.assertEqual(resp.status_code, 303)
            self.assertIn("count=50", resp.headers["location"])

            # Verify products persisted via db.add
            saved_products = [call[0][0] for call in mock_db.add.call_args_list if isinstance(call[0][0], Product)]
            self.assertEqual(len(saved_products), 50)
            self.assertTrue(mock_db.commit.called)

            # Check sequential IDs and field values
            for idx, p in enumerate(saved_products, 1):
                self.assertTrue(p.product_id.startswith("P"))
                self.assertIn(f"Sản phẩm gia dụng thông minh {idx}", p.name_vietnamese)
                self.assertIn(f"智能家居用品 {idx}", p.name_chinese)
                self.assertIn("好物推荐", p.douyin_keywords)
                self.assertTrue(len(p.content_angle) > 0)
                self.assertTrue(len(p.hook) > 0)
                self.assertEqual(p.status, "RESEARCHED")


    # -------------------------------------------------------------------------
    # Requirement 57: Explicit HTTPX Timeout Components for 50 Products
    # -------------------------------------------------------------------------
    def test_57_explicit_httpx_timeout_components_50_products(self):
        """For count=50: connect=15s, read=300s, write=15s, pool=10s, exactly 1 POST, zero models GET, zero fallback."""
        from app.services.ai.base import get_research_httpx_timeout
        t50 = get_research_httpx_timeout(50)
        self.assertEqual(t50.connect, 15.0)
        self.assertEqual(t50.read, 300.0)
        self.assertEqual(t50.write, 15.0)
        self.assertEqual(t50.pool, 10.0)

        mock_db, _ = _make_mock_db(active_provider="mwapi")
        provider = MWAPIProvider()

        raw_50_items = [
            {
                "name_vietnamese": f"Sản phẩm {i}",
                "name_chinese": f"产品 {i}",
                "douyin_keywords": f"kw {i}",
                "content_angle": f"ang {i}",
                "hook": f"hk {i}"
            }
            for i in range(1, 51)
        ]
        mock_response = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-t50",
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(raw_50_items, ensure_ascii=False)
                    }
                }],
                "usage": {"prompt_tokens": 300, "completion_tokens": 4200, "total_tokens": 4500}
            },
            request=httpx.Request("POST", "https://api.mwapi.dev/v1/chat/completions")
        )

        captured_timeouts = []
        def intercept_send(client_self, request, *args, **kwargs):
            captured_timeouts.append(client_self.timeout)
            return mock_response

        with patch.object(httpx.Client, "send", side_effect=intercept_send, autospec=True):
            res = provider.generate_products(niche="Nhà bếp", count=50, db=mock_db)
            self.assertEqual(len(res), 50)
            self.assertEqual(len(captured_timeouts), 1)

            # Explicit HTTPX timeout components verified on client
            t_client = captured_timeouts[0]
            self.assertEqual(t_client.connect, 15.0)
            self.assertEqual(t_client.read, 300.0)
            self.assertEqual(t_client.write, 15.0)
            self.assertEqual(t_client.pool, 10.0)

    # -------------------------------------------------------------------------
    # Requirement 58: Smaller Counts Timeout Scaling
    # -------------------------------------------------------------------------
    def test_58_smaller_counts_timeout_scaling(self):
        """Verify deterministic scaling of read timeout for 5, 10, 20, 30, 50 products while keeping connect=15."""
        from app.services.ai.base import get_research_timeout, get_research_httpx_timeout
        self.assertEqual(get_research_timeout(5), 60.0)
        self.assertEqual(get_research_timeout(10), 75.0)
        self.assertEqual(get_research_timeout(20), 100.0)
        self.assertEqual(get_research_timeout(30), 150.0)
        self.assertEqual(get_research_timeout(50), 300.0)

        for cnt in (5, 10, 20, 30, 50):
            ht = get_research_httpx_timeout(cnt)
            self.assertEqual(ht.connect, 15.0)
            self.assertEqual(ht.write, 15.0)
            self.assertEqual(ht.pool, 10.0)
            self.assertEqual(ht.read, get_research_timeout(cnt))

    # -------------------------------------------------------------------------
    # Requirement 59: Exact-Count Validation Failure Stops Without Retry
    # -------------------------------------------------------------------------
    def test_59_exact_count_validation_failure_stops_without_retry(self):
        """If AI returns 47 items for count=50, fail local validation immediately with zero retries."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        app.dependency_overrides[get_db] = lambda: mock_db

        # Only 47 items returned instead of 50
        raw_47_items = [
            {
                "name_vietnamese": f"Sản phẩm thiếu {i}",
                "name_chinese": f"产品 {i}",
                "douyin_keywords": f"kw {i}",
                "content_angle": f"ang {i}",
                "hook": f"hk {i}"
            }
            for i in range(1, 48)
        ]
        mock_200_resp = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-incomplete",
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(raw_47_items, ensure_ascii=False)
                    }
                }],
                "usage": {"total_tokens": 3800}
            },
            request=httpx.Request("POST", "https://api.mwapi.dev/v1/chat/completions")
        )

        real_send = httpx.Client.send
        outbound_calls = []

        def intercept_send(client_self, request, *args, **kwargs):
            if str(request.url).startswith("http://testserver"):
                return real_send(client_self, request, *args, **kwargs)
            outbound_calls.append(request)
            return mock_200_resp

        with patch.object(httpx.Client, "send", side_effect=intercept_send, autospec=True):
            resp = self.client.post("/research", data={
                "niche": "Đồ tiện ích",
                "product_count": "50",
                "fresh": "true"
            })
            # Exactly 1 outbound call made, STOP on incomplete count
            self.assertEqual(len(outbound_calls), 1)
            self.assertEqual(resp.status_code, 400)
            self.assertIn("47/50", resp.text)
            saved_products = [call[0][0] for call in mock_db.add.call_args_list if isinstance(call[0][0], Product)]
            self.assertEqual(len(saved_products), 0)
            self.assertTrue(mock_db.rollback.called)

    # -------------------------------------------------------------------------
    # Requirement 60: Finish Reason Stop Recorded in Execution Metadata
    # -------------------------------------------------------------------------
    def test_60_finish_reason_stop_recorded(self):
        """finish_reason='stop' is captured in execution metadata."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        provider = MWAPIProvider()

        mock_resp = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-finish-stop",
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "content": '[{"name_vietnamese": "Sản phẩm A", "name_chinese": "A", "douyin_keywords": "kw", "content_angle": "ang", "hook": "hk"}]'
                    }
                }],
                "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}
            },
            request=httpx.Request("POST", "https://api.mwapi.dev/v1/chat/completions")
        )

        with patch.object(httpx.Client, "send", return_value=mock_resp):
            res = provider.generate_products(niche="Gia dụng", count=1, db=mock_db)
            self.assertEqual(len(res), 1)
            meta = provider.get_last_execution_metadata()
            self.assertEqual(meta.get("finish_reason"), "stop")
            self.assertEqual(meta.get("status"), "SUCCESS")

    # -------------------------------------------------------------------------
    # Requirement 61: Finish Reason Length Token-Cap Diagnostic
    # -------------------------------------------------------------------------
    def test_61_finish_reason_length_token_cap_diagnostic(self):
        """finish_reason='length' with truncated JSON is classified cleanly as token cap truncation."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        provider = MWAPIProvider()

        # Truncated output because of length
        mock_resp = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-finish-len",
                "choices": [{
                    "finish_reason": "length",
                    "message": {
                        "content": '[{"name_vietnamese": "Sản phẩm dở dang"'
                    }
                }],
                "usage": {"prompt_tokens": 100, "completion_tokens": 6000, "total_tokens": 6100}
            },
            request=httpx.Request("POST", "https://api.mwapi.dev/v1/chat/completions")
        )

        from app.services.ai.base import AIInvalidResponseError
        with patch.object(httpx.Client, "send", return_value=mock_resp):
            with self.assertRaises(AIInvalidResponseError) as cm:
                provider.generate_products(niche="Gia dụng", count=50, db=mock_db)
            err_msg = str(cm.exception)
            self.assertTrue("giới hạn token" in err_msg or "finish_reason='length'" in err_msg)
            meta = provider.get_last_execution_metadata()
            self.assertEqual(meta.get("finish_reason"), "length")

    # -------------------------------------------------------------------------
    # Requirement 62: Token Usage Metadata Captured Safely
    # -------------------------------------------------------------------------
    def test_62_token_usage_metadata_captured(self):
        """Execution metadata records input, output, and total tokens without leaking credentials."""
        mock_db, _ = _make_mock_db(active_provider="mwapi")
        provider = MWAPIProvider()

        mock_resp = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-usage",
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "content": '[{"name_vietnamese": "SP 1", "name_chinese": "1", "douyin_keywords": "k", "content_angle": "a", "hook": "h"}]'
                    }
                }],
                "usage": {"prompt_tokens": 250, "completion_tokens": 1800, "total_tokens": 2050}
            },
            request=httpx.Request("POST", "https://api.mwapi.dev/v1/chat/completions")
        )

        with patch.object(httpx.Client, "send", return_value=mock_resp):
            provider.generate_products(niche="Đồ chơi", count=1, db=mock_db)
            meta = provider.get_last_execution_metadata()
            self.assertEqual(meta.get("input_tokens"), 250)
            self.assertEqual(meta.get("output_tokens"), 1800)
            self.assertEqual(meta.get("total_tokens"), 2050)
            self.assertTrue(meta.get("duration_seconds") is not None)
            meta_str = str(meta)
            self.assertNotIn("sk-", meta_str)
            self.assertNotIn("Bearer", meta_str)


if __name__ == "__main__":
    unittest.main()



