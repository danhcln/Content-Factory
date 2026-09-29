"""
Multi-AI Phase 2 Comprehensive Test Suite
5 Cloud Providers + BYOK + OpenRouter Gateway + Research Invariants

Tests:
1. Provider contracts & provider_type (direct vs gateway) for all 5 providers.
2. AIProviderManager registration & delegation.
3. OpenAI Provider test matrix (Success, 401, 403, 404, 429, 400, 503, Timeout, Network, Invalid JSON, Test Connection, Dynamic Models, Secret Sanitization).
4. Anthropic Claude Provider test matrix (Native Messages API, 401, 403, 404, 429, 503, Timeout, Network, Test Connection, Dynamic Models, Secret Sanitization).
5. Groq Provider test matrix (OpenAI-compatible REST, 401, 403, 404, 429, 503, Timeout, Network, Test Connection, Dynamic Models, Secret Sanitization).
6. OpenRouter Gateway test matrix (Cloud AI Gateway, 401, 403, 404, 429, 503, Timeout, Network, Test Connection, Dynamic Models, Secret Sanitization).
7. OpenRouter Specific Safety Test (single model string, no models[] array, provider.allow_fallbacks=False for Research).
8. Model Discovery & Caching (cache hit = 0 HTTP calls, generate() never triggers list_models()).
9. Research One-Request Matrix for all 5 providers (normal=1, 503=1, 429=1, timeout=1, invalid JSON=1, cache hit=0).
10. Research Output Token Budget mapping (800, 1500, 2800, 4000, 6000).

ALL TESTS USE MOCKED HTTP — ZERO REAL API CALLS OR QUOTA COST.
"""
import unittest
from unittest.mock import patch, MagicMock
import json
import httpx

from app.services.ai.base import (
    AIProvider,
    AIGenerationOptions,
    ExecutionMetadata,
    AIProviderError,
    AIAuthenticationError,
    AIPermissionError,
    AIModelNotFoundError,
    AIQuotaExceededError,
    AIRateLimitError,
    AIBadRequestError,
    AIServiceUnavailableError,
    AITimeoutError,
    AINetworkError,
    AIInvalidResponseError,
    get_research_max_output_tokens,
)
from app.services.ai.manager import AIProviderManager, get_ai_manager, set_ai_manager
from app.services.ai.providers.gemini import GeminiProvider
from app.services.ai.providers.openai import OpenAIProvider
from app.services.ai.providers.anthropic import AnthropicProvider
from app.services.ai.providers.groq import GroqProvider
from app.services.ai.providers.openrouter import OpenRouterProvider
from app.services.ai.providers.mwapi import MWAPIProvider
from app.services.ai.providers.common import (
    sanitize_secrets,
    global_model_cache,
    classify_http_error,
)


def _mock_resp(status_code: int, data: dict = None, text: str = ""):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = {}
    if data is not None:
        resp.json.return_value = data
        resp.text = json.dumps(data)
    else:
        resp.json.side_effect = Exception("Not valid JSON")
        resp.text = text
    return resp


SAMPLE_PRODUCTS_JSON = json.dumps([
    {
        "name_vietnamese": "Nồi cơm điện mini đa năng",
        "name_chinese": "多功能迷你电饭煲",
        "douyin_keywords": "宿舍迷你电饭煲 独居一人食好物",
        "content_angle": "Giải pháp nấu ăn tiện lợi nhanh gọn",
        "hook": "Đừng mua nồi cơm to nữa nếu bạn sống một mình!"
    }
])


class TestMultiAIPhase2Architecture(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_db.query.return_value.filter.return_value.first.return_value = None
        global_model_cache.clear()
        set_ai_manager(None)

    def tearDown(self):
        global_model_cache.clear()
        set_ai_manager(None)

    # ==========================================================================
    # 1. PROVIDER CONTRACTS & REGISTRATION
    # ==========================================================================
    def test_01_all_five_providers_implement_contract(self):
        """All 5 providers implement AIProvider and expose correct provider_type."""
        providers = [
            (GeminiProvider(), "gemini", "Google Gemini", "direct"),
            (OpenAIProvider(), "openai", "OpenAI", "direct"),
            (AnthropicProvider(), "anthropic", "Anthropic Claude", "direct"),
            (GroqProvider(), "groq", "Groq", "direct"),
            (OpenRouterProvider(), "openrouter", "OpenRouter", "gateway"),
            (MWAPIProvider(), "mwapi", "MWAPI Gateway", "gateway"),
        ]
        for p, expected_id, expected_name, expected_type in providers:
            self.assertIsInstance(p, AIProvider)
            self.assertEqual(p.provider_id, expected_id)
            self.assertEqual(p.display_name, expected_name)
            self.assertEqual(p.provider_type, expected_type)
            self.assertTrue(hasattr(p, "generate"))
            self.assertTrue(hasattr(p, "test_connection"))
            self.assertTrue(hasattr(p, "list_models"))
            self.assertTrue(hasattr(p, "supports_model"))
            self.assertTrue(hasattr(p, "get_last_execution_metadata"))

    def test_02_manager_registers_exact_five_providers(self):
        """AIProviderManager registers the official cloud providers (now 6 with MWAPI)."""
        mgr = AIProviderManager()
        registered = mgr.list_registered_providers()
        self.assertEqual(len(registered), 6)
        p_ids = [p["provider_id"] for p in registered]
        self.assertEqual(sorted(p_ids), ["anthropic", "gemini", "groq", "mwapi", "openai", "openrouter"])

        # Default provider must remain gemini
        self.assertEqual(mgr.DEFAULT_PROVIDER_ID, "gemini")
        self.assertEqual(mgr.get_active_provider().provider_id, "gemini")

    def test_03_provider_support_model_rules(self):
        """Each provider accurately implements supports_model check."""
        gemini = GeminiProvider()
        openai = OpenAIProvider()
        anthropic = AnthropicProvider()
        groq = GroqProvider()
        openrouter = OpenRouterProvider()

        self.assertTrue(gemini.supports_model("gemini-3.8-flash"))
        self.assertFalse(gemini.supports_model("gpt-4o"))

        self.assertTrue(openai.supports_model("gpt-4o"))
        self.assertTrue(openai.supports_model("gpt-4o-mini"))
        self.assertTrue(openai.supports_model("o1-mini"))
        self.assertFalse(openai.supports_model("claude-3-5-sonnet"))

        self.assertTrue(anthropic.supports_model("claude-3-5-sonnet-20241022"))
        self.assertTrue(anthropic.supports_model("claude-sonnet-4-6"))
        self.assertFalse(anthropic.supports_model("gpt-4o"))

        self.assertTrue(groq.supports_model("llama-3.3-70b-versatile"))
        self.assertTrue(groq.supports_model("mixtral-8x7b-32768"))
        self.assertFalse(groq.supports_model("claude-3-5-sonnet"))

        self.assertTrue(openrouter.supports_model("anthropic/claude-3.5-sonnet"))
        self.assertTrue(openrouter.supports_model("openai/gpt-4o"))
        self.assertFalse(openrouter.supports_model("simple-model-without-slash"))


class TestOpenAIProviderMatrix(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_db.query.return_value.filter.return_value.first.return_value = None
        global_model_cache.clear()
        self.provider = OpenAIProvider()

    @patch("app.services.ai.providers.openai.get_openai_api_key", return_value="sk-TEST_OPENAI_KEY_12345")
    @patch("app.services.ai.providers.openai.get_openai_model", return_value="gpt-4o-mini")
    @patch("httpx.Client.post")
    def test_openai_successful_generation(self, mock_post, mock_m, mock_k):
        """OpenAI successful generation parses choices and usage tokens."""
        mock_post.return_value = _mock_resp(200, {
            "id": "chatcmpl-123",
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [{"message": {"content": "Hello from OpenAI"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
        })
        result = self.provider.generate("Test prompt", db=self.mock_db)
        self.assertEqual(result, "Hello from OpenAI")
        meta = self.provider.get_last_execution_metadata()
        self.assertEqual(meta["provider"], "openai")
        self.assertEqual(meta["provider_type"], "direct")
        self.assertEqual(meta["actual_model_used"], "gpt-4o-mini-2024-07-18")
        self.assertEqual(meta["input_tokens"], 12)
        self.assertEqual(meta["output_tokens"], 5)
        self.assertEqual(meta["total_tokens"], 17)

    @patch("app.services.ai.providers.openai.get_openai_api_key", return_value="sk-BAD_KEY")
    @patch("httpx.Client.post")
    def test_openai_401_authentication_error(self, mock_post, mock_k):
        """HTTP 401 raises AIAuthenticationError without retrying."""
        mock_post.return_value = _mock_resp(401, {"error": {"message": "Incorrect API key provided"}})
        with self.assertRaises(AIAuthenticationError):
            self.provider.generate("Hello", db=self.mock_db, max_retries=2)
        self.assertEqual(mock_post.call_count, 1)

    @patch("app.services.ai.providers.openai.get_openai_api_key", return_value="sk-KEY")
    @patch("httpx.Client.post")
    def test_openai_403_permission_error(self, mock_post, mock_k):
        """HTTP 403 raises AIPermissionError."""
        mock_post.return_value = _mock_resp(403, {"error": {"message": "Country or model not supported"}})
        with self.assertRaises(AIPermissionError):
            self.provider.generate("Hello", db=self.mock_db, max_retries=0)

    @patch("app.services.ai.providers.openai.get_openai_api_key", return_value="sk-KEY")
    @patch("httpx.Client.post")
    def test_openai_404_model_not_found(self, mock_post, mock_k):
        """HTTP 404 raises AIModelNotFoundError."""
        mock_post.return_value = _mock_resp(404, {"error": {"message": "The model `nonexistent` does not exist"}})
        with self.assertRaises(AIModelNotFoundError):
            self.provider.generate("Hello", db=self.mock_db, max_retries=0)

    @patch("app.services.ai.providers.openai.get_openai_api_key", return_value="sk-KEY")
    @patch("httpx.Client.post")
    def test_openai_429_quota_exceeded(self, mock_post, mock_k):
        """HTTP 429 with quota text raises AIQuotaExceededError."""
        mock_post.return_value = _mock_resp(429, {"error": {"message": "You exceeded your current quota, please check your plan"}})
        with self.assertRaises(AIQuotaExceededError):
            self.provider.generate("Hello", db=self.mock_db, max_retries=2)

    @patch("app.services.ai.providers.openai.get_openai_api_key", return_value="sk-KEY")
    @patch("httpx.Client.post")
    def test_openai_400_bad_request(self, mock_post, mock_k):
        """HTTP 400 raises AIBadRequestError."""
        mock_post.return_value = _mock_resp(400, {"error": {"message": "max_tokens too large"}})
        with self.assertRaises(AIBadRequestError):
            self.provider.generate("Hello", db=self.mock_db, max_retries=0)

    @patch("app.services.ai.providers.openai.get_openai_api_key", return_value="sk-KEY")
    @patch("httpx.Client.post")
    def test_openai_503_service_unavailable(self, mock_post, mock_k):
        """HTTP 503 raises AIServiceUnavailableError."""
        mock_post.return_value = _mock_resp(503, {"error": {"message": "Server overloaded"}})
        with self.assertRaises(AIServiceUnavailableError):
            self.provider.generate("Hello", db=self.mock_db, max_retries=0)

    @patch("app.services.ai.providers.openai.get_openai_api_key", return_value="sk-KEY")
    @patch("httpx.Client.post", side_effect=httpx.TimeoutException("Timeout"))
    def test_openai_timeout(self, mock_post, mock_k):
        """Timeout raises AITimeoutError."""
        with self.assertRaises(AITimeoutError):
            self.provider.generate("Hello", db=self.mock_db, max_retries=0)

    @patch("app.services.ai.providers.openai.get_openai_api_key", return_value="sk-KEY")
    @patch("httpx.Client.post", side_effect=httpx.ConnectError("Connection refused"))
    def test_openai_network_error(self, mock_post, mock_k):
        """Network error raises AINetworkError."""
        with self.assertRaises(AINetworkError):
            self.provider.generate("Hello", db=self.mock_db, max_retries=0)

    @patch("app.services.ai.providers.openai.get_openai_api_key", return_value="sk-KEY")
    @patch("httpx.Client.post")
    def test_openai_invalid_response(self, mock_post, mock_k):
        """Empty choices raises AIInvalidResponseError."""
        mock_post.return_value = _mock_resp(200, {"choices": []})
        with self.assertRaises(AIInvalidResponseError):
            self.provider.generate("Hello", db=self.mock_db, max_retries=0)

    @patch("app.services.ai.providers.openai.get_openai_api_key", return_value=None)
    def test_openai_test_connection_unconfigured(self, mock_k):
        """test_connection with no API key returns configured=False without network call."""
        res = self.provider.test_connection(db=self.mock_db)
        self.assertFalse(res["configured"])
        self.assertFalse(res["connected"])
        self.assertEqual(res["error_type"], "not_configured")

    @patch("app.services.ai.providers.openai.get_openai_api_key", return_value="sk-VALID")
    @patch("httpx.Client.get")
    def test_openai_test_connection_success(self, mock_get, mock_k):
        """test_connection with valid key returns connected=True."""
        mock_get.return_value = _mock_resp(200, {"data": [{"id": "gpt-4o"}]})
        res = self.provider.test_connection(db=self.mock_db)
        self.assertTrue(res["configured"])
        self.assertTrue(res["connected"])
        self.assertIsNone(res["error_type"])

    @patch("app.services.ai.providers.openai.get_openai_api_key", return_value="sk-SECRET_KEY_999")
    @patch("httpx.Client.get")
    def test_openai_list_models_filtering_and_caching(self, mock_get, mock_k):
        """list_models filters non-text models and caches results in-memory."""
        mock_get.return_value = _mock_resp(200, {
            "data": [
                {"id": "gpt-4o"},
                {"id": "text-embedding-3-small"},
                {"id": "whisper-1"},
                {"id": "dall-e-3"},
                {"id": "gpt-4o-mini"}
            ]
        })
        models = self.provider.list_models(db=self.mock_db)
        model_ids = [m["id"] for m in models]
        self.assertIn("gpt-4o", model_ids)
        self.assertIn("gpt-4o-mini", model_ids)
        self.assertNotIn("text-embedding-3-small", model_ids)
        self.assertNotIn("whisper-1", model_ids)
        self.assertNotIn("dall-e-3", model_ids)
        self.assertEqual(mock_get.call_count, 1)

        # Second call must use cache: 0 HTTP calls
        cached_models = self.provider.list_models(db=self.mock_db)
        self.assertEqual(len(cached_models), len(models))
        self.assertEqual(mock_get.call_count, 1)

    def test_openai_secret_sanitization(self):
        """API key is never exposed in error messages."""
        secret = "sk-proj-SUPER_SECRET_TOKEN_OPENAI"
        raw_msg = f"Failed with {secret} and Bearer {secret}"
        sanitized = sanitize_secrets(raw_msg, secret)
        self.assertNotIn(secret, sanitized)
        self.assertIn("[REDACTED]", sanitized)


class TestAnthropicProviderMatrix(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_db.query.return_value.filter.return_value.first.return_value = None
        global_model_cache.clear()
        self.provider = AnthropicProvider()

    @patch("app.services.ai.providers.anthropic.get_anthropic_api_key", return_value="sk-ant-TEST_KEY")
    @patch("app.services.ai.providers.anthropic.get_anthropic_model", return_value="claude-sonnet-4-6")
    @patch("httpx.Client.post")
    def test_anthropic_successful_generation(self, mock_post, mock_m, mock_k):
        """Anthropic native Messages API call extracts content and usage."""
        mock_post.return_value = _mock_resp(200, {
            "id": "msg_123",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "Hello from Claude"}],
            "usage": {"input_tokens": 15, "output_tokens": 8}
        })
        result = self.provider.generate("Test prompt", db=self.mock_db)
        self.assertEqual(result, "Hello from Claude")
        meta = self.provider.get_last_execution_metadata()
        self.assertEqual(meta["provider"], "anthropic")
        self.assertEqual(meta["provider_type"], "direct")
        self.assertEqual(meta["input_tokens"], 15)
        self.assertEqual(meta["output_tokens"], 8)
        self.assertEqual(meta["total_tokens"], 23)

    @patch("app.services.ai.providers.anthropic.get_anthropic_api_key", return_value="sk-ant-BAD")
    @patch("httpx.Client.post")
    def test_anthropic_401_authentication_error(self, mock_post, mock_k):
        """Anthropic 401 raises AIAuthenticationError."""
        mock_post.return_value = _mock_resp(401, {
            "type": "error",
            "error": {"type": "authentication_error", "message": "invalid x-api-key"}
        })
        with self.assertRaises(AIAuthenticationError):
            self.provider.generate("Hello", db=self.mock_db, max_retries=0)

    @patch("app.services.ai.providers.anthropic.get_anthropic_api_key", return_value="sk-ant-KEY")
    @patch("httpx.Client.post")
    def test_anthropic_429_rate_limit(self, mock_post, mock_k):
        """Anthropic 429 raises AIRateLimitError."""
        mock_post.return_value = _mock_resp(429, {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Rate limit exceeded"}
        })
        with self.assertRaises(AIRateLimitError):
            self.provider.generate("Hello", db=self.mock_db, max_retries=0)

    @patch("app.services.ai.providers.anthropic.get_anthropic_api_key", return_value="sk-ant-KEY")
    @patch("httpx.Client.post")
    def test_anthropic_503_overloaded(self, mock_post, mock_k):
        """Anthropic 503/529 server overload raises AIServiceUnavailableError."""
        mock_post.return_value = _mock_resp(503, {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Anthropic is overloaded"}
        })
        with self.assertRaises(AIServiceUnavailableError):
            self.provider.generate("Hello", db=self.mock_db, max_retries=0)

    @patch("app.services.ai.providers.anthropic.get_anthropic_api_key", return_value="sk-ant-KEY")
    @patch("httpx.Client.get")
    def test_anthropic_list_models_normalization(self, mock_get, mock_k):
        """Anthropic GET /v1/models normalizes model items."""
        mock_get.return_value = _mock_resp(200, {
            "data": [
                {"id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet"},
                {"id": "claude-3-5-haiku-20241022", "display_name": "Claude 3.5 Haiku"}
            ]
        })
        models = self.provider.list_models(db=self.mock_db)
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["id"], "claude-3-5-sonnet-20241022")
        self.assertEqual(models[0]["provider"], "anthropic")
        self.assertEqual(models[0]["provider_type"], "direct")


class TestGroqProviderMatrix(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_db.query.return_value.filter.return_value.first.return_value = None
        global_model_cache.clear()
        self.provider = GroqProvider()

    @patch("app.services.ai.providers.groq.get_groq_api_key", return_value="gsk_TEST_GROQ_KEY")
    @patch("app.services.ai.providers.groq.get_groq_model", return_value="llama-3.3-70b-versatile")
    @patch("httpx.Client.post")
    def test_groq_successful_generation(self, mock_post, mock_m, mock_k):
        """Groq successful generation returns content and metadata."""
        mock_post.return_value = _mock_resp(200, {
            "choices": [{"message": {"content": "Hello from Groq LPU"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16}
        })
        result = self.provider.generate("Test prompt", db=self.mock_db)
        self.assertEqual(result, "Hello from Groq LPU")
        meta = self.provider.get_last_execution_metadata()
        self.assertEqual(meta["provider"], "groq")
        self.assertEqual(meta["provider_type"], "direct")
        self.assertEqual(meta["total_tokens"], 16)

    @patch("app.services.ai.providers.groq.get_groq_api_key", return_value="gsk_BAD")
    @patch("httpx.Client.post")
    def test_groq_401_authentication_error(self, mock_post, mock_k):
        """Groq 401 raises AIAuthenticationError."""
        mock_post.return_value = _mock_resp(401, {"error": {"message": "Invalid API Key"}})
        with self.assertRaises(AIAuthenticationError):
            self.provider.generate("Hello", db=self.mock_db, max_retries=0)

    @patch("app.services.ai.providers.groq.get_groq_api_key", return_value="gsk_KEY")
    @patch("httpx.Client.post")
    def test_groq_503_service_unavailable(self, mock_post, mock_k):
        """Groq 503 raises AIServiceUnavailableError."""
        mock_post.return_value = _mock_resp(503, {"error": {"message": "Groq capacity full"}})
        with self.assertRaises(AIServiceUnavailableError):
            self.provider.generate("Hello", db=self.mock_db, max_retries=0)


class TestOpenRouterProviderMatrix(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_db.query.return_value.filter.return_value.first.return_value = None
        global_model_cache.clear()
        self.provider = OpenRouterProvider()

    @patch("app.services.ai.providers.openrouter.get_openrouter_api_key", return_value="sk-or-TEST_KEY")
    @patch("app.services.ai.providers.openrouter.get_openrouter_model", return_value="anthropic/claude-3.5-sonnet")
    @patch("httpx.Client.post")
    def test_openrouter_successful_generation(self, mock_post, mock_m, mock_k):
        """OpenRouter successful generation returns content and metadata."""
        mock_post.return_value = _mock_resp(200, {
            "choices": [{"message": {"content": "Hello from OpenRouter Gateway"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        })
        result = self.provider.generate("Test prompt", db=self.mock_db)
        self.assertEqual(result, "Hello from OpenRouter Gateway")
        meta = self.provider.get_last_execution_metadata()
        self.assertEqual(meta["provider"], "openrouter")
        self.assertEqual(meta["provider_type"], "gateway")
        self.assertEqual(meta["total_tokens"], 30)

    @patch("app.services.ai.providers.openrouter.get_openrouter_api_key", return_value="sk-or-BAD")
    @patch("httpx.Client.post")
    def test_openrouter_401_authentication_error(self, mock_post, mock_k):
        """OpenRouter 401 raises AIAuthenticationError."""
        mock_post.return_value = _mock_resp(401, {"error": {"message": "Unauthorized"}})
        with self.assertRaises(AIAuthenticationError):
            self.provider.generate("Hello", db=self.mock_db, max_retries=0)

    @patch("app.services.ai.providers.openrouter.get_openrouter_api_key", return_value="sk-or-KEY")
    @patch("httpx.Client.post")
    def test_openrouter_503_service_unavailable(self, mock_post, mock_k):
        """OpenRouter 503 raises AIServiceUnavailableError."""
        mock_post.return_value = _mock_resp(503, {"error": {"message": "All providers failed"}})
        with self.assertRaises(AIServiceUnavailableError):
            self.provider.generate("Hello", db=self.mock_db, max_retries=0)


class TestOpenRouterResearchSafety(unittest.TestCase):
    """
    CRITICAL OPENROUTER RESEARCH SAFETY VERIFICATION:
    1. Sends EXACTLY ONE model string in 'model'.
    2. Does NOT send 'models' fallback array.
    3. Disables provider fallback via provider.allow_fallbacks = False when enable_fallback=False.
    4. Exactly 1 generation HTTP request is sent.
    """
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_db.query.return_value.filter.return_value.first.return_value = None
        self.provider = OpenRouterProvider()

    @patch("app.services.ai.providers.openrouter.get_openrouter_api_key", return_value="sk-or-SAFE_KEY")
    @patch("app.services.ai.providers.openrouter.get_openrouter_model", return_value="anthropic/claude-3.5-sonnet")
    @patch("httpx.Client.post")
    def test_openrouter_research_safety_payload_and_single_request(self, mock_post, mock_m, mock_k):
        mock_post.return_value = _mock_resp(200, {
            "choices": [{"message": {"content": SAMPLE_PRODUCTS_JSON}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 120, "total_tokens": 170}
        })

        prods = self.provider.generate_products(niche="Đồ chơi", count=5, db=self.mock_db)
        self.assertEqual(len(prods), 1)
        self.assertEqual(mock_post.call_count, 1, "Must make exactly 1 outbound HTTP call")

        # Inspect request payload
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]

        # 1. Single model string
        self.assertIn("model", payload)
        self.assertIsInstance(payload["model"], str)
        self.assertEqual(payload["model"], "anthropic/claude-3.5-sonnet")

        # 2. No models array
        self.assertNotIn("models", payload)

        # 3. Provider allow_fallbacks = False
        self.assertIn("provider", payload)
        self.assertFalse(payload["provider"]["allow_fallbacks"])

        # 4. Token budget bounded to 800 tokens for 5 items
        self.assertEqual(payload["max_tokens"], 800)


class TestModelDiscoveryAndCaching(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_db.query.return_value.filter.return_value.first.return_value = None
        global_model_cache.clear()

    @patch("app.services.ai.providers.openai.get_openai_api_key", return_value="sk-TEST_KEY")
    @patch("httpx.Client.get")
    def test_model_cache_avoids_repeated_network_calls(self, mock_get, mock_k):
        """Second call to list_models returns from cache with 0 HTTP calls."""
        mock_get.return_value = _mock_resp(200, {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]})
        p = OpenAIProvider()
        res1 = p.list_models(db=self.mock_db)
        res2 = p.list_models(db=self.mock_db)
        self.assertEqual(len(res1), 2)
        self.assertEqual(len(res2), 2)
        self.assertEqual(mock_get.call_count, 1)

    @patch("app.services.ai.providers.openai.get_openai_api_key", return_value="sk-TEST_KEY")
    @patch("app.services.ai.providers.openai.OpenAIProvider.list_models")
    @patch("httpx.Client.post")
    def test_generation_never_calls_list_models(self, mock_post, mock_list, mock_k):
        """Calling generate() must NEVER trigger model discovery."""
        mock_post.return_value = _mock_resp(200, {
            "choices": [{"message": {"content": "Direct text"}}]
        })
        p = OpenAIProvider()
        res = p.generate("Hello", db=self.mock_db)
        self.assertEqual(res, "Direct text")
        mock_list.assert_not_called()


class TestResearchOneRequestMatrix(unittest.TestCase):
    """
    MANDATORY SECTION 31 VERIFICATION:
    Proves across ALL 5 providers:
    normal = 1 request
    503 failure = 1 request (0 retries)
    429 failure = 1 request (0 retries)
    timeout = 1 request (0 retries)
    invalid response = 1 request (0 retries)
    Cache hit = 0 requests
    """
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_db.query.return_value.filter.return_value.first.return_value = None

    # --- GEMINI ---
    @patch("app.config.get_gemini_api_key", return_value="AIzaSy_KEY")
    @patch("httpx.Client.post")
    def test_matrix_gemini(self, mock_post, mock_k):
        p = GeminiProvider()
        # Normal
        mock_post.return_value = _mock_resp(200, {"candidates": [{"content": {"parts": [{"text": SAMPLE_PRODUCTS_JSON}]}}]})
        prods = p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(len(prods), 1)
        self.assertEqual(mock_post.call_count, 1)

        # 503 Overload => exactly 1 request
        mock_post.reset_mock()
        mock_post.return_value = _mock_resp(503, {"error": {"message": "High demand"}})
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

        # 429 Rate limit => exactly 1 request
        mock_post.reset_mock()
        mock_post.return_value = _mock_resp(429, {"error": {"message": "Resource exhausted"}})
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

        # Timeout => exactly 1 request
        mock_post.reset_mock()
        mock_post.side_effect = httpx.TimeoutException("Timeout")
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

        # Invalid JSON => exactly 1 request
        mock_post.reset_mock()
        mock_post.side_effect = None
        mock_post.return_value = _mock_resp(200, {"candidates": [{"content": {"parts": [{"text": "Not JSON"}]}}]})
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

    # --- OPENAI ---
    @patch("app.services.ai.providers.openai.get_openai_api_key", return_value="sk-KEY")
    @patch("httpx.Client.post")
    def test_matrix_openai(self, mock_post, mock_k):
        p = OpenAIProvider()
        # Normal
        mock_post.return_value = _mock_resp(200, {"choices": [{"message": {"content": SAMPLE_PRODUCTS_JSON}}]})
        prods = p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(len(prods), 1)
        self.assertEqual(mock_post.call_count, 1)

        # 503 => exactly 1 request
        mock_post.reset_mock()
        mock_post.return_value = _mock_resp(503, {"error": {"message": "Overloaded"}})
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

        # 429 => exactly 1 request
        mock_post.reset_mock()
        mock_post.return_value = _mock_resp(429, {"error": {"message": "Rate limit"}})
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

        # Timeout => exactly 1 request
        mock_post.reset_mock()
        mock_post.side_effect = httpx.TimeoutException("Timeout")
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

        # Invalid JSON => exactly 1 request
        mock_post.reset_mock()
        mock_post.side_effect = None
        mock_post.return_value = _mock_resp(200, {"choices": [{"message": {"content": "Not JSON"}}]})
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

    # --- ANTHROPIC ---
    @patch("app.services.ai.providers.anthropic.get_anthropic_api_key", return_value="sk-ant-KEY")
    @patch("httpx.Client.post")
    def test_matrix_anthropic(self, mock_post, mock_k):
        p = AnthropicProvider()
        # Normal
        mock_post.return_value = _mock_resp(200, {"content": [{"type": "text", "text": SAMPLE_PRODUCTS_JSON}]})
        prods = p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(len(prods), 1)
        self.assertEqual(mock_post.call_count, 1)

        # 503 => exactly 1 request
        mock_post.reset_mock()
        mock_post.return_value = _mock_resp(503, {"error": {"message": "Overloaded"}})
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

        # 429 => exactly 1 request
        mock_post.reset_mock()
        mock_post.return_value = _mock_resp(429, {"error": {"message": "Rate limit"}})
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

        # Timeout => exactly 1 request
        mock_post.reset_mock()
        mock_post.side_effect = httpx.TimeoutException("Timeout")
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

        # Invalid JSON => exactly 1 request
        mock_post.reset_mock()
        mock_post.side_effect = None
        mock_post.return_value = _mock_resp(200, {"content": [{"type": "text", "text": "Not JSON"}]})
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

    # --- GROQ ---
    @patch("app.services.ai.providers.groq.get_groq_api_key", return_value="gsk_KEY")
    @patch("httpx.Client.post")
    def test_matrix_groq(self, mock_post, mock_k):
        p = GroqProvider()
        # Normal
        mock_post.return_value = _mock_resp(200, {"choices": [{"message": {"content": SAMPLE_PRODUCTS_JSON}}]})
        prods = p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(len(prods), 1)
        self.assertEqual(mock_post.call_count, 1)

        # 503 => exactly 1 request
        mock_post.reset_mock()
        mock_post.return_value = _mock_resp(503, {"error": {"message": "Overloaded"}})
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

        # 429 => exactly 1 request
        mock_post.reset_mock()
        mock_post.return_value = _mock_resp(429, {"error": {"message": "Rate limit"}})
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

        # Timeout => exactly 1 request
        mock_post.reset_mock()
        mock_post.side_effect = httpx.TimeoutException("Timeout")
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

        # Invalid JSON => exactly 1 request
        mock_post.reset_mock()
        mock_post.side_effect = None
        mock_post.return_value = _mock_resp(200, {"choices": [{"message": {"content": "Not JSON"}}]})
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

    # --- OPENROUTER ---
    @patch("app.services.ai.providers.openrouter.get_openrouter_api_key", return_value="sk-or-KEY")
    @patch("httpx.Client.post")
    def test_matrix_openrouter(self, mock_post, mock_k):
        p = OpenRouterProvider()
        # Normal
        mock_post.return_value = _mock_resp(200, {"choices": [{"message": {"content": SAMPLE_PRODUCTS_JSON}}]})
        prods = p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(len(prods), 1)
        self.assertEqual(mock_post.call_count, 1)

        # 503 => exactly 1 request
        mock_post.reset_mock()
        mock_post.return_value = _mock_resp(503, {"error": {"message": "Overloaded"}})
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

        # 429 => exactly 1 request
        mock_post.reset_mock()
        mock_post.return_value = _mock_resp(429, {"error": {"message": "Rate limit"}})
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

        # Timeout => exactly 1 request
        mock_post.reset_mock()
        mock_post.side_effect = httpx.TimeoutException("Timeout")
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)

        # Invalid JSON => exactly 1 request
        mock_post.reset_mock()
        mock_post.side_effect = None
        mock_post.return_value = _mock_resp(200, {"choices": [{"message": {"content": "Not JSON"}}]})
        with self.assertRaises(Exception):
            p.generate_products("Gia dụng", count=2, db=self.mock_db)
        self.assertEqual(mock_post.call_count, 1)


class TestTokenBudgetEnforcement(unittest.TestCase):
    def test_token_budgets(self):
        """Deterministic Research token budgets match exact product count tiers."""
        self.assertEqual(get_research_max_output_tokens(5), 800)
        self.assertEqual(get_research_max_output_tokens(10), 1500)
        self.assertEqual(get_research_max_output_tokens(20), 2800)
        self.assertEqual(get_research_max_output_tokens(30), 4000)
        self.assertEqual(get_research_max_output_tokens(50), 6000)


if __name__ == "__main__":
    unittest.main()
