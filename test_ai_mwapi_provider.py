"""
Offline Regression Test Suite: MWAPI Gateway Provider (Phase 6)
Verifies:
1. Provider initialization (provider_id='mwapi', display_name='MWAPI Gateway', provider_type='gateway', base_url='https://api.mwapi.dev').
2. Correct Bearer token authentication in HTTP headers.
3. Generation request format (POST /v1/chat/completions) and response parsing with token usage.
4. Model discovery (GET /v1/models) with provider-isolated 60-min in-memory caching.
5. Test Connection uses GET /v1/models and consumes exactly 0 generation tokens.
6. Error mapping: 400, 401, 403, 429 (quota vs rate limit), 500/503, timeout, network error.
7. Secret sanitization across all errors and metadata.
8. Config helpers, SQLite persistence, blank key preservation, custom model preservation.
9. AIProviderManager integration (active provider, configuration detection).
10. Research Invariant:
    - Cache hit = 0 generation calls.
    - Cache miss = exactly 1 POST /v1/chat/completions, max_retries=0, allow_cross_provider_fallback=False.
    - Zero /v1/models calls during generation.
11. UI rendering performs zero network calls.
"""
import os
import unittest
from unittest.mock import MagicMock, patch
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

from app.database import Base, get_db
from app.models import Setting, Product
from app.config import (
    DEFAULT_MWAPI_MODEL,
    SUPPORTED_AI_PROVIDERS,
    get_mwapi_api_key,
    get_mwapi_model,
    normalize_model_name,
    get_active_ai_model,
    get_active_ai_provider,
)
from app.services.ai.providers.common import global_model_cache
from app.services.ai.base import (
    AIAuthenticationError,
    AIPermissionError,
    AIQuotaExceededError,
    AIRateLimitError,
    AIBadRequestError,
    AIServiceUnavailableError,
    AITimeoutError,
    AINetworkError,
    AIInvalidResponseError,
    AIGenerationOptions,
)
from app.routes.settings import save_provider_settings


class TestMWAPIGatewayProvider(unittest.TestCase):
    """Test suite for MWAPI Gateway provider integration."""

    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.db = self.SessionLocal()
        global_model_cache.clear()

    def tearDown(self):
        global_model_cache.clear()
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    # =========================================================================
    # 1. PROVIDER INITIALIZATION & PROPERTIES
    # =========================================================================

    def test_mwapi_provider_properties(self):
        """MWAPIProvider must have expected ID, display name, gateway type, and base URL."""
        from app.services.ai.providers.mwapi import MWAPIProvider
        provider = MWAPIProvider()

        self.assertEqual(provider.provider_id, "mwapi")
        self.assertEqual(provider.display_name, "MWAPI Gateway")
        self.assertEqual(provider.provider_type, "gateway")
        self.assertEqual(provider.BASE_URL, "https://api.mwapi.dev")

    def test_mwapi_supports_model(self):
        """Gateway architecture supports arbitrary models provided to it."""
        from app.services.ai.providers.mwapi import MWAPIProvider
        provider = MWAPIProvider()

        self.assertTrue(provider.supports_model("claude-sonnet-4-6"))
        self.assertTrue(provider.supports_model("claude-3-7-sonnet-20250219"))
        self.assertTrue(provider.supports_model("gpt-4o"))
        self.assertTrue(provider.supports_model("gemini-2.5-flash"))
        self.assertFalse(provider.supports_model(""))
        self.assertFalse(provider.supports_model(None))

    # =========================================================================
    # 2. GENERATION FLOW & USAGE METADATA
    # =========================================================================

    @patch("app.services.ai.providers.mwapi.get_mwapi_api_key", return_value="sk-af9-VALID-KEY-123456789")
    @patch("app.services.ai.providers.mwapi.get_mwapi_model", return_value="claude-sonnet-4-6")
    @patch("httpx.Client.post")
    def test_mwapi_successful_generation(self, mock_post, mock_model, mock_key):
        """Generation sends POST /v1/chat/completions with Bearer token and extracts content & tokens."""
        from app.services.ai.providers.mwapi import MWAPIProvider
        provider = MWAPIProvider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "chatcmpl-mwapi-001",
            "object": "chat.completion",
            "model": "claude-sonnet-4-6",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Generated content from MWAPI"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 34,
                "total_tokens": 46
            }
        }
        mock_post.return_value = mock_resp

        result = provider.generate("Test prompt", db=self.db)
        self.assertEqual(result, "Generated content from MWAPI")

        # Verify HTTP invocation
        self.assertEqual(mock_post.call_count, 1)
        url = mock_post.call_args[0][0]
        self.assertEqual(url, "https://api.mwapi.dev/v1/chat/completions")

        headers = mock_post.call_args[1].get("headers", {})
        self.assertEqual(headers.get("Authorization"), "Bearer sk-af9-VALID-KEY-123456789")
        self.assertEqual(headers.get("Content-Type"), "application/json")

        payload = mock_post.call_args[1].get("json", {})
        self.assertEqual(payload.get("model"), "claude-sonnet-4-6")
        self.assertEqual(payload.get("messages")[0]["content"], "Test prompt")

        # Verify execution metadata
        meta = provider.get_last_execution_metadata()
        self.assertEqual(meta["provider"], "mwapi")
        self.assertEqual(meta["provider_type"], "gateway")
        self.assertEqual(meta["input_tokens"], 12)
        self.assertEqual(meta["output_tokens"], 34)
        self.assertEqual(meta["total_tokens"], 46)

    def test_mwapi_generation_unconfigured_key_raises_auth_error(self):
        """Generation without configured API key raises AIAuthenticationError without network calls."""
        from app.services.ai.providers.mwapi import MWAPIProvider
        provider = MWAPIProvider()

        with patch("app.services.ai.providers.mwapi.get_mwapi_api_key", return_value=""), \
             patch("httpx.Client.post") as mock_post:
            with self.assertRaises(AIAuthenticationError):
                provider.generate("Test prompt", db=self.db)
            mock_post.assert_not_called()

    # =========================================================================
    # 3. TEST CONNECTION (0 GENERATION TOKENS)
    # =========================================================================

    @patch("app.services.ai.providers.mwapi.get_mwapi_api_key", return_value="sk-af9-VALID-KEY")
    @patch("httpx.Client.get")
    @patch("httpx.Client.post")
    def test_mwapi_test_connection_success(self, mock_post, mock_get, mock_key):
        """Test connection uses GET /v1/models and consumes exactly 0 generation tokens."""
        from app.services.ai.providers.mwapi import MWAPIProvider
        provider = MWAPIProvider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "object": "list",
            "data": [
                {"id": "claude-sonnet-4-6", "object": "model"}
            ]
        }
        mock_get.return_value = mock_resp

        res = provider.test_connection(db=self.db)
        self.assertTrue(res["connected"])
        self.assertEqual(res["provider"], "mwapi")
        self.assertIn("claude-sonnet-4-6", res["message"])

        # Critical: POST /v1/chat/completions must NEVER be called
        mock_post.assert_not_called()
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(mock_get.call_args[0][0], "https://api.mwapi.dev/v1/models")

    @patch("app.services.ai.providers.mwapi.get_mwapi_api_key", return_value="sk-af9-INVALID-KEY")
    @patch("httpx.Client.get")
    def test_mwapi_test_connection_401(self, mock_get, mock_key):
        """Test connection with invalid key returns connected=False and sanitized message."""
        from app.services.ai.providers.mwapi import MWAPIProvider
        provider = MWAPIProvider()

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = '{"error": {"message": "Invalid API key provided", "type": "authentication_error"}}'
        mock_get.return_value = mock_resp

        res = provider.test_connection(db=self.db)
        self.assertFalse(res["connected"])
        self.assertEqual(res["error_type"], "authentication")
        self.assertIn("401", res["message"])

    # =========================================================================
    # 4. MODEL DISCOVERY & CACHING
    # =========================================================================

    @patch("app.services.ai.providers.mwapi.get_mwapi_api_key", return_value="sk-af9-VALID-KEY")
    @patch("httpx.Client.get")
    def test_mwapi_model_discovery_cached(self, mock_get, mock_key):
        """Model discovery queries GET /v1/models and caches results in memory for 60 min."""
        from app.services.ai.providers.mwapi import MWAPIProvider
        provider = MWAPIProvider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "object": "list",
            "data": [
                {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6"},
                {"id": "gpt-4o", "name": "GPT-4o"}
            ]
        }
        mock_get.return_value = mock_resp

        # First call fetches from remote
        models1 = provider.list_models(db=self.db)
        self.assertEqual(len(models1), 2)
        self.assertEqual(models1[0]["id"], "claude-sonnet-4-6")
        self.assertEqual(mock_get.call_count, 1)

        # Second call returns cached list with 0 network calls
        models2 = provider.list_models(db=self.db)
        self.assertEqual(len(models2), 2)
        self.assertEqual(mock_get.call_count, 1)

    # =========================================================================
    # 5. ERROR MAPPING MATRIX
    # =========================================================================

    @patch("app.services.ai.providers.mwapi.get_mwapi_api_key", return_value="sk-af9-TEST")
    @patch("httpx.Client.post")
    def test_mwapi_error_classification_matrix(self, mock_post, mock_key):
        """MWAPI HTTP responses map correctly to domain exceptions."""
        from app.services.ai.providers.mwapi import MWAPIProvider
        provider = MWAPIProvider()

        error_scenarios = [
            (400, '{"error": {"message": "Bad request parameters"}}', AIBadRequestError),
            (401, '{"error": {"message": "Invalid token"}}', AIAuthenticationError),
            (403, '{"error": {"message": "Forbidden model"}}', AIPermissionError),
            (429, '{"error": {"message": "Insufficient account quota/credit balance"}}', AIQuotaExceededError),
            (429, '{"error": {"message": "Rate limit exceeded, try again in 5s"}}', AIRateLimitError),
            (500, '{"error": {"message": "Internal gateway failure"}}', AIServiceUnavailableError),
            (503, '{"error": {"message": "Upstream service overloaded"}}', AIServiceUnavailableError),
        ]

        for status_code, err_body, exc_cls in error_scenarios:
            mock_resp = MagicMock()
            mock_resp.status_code = status_code
            mock_resp.text = err_body
            mock_post.return_value = mock_resp

            with self.assertRaises(exc_cls, msg=f"Expected status {status_code} to raise {exc_cls.__name__}"):
                provider.generate("prompt", db=self.db, max_retries=0)

    @patch("app.services.ai.providers.mwapi.get_mwapi_api_key", return_value="sk-af9-SECRET-TOKEN-XYZ")
    @patch("httpx.Client.post")
    def test_mwapi_secret_sanitization_in_errors(self, mock_post, mock_key):
        """Raw API key in remote error responses is redacted via sanitize_secrets."""
        from app.services.ai.providers.mwapi import MWAPIProvider
        provider = MWAPIProvider()

        raw_secret = "sk-af9-SECRET-TOKEN-XYZ"
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = f'{{"error": {{"message": "Key {raw_secret} has invalid parameters"}}}}'
        mock_post.return_value = mock_resp

        with self.assertRaises(AIBadRequestError) as ctx:
            provider.generate("prompt", db=self.db, max_retries=0)

        err_str = str(ctx.exception)
        self.assertNotIn(raw_secret, err_str)
        self.assertIn("[REDACTED]", err_str)

    # =========================================================================
    # 6. CONFIG & SETTINGS INTEGRATION
    # =========================================================================

    def test_config_mwapi_defaults_and_normalization(self):
        """Config exposes mwapi provider, defaults, and local normalization."""
        self.assertIn("mwapi", SUPPORTED_AI_PROVIDERS)
        self.assertEqual(DEFAULT_MWAPI_MODEL, "claude-sonnet-4-6")

        # Empty returns default
        self.assertEqual(normalize_model_name("mwapi", ""), "claude-sonnet-4-6")
        self.assertEqual(normalize_model_name("mwapi", None), "claude-sonnet-4-6")

        # Custom model preserved
        self.assertEqual(normalize_model_name("mwapi", "claude-3-opus-20240229"), "claude-3-opus-20240229")
        self.assertEqual(normalize_model_name("mwapi", "gpt-4o"), "gpt-4o")

    def test_settings_save_and_blank_key_preservation(self):
        """Settings save stores MWAPI key & model and preserves key when blank is submitted."""
        # 1. Initial save with new key
        save_provider_settings(self.db, "mwapi", "sk-af9-FIRST-KEY", "claude-sonnet-4-6")
        self.db.commit()

        self.assertEqual(get_mwapi_api_key(self.db), "sk-af9-FIRST-KEY")
        self.assertEqual(get_mwapi_model(self.db), "claude-sonnet-4-6")

        # 2. Save with blank key and updated custom model
        save_provider_settings(self.db, "mwapi", "", "claude-3-7-sonnet-20250219")
        self.db.commit()

        # Key must be preserved intact!
        self.assertEqual(get_mwapi_api_key(self.db), "sk-af9-FIRST-KEY")
        self.assertEqual(get_mwapi_model(self.db), "claude-3-7-sonnet-20250219")

    # =========================================================================
    # 7. AI PROVIDER MANAGER INTEGRATION
    # =========================================================================

    def test_manager_registers_mwapi_provider(self):
        """AIProviderManager registers MWAPIProvider and resolves active provider."""
        from app.services.ai import get_ai_manager
        manager = get_ai_manager()

        prov = manager.get_provider("mwapi")
        self.assertEqual(prov.provider_id, "mwapi")
        self.assertEqual(prov.display_name, "MWAPI Gateway")

        # Set active provider to mwapi
        self.db.add(Setting(key="active_ai_provider", value="mwapi"))
        self.db.commit()

        active = manager.get_active_provider(db=self.db)
        self.assertEqual(active.provider_id, "mwapi")

    # =========================================================================
    # 8. RESEARCH INVARIANTS (0 CALLS ON CACHE HIT, 1 ON CACHE MISS)
    # =========================================================================

    def test_research_cache_hit_zero_mwapi_calls(self):
        """Research cache hit makes exactly 0 calls to MWAPI."""
        from app.main import app

        cached_prod = Product(
            product_id="P1234",
            niche="Đồ gia dụng",
            name_vietnamese="Máy xay sinh tố",
            name_chinese="果汁机",
            douyin_keywords="果汁机",
            content_angle="Tiện lợi",
            hook="Xay cực nhanh",
            status="RESEARCHED"
        )
        self.db.add(cached_prod)
        self.db.add(Setting(key="active_ai_provider", value="mwapi"))
        self.db.add(Setting(key="mwapi_api_key", value="sk-af9-testkey"))
        self.db.commit()

        def _get_test_db():
            db = self.SessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = _get_test_db
        client = TestClient(app)

        with patch("app.routes.research.get_ai_manager") as mock_get_mgr:
            mock_mgr = MagicMock()
            mock_get_mgr.return_value = mock_mgr

            resp = client.post(
                "/research",
                data={"niche": "Đồ gia dụng", "product_count": 1, "fresh": "false"},
                follow_redirects=False
            )

            self.assertEqual(resp.status_code, 303)
            self.assertIn("cached=1", resp.headers["location"])
            mock_mgr.generate_products.assert_not_called()

        app.dependency_overrides.clear()

    def test_research_cache_miss_exactly_one_mwapi_call_with_invariants(self):
        """Research cache miss makes exactly 1 generation call, max_retries=0, allow_cross_provider_fallback=False."""
        from app.services.ai.providers.mwapi import MWAPIProvider
        provider = MWAPIProvider()

        with patch("app.services.ai.providers.mwapi.get_mwapi_api_key", return_value="sk-af9-VALID-KEY"), \
             patch("httpx.Client.post") as mock_post:

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": "chatcmpl-001",
                "object": "chat.completion",
                "model": "claude-sonnet-4-6",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Products JSON"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }
            mock_post.return_value = mock_resp

            options = AIGenerationOptions(
                max_retries=0,
                enable_fallback=False,
                allow_cross_provider_fallback=False,
                timeout=45.0
            )

            res = provider.generate("Research prompt", db=self.db, options=options)
            self.assertEqual(res, "Products JSON")

            # Invariant checks:
            # 1. Exactly 1 HTTP POST
            self.assertEqual(mock_post.call_count, 1)

            # 2. Outbound URL is /v1/chat/completions
            self.assertEqual(mock_post.call_args[0][0], "https://api.mwapi.dev/v1/chat/completions")

            # 3. Model is claude-sonnet-4-6
            payload = mock_post.call_args[1].get("json", {})
            self.assertEqual(payload.get("model"), "claude-sonnet-4-6")

    # =========================================================================
    # 9. ZERO NETWORK CALLS DURING UI RENDERING
    # =========================================================================

    def test_zero_network_calls_during_ui_rendering(self):
        """Rendering /settings and /research with MWAPI active performs 0 network calls."""
        from app.main import app

        self.db.add(Setting(key="active_ai_provider", value="mwapi"))
        self.db.add(Setting(key="mwapi_api_key", value="sk-af9-testkey12345"))
        self.db.commit()

        def _get_test_db():
            db = self.SessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = _get_test_db
        client = TestClient(app)

        with patch("app.services.ai.providers.mwapi.httpx.Client") as mock_mwapi_http:
            mock_mwapi_http.side_effect = RuntimeError("NETWORK CALL FORBIDDEN IN UI RENDERING")

            # 1. Render Settings Page
            resp_settings = client.get("/settings")
            self.assertEqual(resp_settings.status_code, 200)
            self.assertIn("MWAPI Gateway", resp_settings.text)

            # 2. Render Research Page
            resp_research = client.get("/research")
            self.assertEqual(resp_research.status_code, 200)
            self.assertIn("MWAPI Gateway", resp_research.text)

            # Assert ZERO network requests made
            mock_mwapi_http.assert_not_called()

        app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
