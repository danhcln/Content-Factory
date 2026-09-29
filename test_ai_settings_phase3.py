"""
Multi-AI Phase 3 Settings/UI Comprehensive Test Suite
Tests:
1. Active Provider Resolution: SQLite Setting > .env fallback > Gemini default.
2. Invalid active provider string cleanly falls back to Gemini.
3. Secret UI Handling: Raw API keys NEVER in template context, HTML, or JSON.
4. Key Hint Generation: Non-sensitive last-4 hint only.
5. Blank submission preserves existing stored keys in SQLite.
6. New key submission explicitly replaces existing key in SQLite.
7. Direct BYOK vs Cloud AI Gateway classification.
8. Zero-token Test Connection for all 5 providers (Gemini, OpenAI, Anthropic, Groq, OpenRouter).
9. Secret sanitization on test connection error responses.
10. Dynamic model discovery endpoint with ModelCatalogCache verification.
11. Provider status endpoint verification (zero secret leakage).
12. Research invariant preservation across all 5 providers.

ALL TESTS RUN OFFLINE WITH MOCKED HTTP / SERVICES — ZERO PAID QUOTA CONSUMED.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import json
import httpx
from starlette.testclient import TestClient

from app.services.ai.base import (
    AIProvider,
    AIGenerationOptions,
    ExecutionMetadata,
    AIProviderError,
    AIAuthenticationError,
    get_research_max_output_tokens,
)
from app.services.ai.manager import AIProviderManager, get_ai_manager, set_ai_manager
from app.services.ai.providers.common import global_model_cache, sanitize_secrets
from app.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_OPENROUTER_MODEL,
)


class TestMultiAISettingsPhase3(unittest.TestCase):
    """Phase 3 Multi-AI Settings, UI & Invariant Verification."""

    def setUp(self):
        # Reset provider manager singleton
        set_ai_manager(AIProviderManager())
        global_model_cache.clear()

    def tearDown(self):
        global_model_cache.clear()

    # ==========================================================================
    # 1. ACTIVE PROVIDER RESOLUTION & DEFAULTS
    # ==========================================================================

    def test_active_provider_default_resolves_to_gemini(self):
        """When no DB setting or env var exists, default provider is strictly Gemini."""
        from app.config import get_active_ai_provider
        with patch.dict(os.environ, {}, clear=True):
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            active_id = get_active_ai_provider(db=mock_db)
            self.assertEqual(active_id, "gemini")

            manager = get_ai_manager()
            provider = manager.get_active_provider(db=mock_db)
            self.assertEqual(provider.provider_id, "gemini")

    def test_active_provider_db_overrides_env(self):
        """SQLite Setting table takes precedence over .env for active provider."""
        from app.config import get_active_ai_provider
        with patch.dict(os.environ, {"ACTIVE_AI_PROVIDER": "anthropic"}):
            mock_db = MagicMock()
            mock_setting = MagicMock()
            mock_setting.value = "openai"
            mock_db.query.return_value.filter.return_value.first.return_value = mock_setting

            active_id = get_active_ai_provider(db=mock_db)
            self.assertEqual(active_id, "openai")

            manager = get_ai_manager()
            provider = manager.get_active_provider(db=mock_db)
            self.assertEqual(provider.provider_id, "openai")

    def test_active_provider_env_fallback_when_db_empty(self):
        """When SQLite Setting has no entry, ACTIVE_AI_PROVIDER in .env is used."""
        from app.config import get_active_ai_provider
        with patch.dict(os.environ, {"ACTIVE_AI_PROVIDER": "groq"}):
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None

            active_id = get_active_ai_provider(db=mock_db)
            self.assertEqual(active_id, "groq")

            manager = get_ai_manager()
            provider = manager.get_active_provider(db=mock_db)
            self.assertEqual(provider.provider_id, "groq")

    def test_active_provider_invalid_string_falls_back_to_gemini(self):
        """An invalid or prohibited provider (e.g., 'ollama') safely falls back to Gemini."""
        from app.config import get_active_ai_provider
        with patch.dict(os.environ, {"ACTIVE_AI_PROVIDER": "ollama"}):
            mock_db = MagicMock()
            mock_setting = MagicMock()
            mock_setting.value = "some_unknown_local_model"
            mock_db.query.return_value.filter.return_value.first.return_value = mock_setting

            active_id = get_active_ai_provider(db=mock_db)
            self.assertEqual(active_id, "gemini")

            manager = get_ai_manager()
            provider = manager.get_active_provider(db=mock_db)
            self.assertEqual(provider.provider_id, "gemini")

    # ==========================================================================
    # 2. SECRET UI HANDLING & KEY MASKING
    # ==========================================================================

    def test_key_hint_generation(self):
        """Key hint exposes only non-sensitive last-4 chars for long keys, or generic hint."""
        from app.config import mask_api_key, get_key_hint
        self.assertEqual(get_key_hint(""), "")
        self.assertEqual(get_key_hint(None), "")
        self.assertEqual(get_key_hint("short"), "••••••••")
        self.assertEqual(get_key_hint("sk-proj-1234567890abcdef"), "••••cdef")
        self.assertEqual(get_key_hint("AIzaSyB1234567890XYZ"), "••••0XYZ")

    def _create_test_db(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from app.database import Base
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        return Session()

    def test_settings_context_zero_secret_leak(self):
        """Settings context provided to templates must NEVER contain raw API keys."""
        from app.routes.settings import get_current_settings
        from app.models import Setting

        db = self._create_test_db()
        db.add(Setting(key="openai_api_key", value="sk-proj-SecretOpenAIKey5678"))
        db.add(Setting(key="anthropic_api_key", value="sk-ant-SecretAnthropicKey9999"))
        db.add(Setting(key="groq_api_key", value="gsk_SecretGroqKey1111"))
        db.add(Setting(key="openrouter_api_key", value="sk-or-SecretOpenRouterKey2222"))
        db.add(Setting(key="gemini_api_key", value="AIzaSySecretGeminiKey1234"))
        db.commit()

        settings = get_current_settings(db=db)

        # Dump whole settings dictionary to JSON to inspect all nested values
        settings_str = json.dumps(settings, default=str)

        # Assert ZERO raw secret strings are present anywhere in the context
        self.assertNotIn("AIzaSySecretGeminiKey1234", settings_str)
        self.assertNotIn("sk-proj-SecretOpenAIKey5678", settings_str)
        self.assertNotIn("sk-ant-SecretAnthropicKey9999", settings_str)
        self.assertNotIn("gsk_SecretGroqKey1111", settings_str)
        self.assertNotIn("sk-or-SecretOpenRouterKey2222", settings_str)

        # Check that ai_providers section exists with configured=True and key_hint
        self.assertIn("ai_providers", settings)
        providers = settings["ai_providers"]["providers"]
        self.assertTrue(providers["openai"]["configured"])
        self.assertEqual(providers["openai"]["key_hint"], "••••5678")
        self.assertTrue(providers["anthropic"]["configured"])
        self.assertEqual(providers["anthropic"]["key_hint"], "••••9999")
        db.close()

    def test_blank_key_submission_preserves_existing_key(self):
        """Submitting a blank or empty key preserves the existing key in the database."""
        from app.routes.settings import save_provider_settings
        from app.models import Setting

        db = self._create_test_db()
        db.add(Setting(key="openai_api_key", value="sk-proj-existingSecretKey1234", description="OpenAI API Key"))
        db.add(Setting(key="openai_model", value="gpt-4o-mini", description="OpenAI AI Model"))
        db.commit()

        # Blank submission
        save_provider_settings(
            db=db,
            provider_id="openai",
            submitted_key="",
            submitted_model="gpt-4o-mini"
        )
        db.commit()

        # Value must remain the original secret key, NOT overwritten by ""
        rec = db.query(Setting).filter(Setting.key == "openai_api_key").first()
        self.assertEqual(rec.value, "sk-proj-existingSecretKey1234")
        db.close()

    def test_new_key_submission_explicitly_replaces_existing(self):
        """Entering a new non-empty key explicitly replaces the database value."""
        from app.routes.settings import save_provider_settings
        from app.models import Setting

        db = self._create_test_db()
        db.add(Setting(key="openai_api_key", value="sk-proj-oldKey1234", description="OpenAI API Key"))
        db.add(Setting(key="openai_model", value="gpt-4o-mini", description="OpenAI AI Model"))
        db.commit()

        save_provider_settings(
            db=db,
            provider_id="openai",
            submitted_key="sk-proj-brandNewKey9999",
            submitted_model="gpt-4o"
        )
        db.commit()

        key_rec = db.query(Setting).filter(Setting.key == "openai_api_key").first()
        model_rec = db.query(Setting).filter(Setting.key == "openai_model").first()
        self.assertEqual(key_rec.value, "sk-proj-brandNewKey9999")
        self.assertEqual(model_rec.value, "gpt-4o")
        db.close()

    # ==========================================================================
    # 3. DIRECT BYOK VS GATEWAY CLASSIFICATION
    # ==========================================================================

    def test_provider_role_classification(self):
        """Gemini, OpenAI, Anthropic, Groq are 'direct'; OpenRouter is 'gateway'."""
        manager = get_ai_manager()
        self.assertEqual(manager.get_provider("gemini").provider_type, "direct")
        self.assertEqual(manager.get_provider("openai").provider_type, "direct")
        self.assertEqual(manager.get_provider("anthropic").provider_type, "direct")
        self.assertEqual(manager.get_provider("groq").provider_type, "direct")
        self.assertEqual(manager.get_provider("openrouter").provider_type, "gateway")

    # ==========================================================================
    # 4. ZERO-TOKEN TEST CONNECTION & SANITIZATION
    # ==========================================================================

    def test_connection_all_providers_zero_generation_tokens(self):
        """test_connection across all 5 providers must invoke metadata checks and consume 0 generation tokens."""
        manager = get_ai_manager()
        mock_db = MagicMock()

        # Mock each provider's underlying test_connection to verify manager delegation
        for pid in ["gemini", "openai", "anthropic", "groq", "openrouter"]:
            provider = manager.get_provider(pid)
            with patch.object(provider, "test_connection") as mock_test, \
                 patch.object(provider, "generate") as mock_gen:
                mock_test.return_value = {
                    "provider": pid,
                    "provider_type": provider.provider_type,
                    "configured": True,
                    "connected": True,
                    "configured_model": "test-model",
                    "message": "Connected successfully",
                    "error_type": None
                }

                res = manager.test_connection(provider_id=pid, db=mock_db)

                mock_test.assert_called_once_with(db=mock_db)
                # CRITICAL: generate() must NEVER be called during connection testing
                mock_gen.assert_not_called()
                self.assertTrue(res["connected"])
                self.assertEqual(res["provider"], pid)

    def test_connection_sanitizes_secret_error(self):
        """Upstream error in test connection must be sanitized to scrub credentials."""
        from app.routes.settings import test_provider_connection_logic

        mock_db = MagicMock()
        manager = get_ai_manager()
        openai_provider = manager.get_provider("openai")

        secret_key = "sk-proj-SUPER_SECRET_TOKEN_XYZ12345"

        with patch("app.services.ai.providers.openai.get_openai_api_key", return_value=secret_key), \
             patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            mock_resp.text = f'{{"error": {{"message": "Invalid API key provided: {secret_key}"}}}}'
            mock_get.return_value = mock_resp

            result = test_provider_connection_logic(provider_id="openai", db=mock_db)

            self.assertFalse(result["connected"])
            self.assertNotIn(secret_key, result["message"])
            self.assertIn("[REDACTED]", result["message"])

    # ==========================================================================
    # 5. DYNAMIC MODEL DISCOVERY & CACHE EFFICIENCY
    # ==========================================================================

    def test_dynamic_model_discovery_with_cache(self):
        """Model discovery returns models and caches result with 0 redundant HTTP calls."""
        from app.routes.settings import get_provider_models_logic

        mock_db = MagicMock()
        sample_models = [
            {"id": "gpt-4o", "name": "GPT-4o", "provider": "openai", "provider_type": "direct", "available": True},
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "openai", "provider_type": "direct", "available": True},
        ]

        with patch("app.services.ai.providers.openai.get_openai_api_key", return_value="test_key_12345"), \
             patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "data": [
                    {"id": "gpt-4o"},
                    {"id": "gpt-4o-mini"}
                ]
            }
            mock_get.return_value = mock_resp

            # First call: hits provider list_models and HTTP client
            res1 = get_provider_models_logic(provider_id="openai", db=mock_db)
            self.assertEqual(len(res1["models"]), 2)
            self.assertEqual(mock_get.call_count, 1)

            # Subsequent call within TTL returns cached catalog with 0 HTTP calls
            res2 = get_provider_models_logic(provider_id="openai", db=mock_db)
            self.assertEqual(len(res2["models"]), 2)
            # Ensure no second HTTP call was made
            self.assertEqual(mock_get.call_count, 1)

    # ==========================================================================
    # 6. RESEARCH INVARIANT DEFENSE ACROSS ALL PROVIDERS
    # ==========================================================================

    def test_research_invariants_preserved_with_active_provider(self):
        """Research batch generation preserves strict 1-call limit regardless of active provider."""
        manager = get_ai_manager()
        mock_db = MagicMock()

        for pid in ["gemini", "openai", "anthropic", "groq", "openrouter"]:
            provider = manager.get_provider(pid)

            valid_json = json.dumps([
                {
                    "name_vietnamese": f"Sản phẩm {pid}",
                    "name_chinese": "产品",
                    "douyin_keywords": "好物 推荐",
                    "content_angle": "Góc quay độc đáo",
                    "hook": "Câu mở đầu hấp dẫn"
                }
            ])

            if pid == "gemini":
                with patch.object(provider._gemini_service, "call_gemini", return_value=valid_json) as mock_call, \
                     patch.object(manager, "get_active_provider", return_value=provider):
                    res = manager.generate_products(niche="Nhà bếp", count=5, db=mock_db)
                    self.assertEqual(len(res), 1)
                    self.assertEqual(res[0]["name_vietnamese"], "Sản phẩm gemini")
                    self.assertEqual(mock_call.call_count, 1)
                    _, kwargs = mock_call.call_args
                    self.assertEqual(kwargs.get("max_retries"), 0)
                    self.assertFalse(kwargs.get("enable_fallback"))
                    self.assertEqual(kwargs.get("max_output_tokens"), 800)
            else:
                with patch.object(provider, "generate", return_value=valid_json) as mock_gen, \
                     patch.object(manager, "get_active_provider", return_value=provider):

                    res = manager.generate_products(niche="Nhà bếp", count=5, db=mock_db)

                    self.assertEqual(len(res), 1)
                    self.assertEqual(res[0]["name_vietnamese"], f"Sản phẩm {pid}")

                    # Must be called exactly ONCE with max_retries=0 and enable_fallback=False
                    self.assertEqual(mock_gen.call_count, 1)
                    _, kwargs = mock_gen.call_args
                    self.assertEqual(kwargs.get("max_retries"), 0)
                    self.assertFalse(kwargs.get("enable_fallback"))
                    options = kwargs.get("options")
                    self.assertIsNotNone(options)
                    self.assertEqual(options.max_retries, 0)
                    self.assertFalse(options.enable_fallback)
                    self.assertEqual(options.max_output_tokens, 800)


    # ==========================================================================
    # 7. FASTAPI SETTINGS ROUTES & API ENDPOINTS VERIFICATION
    # ==========================================================================

    def test_settings_page_html_render(self):
        """GET /settings renders successfully with Phase 3 Multi-AI sections and zero secrets."""
        from app.main import app
        client = TestClient(app)

        resp = client.get("/settings")
        self.assertEqual(resp.status_code, 200)
        html = resp.text
        self.assertIn("Active AI Provider", html)
        self.assertIn("Direct BYOK", html)
        self.assertIn("OpenRouter Gateway", html)
        self.assertIn("rad_gemini", html)
        self.assertIn("rad_openrouter", html)
        self.assertIn("btn-test-provider", html)
        self.assertIn("btn-discover-models", html)

    def test_api_test_connection_endpoint(self):
        """POST /api/settings/ai/test-connection tests provider without consuming generation quota."""
        from app.main import app
        client = TestClient(app)

        with patch.object(get_ai_manager().get_provider("openai"), "test_connection") as mock_test:
            mock_test.return_value = {
                "provider": "openai",
                "provider_type": "direct",
                "configured": True,
                "connected": True,
                "configured_model": "gpt-4o-mini",
                "message": "Connected to OpenAI",
                "error_type": None
            }

            resp = client.post("/api/settings/ai/test-connection", json={"provider_id": "openai"})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["connected"])
            self.assertEqual(data["provider"], "openai")
            self.assertEqual(data["provider_type"], "direct")

    def test_api_models_endpoint(self):
        """GET /api/settings/ai/models returns discovered models."""
        from app.main import app
        client = TestClient(app)

        with patch.object(get_ai_manager().get_provider("groq"), "list_models") as mock_list:
            mock_list.return_value = [
                {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "available": True}
            ]

            resp = client.get("/api/settings/ai/models?provider_id=groq")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["provider"], "groq")
            self.assertEqual(len(data["models"]), 1)

    def test_api_status_endpoint_zero_secret_leak(self):
        """GET /api/settings/ai/status returns status of all providers without raw credentials."""
        from app.main import app
        client = TestClient(app)

        resp = client.get("/api/settings/ai/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("active_provider", data)
        self.assertIn("providers", data)
        self.assertIn("gemini", data["providers"])
        self.assertIn("openai", data["providers"])
        self.assertIn("anthropic", data["providers"])
        self.assertIn("groq", data["providers"])
        self.assertIn("openrouter", data["providers"])

        # String search across raw response
        text = resp.text
        self.assertNotIn("AIzaSy", text)
        self.assertNotIn("sk-proj-", text)
        self.assertNotIn("sk-ant-", text)
        self.assertNotIn("gsk_", text)
        self.assertNotIn("sk-or-", text)


if __name__ == "__main__":
    unittest.main()
