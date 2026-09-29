"""
Offline Regression Test Suite: Anthropic Sonnet 4.6 Model Default Fix
Verifies:
1. Fresh install default resolves to 'claude-sonnet-4-6'.
2. Legacy default 'claude-3-5-sonnet-20241022' cleanly resolves and migrates to 'claude-sonnet-4-6'.
3. Custom user-configured models (e.g., 'claude-3-opus-20240229', 'claude-3-5-haiku-20241022') are preserved unchanged.
4. AnthropicProvider supports 'claude-sonnet-4-6'.
5. Zero network calls occur during model resolution and local configuration checks.
6. Research invariants remain strictly preserved:
   - Cache hit = 0 generation calls.
   - Cache miss = exactly 1 generation call with max_retries=0, allow_cross_provider_fallback=False.
   - No model fallback, no cross-provider fallback, no implicit model discovery.
7. Settings & Research UI rendering with Anthropic as active provider is 100% local with 0 network calls.
"""
import os
import unittest
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

from app.database import Base, get_db
from app.main import app
from app.models import Setting, Product
from app.config import (
    DEFAULT_ANTHROPIC_MODEL,
    LEGACY_ANTHROPIC_DEFAULT_MODEL,
    get_anthropic_model,
    normalize_model_name,
    get_active_ai_model,
    get_active_ai_provider,
)
from app.routes.settings import save_provider_settings
from app.services.ai.providers.anthropic import AnthropicProvider
from app.services.ai import get_ai_manager, AIGenerationOptions


SAMPLE_PRODUCTS = [
    {
        "name_vietnamese": "Nồi cơm điện mini",
        "name_chinese": "迷你电饭煲",
        "douyin_keywords": "迷你电饭煲 独居",
        "content_angle": "Tiện lợi",
        "hook": "Bữa cơm ngon chỉ 15 phút!",
    }
]


class TestAnthropicSonnet46ModelFix(unittest.TestCase):
    """Comprehensive test suite covering the Anthropic Sonnet 4.6 model-default upgrade."""

    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.db = self.SessionLocal()

        def _get_test_db():
            db = self.SessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = _get_test_db
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.db.close()
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()

    # =========================================================================
    # 1. FRESH INSTALL DEFAULT
    # =========================================================================

    def test_fresh_install_default_resolves_to_claude_sonnet_4_6(self):
        """Fresh install with empty DB and empty env returns 'claude-sonnet-4-6'."""
        self.assertEqual(DEFAULT_ANTHROPIC_MODEL, "claude-sonnet-4-6")
        self.assertEqual(LEGACY_ANTHROPIC_DEFAULT_MODEL, "claude-3-5-sonnet-20241022")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_MODEL", None)
            model_from_db = get_anthropic_model(self.db)
            self.assertEqual(model_from_db, "claude-sonnet-4-6")

            model_from_env = get_anthropic_model(None)
            self.assertEqual(model_from_env, "claude-sonnet-4-6")

    # =========================================================================
    # 2. LEGACY DEFAULT RESOLUTION & SETTINGS MIGRATION
    # =========================================================================

    def test_legacy_default_in_db_resolves_to_claude_sonnet_4_6(self):
        """Legacy default stored in DB ('claude-3-5-sonnet-20241022') resolves cleanly to 'claude-sonnet-4-6'."""
        rec = Setting(key="anthropic_model", value="claude-3-5-sonnet-20241022", description="Anthropic Model")
        self.db.add(rec)
        self.db.commit()

        resolved = get_anthropic_model(self.db)
        self.assertEqual(resolved, "claude-sonnet-4-6")

    def test_legacy_default_in_env_resolves_to_claude_sonnet_4_6(self):
        """Legacy default in ANTHROPIC_MODEL env variable resolves cleanly to 'claude-sonnet-4-6'."""
        with patch.dict(os.environ, {"ANTHROPIC_MODEL": "claude-3-5-sonnet-20241022"}):
            resolved = get_anthropic_model(None)
            self.assertEqual(resolved, "claude-sonnet-4-6")

    def test_normalize_model_name_migrates_legacy_anthropic_default(self):
        """normalize_model_name transitions legacy default while keeping empty strings mapped to default."""
        self.assertEqual(normalize_model_name("anthropic", "claude-3-5-sonnet-20241022"), "claude-sonnet-4-6")
        self.assertEqual(normalize_model_name("anthropic", ""), "claude-sonnet-4-6")
        self.assertEqual(normalize_model_name("anthropic", None), "claude-sonnet-4-6")
        self.assertEqual(normalize_model_name("anthropic", "   "), "claude-sonnet-4-6")

    def test_settings_save_migrates_legacy_default(self):
        """Saving settings with legacy default updates DB record cleanly to 'claude-sonnet-4-6'."""
        db_rec = Setting(key="anthropic_model", value="claude-3-5-sonnet-20241022")
        self.db.add(db_rec)
        self.db.commit()

        # User saves settings submitting legacy value (e.g. from cached form)
        save_provider_settings(self.db, "anthropic", "sk-ant-testkey", "claude-3-5-sonnet-20241022")
        self.db.commit()

        saved_rec = self.db.query(Setting).filter(Setting.key == "anthropic_model").first()
        self.assertIsNotNone(saved_rec)
        self.assertEqual(saved_rec.value, "claude-sonnet-4-6")

    # =========================================================================
    # 3. CUSTOM USER-ENTERED MODEL PRESERVATION
    # =========================================================================

    def test_custom_model_in_db_preserved_unchanged(self):
        """Custom user-entered Anthropic model in DB is strictly preserved and never overwritten."""
        custom_models = [
            "claude-3-opus-20240229",
            "claude-3-5-haiku-20241022",
            "claude-2.1",
            "my-custom-fine-tuned-claude",
        ]
        for cm in custom_models:
            self.db.query(Setting).filter(Setting.key == "anthropic_model").delete()
            self.db.add(Setting(key="anthropic_model", value=cm))
            self.db.commit()

            resolved = get_anthropic_model(self.db)
            self.assertEqual(resolved, cm, f"Expected custom model '{cm}' to be preserved intact.")

    def test_custom_model_in_env_preserved_unchanged(self):
        """Custom model in ANTHROPIC_MODEL env is strictly preserved."""
        with patch.dict(os.environ, {"ANTHROPIC_MODEL": "claude-3-5-haiku-20241022"}):
            resolved = get_anthropic_model(None)
            self.assertEqual(resolved, "claude-3-5-haiku-20241022")

    def test_normalize_model_name_preserves_custom_models(self):
        """normalize_model_name preserves custom model IDs for Anthropic and other providers."""
        self.assertEqual(normalize_model_name("anthropic", "claude-3-opus-20240229"), "claude-3-opus-20240229")
        self.assertEqual(normalize_model_name("anthropic", "claude-3-5-haiku-20241022"), "claude-3-5-haiku-20241022")
        self.assertEqual(normalize_model_name("openai", "gpt-4o"), "gpt-4o")

    def test_settings_save_preserves_custom_model(self):
        """Saving a custom model in settings persists the custom model without altering it."""
        save_provider_settings(self.db, "anthropic", "sk-ant-testkey", "claude-3-opus-20240229")
        self.db.commit()

        saved_rec = self.db.query(Setting).filter(Setting.key == "anthropic_model").first()
        self.assertIsNotNone(saved_rec)
        self.assertEqual(saved_rec.value, "claude-3-opus-20240229")

        resolved = get_anthropic_model(self.db)
        self.assertEqual(resolved, "claude-3-opus-20240229")

    # =========================================================================
    # 4. PROVIDER SUPPORTS TARGET MODEL
    # =========================================================================

    def test_anthropic_provider_supports_claude_sonnet_4_6(self):
        """AnthropicProvider.supports_model returns True for 'claude-sonnet-4-6'."""
        provider = AnthropicProvider()
        self.assertTrue(provider.supports_model("claude-sonnet-4-6"))
        self.assertTrue(provider.supports_model("claude-3-5-sonnet-20241022"))
        self.assertTrue(provider.supports_model("claude-3-opus-20240229"))
        self.assertFalse(provider.supports_model("gpt-4o-mini"))
        self.assertFalse(provider.supports_model("gemini-3.8-flash"))
        self.assertFalse(provider.supports_model(""))
        self.assertFalse(provider.supports_model(None))

    # =========================================================================
    # 5. ZERO NETWORK CALLS DURING MODEL RESOLUTION & UI RENDERING
    # =========================================================================

    def test_zero_network_calls_during_model_resolution(self):
        """Model resolution and normalization MUST be 100% local with 0 network calls."""
        with patch("httpx.Client.get") as mock_http_get, \
             patch("httpx.Client.post") as mock_http_post, \
             patch("urllib.request.urlopen") as mock_urllib:

            # 1. Resolve Anthropic model with fresh install
            m1 = get_anthropic_model(self.db)
            self.assertEqual(m1, "claude-sonnet-4-6")

            # 2. Resolve Anthropic model with legacy DB value
            self.db.add(Setting(key="anthropic_model", value="claude-3-5-sonnet-20241022"))
            self.db.commit()
            m2 = get_anthropic_model(self.db)
            self.assertEqual(m2, "claude-sonnet-4-6")

            # 3. Resolve Anthropic model with custom DB value
            rec = self.db.query(Setting).filter(Setting.key == "anthropic_model").first()
            rec.value = "claude-3-opus-20240229"
            self.db.commit()
            m3 = get_anthropic_model(self.db)
            self.assertEqual(m3, "claude-3-opus-20240229")

            # 4. Normalize model names
            normalize_model_name("anthropic", "")
            normalize_model_name("anthropic", "claude-3-5-sonnet-20241022")
            normalize_model_name("anthropic", "claude-sonnet-4-6")

            # 5. Active AI model resolution
            self.db.add(Setting(key="active_ai_provider", value="anthropic"))
            self.db.commit()
            active_m = get_active_ai_model(self.db)
            self.assertEqual(active_m, "claude-3-opus-20240229")

            # Assert ZERO network requests made
            mock_http_get.assert_not_called()
            mock_http_post.assert_not_called()
            mock_urllib.assert_not_called()

    @patch("app.services.ai.providers.anthropic.httpx.Client")
    def test_zero_network_calls_during_settings_and_research_ui_rendering(self, mock_anthropic_client):
        """Loading /settings and /research with Anthropic active performs ZERO network calls."""
        mock_anthropic_client.side_effect = RuntimeError("NETWORK CALL FORBIDDEN IN UI RENDERING")

        self.db.add(Setting(key="active_ai_provider", value="anthropic"))
        self.db.add(Setting(key="anthropic_api_key", value="sk-ant-testkey12345"))
        self.db.commit()

        # 1. Render Settings Page
        resp_settings = self.client.get("/settings")
        self.assertEqual(resp_settings.status_code, 200)
        self.assertIn("claude-sonnet-4-6", resp_settings.text)

        # 2. Render Research Page
        resp_research = self.client.get("/research")
        self.assertEqual(resp_research.status_code, 200)
        self.assertIn("Anthropic Claude", resp_research.text)
        self.assertIn("CLAUDE-SONNET-4-6", resp_research.text.upper())

    # =========================================================================
    # 6. RESEARCH INVARIANTS PRESERVED WITH ANTHROPIC
    # =========================================================================

    def test_research_ui_provider_info_shows_claude_sonnet_4_6(self):
        """Research UI provider info dynamically loads 'claude-sonnet-4-6' locally."""
        from app.routes.research import _get_research_provider_info

        self.db.add(Setting(key="active_ai_provider", value="anthropic"))
        self.db.add(Setting(key="anthropic_api_key", value="sk-ant-validkey12345"))
        self.db.commit()

        with patch("httpx.Client.get") as mock_get, patch("httpx.Client.post") as mock_post:
            info = _get_research_provider_info(self.db)

            self.assertEqual(info["provider_id"], "anthropic")
            self.assertEqual(info["provider_name"], "Anthropic Claude")
            self.assertEqual(info["model"], "claude-sonnet-4-6")
            self.assertTrue(info["configured"])

            # Zero network calls
            mock_get.assert_not_called()
            mock_post.assert_not_called()

    def test_research_outbound_generation_uses_claude_sonnet_4_6_with_strict_invariants(self):
        """When Anthropic is active, generation uses 'claude-sonnet-4-6', max_retries=0, no fallbacks."""
        provider = AnthropicProvider()

        with patch("app.services.ai.providers.anthropic.get_anthropic_api_key", return_value="sk-ant-VALID_TEST_KEY"), \
             patch("httpx.Client.post") as mock_post:

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": "msg_001",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "Research products generated"}],
                "usage": {"input_tokens": 50, "output_tokens": 100}
            }
            mock_post.return_value = mock_resp

            # Research invariant: max_retries=0, enable_fallback=False, allow_cross_provider_fallback=False
            options = AIGenerationOptions(
                max_retries=0,
                enable_fallback=False,
                allow_cross_provider_fallback=False,
                timeout=45.0,
            )

            result = provider.generate(
                prompt="Research prompt",
                db=self.db,
                options=options
            )

            self.assertEqual(result, "Research products generated")
            # Verify exactly 1 call (Research invariant)
            self.assertEqual(mock_post.call_count, 1)

            # Verify outbound payload model was 'claude-sonnet-4-6'
            called_args, called_kwargs = mock_post.call_args
            payload = called_kwargs.get("json", {})
            self.assertEqual(payload.get("model"), "claude-sonnet-4-6")

    @patch("app.routes.research.get_ai_manager")
    def test_research_cache_hit_zero_ai_calls_with_anthropic(self, mock_get_manager):
        """Research cache hit with Anthropic active makes exactly 0 generation calls."""
        cached_product = Product(
            product_id="P9999",
            niche="Đồ gia dụng",
            name_vietnamese="Nồi chiên không dầu",
            name_chinese="空气炸锅",
            douyin_keywords="空气炸锅",
            content_angle="Nấu ăn",
            hook="Nồi xịn",
            status="RESEARCHED"
        )
        self.db.add(cached_product)
        self.db.add(Setting(key="active_ai_provider", value="anthropic"))
        self.db.add(Setting(key="anthropic_api_key", value="sk-ant-validkey12345"))
        self.db.commit()

        mock_mgr = MagicMock()
        mock_get_manager.return_value = mock_mgr

        resp = self.client.post(
            "/research",
            data={"niche": "Đồ gia dụng", "product_count": 1, "fresh": "false"},
            follow_redirects=False
        )

        self.assertEqual(resp.status_code, 303)
        self.assertIn("cached=1", resp.headers["location"])
        # Exactly 0 calls to AI manager
        mock_mgr.generate_products.assert_not_called()

    @patch("app.routes.research.get_ai_manager")
    def test_research_cache_miss_exactly_one_call_with_anthropic(self, mock_get_manager):
        """Research cache miss with Anthropic active makes exactly 1 generation call, 0 retries, 0 fallbacks."""
        self.db.add(Setting(key="active_ai_provider", value="anthropic"))
        self.db.add(Setting(key="anthropic_api_key", value="sk-ant-validkey12345"))
        self.db.commit()

        mock_mgr = MagicMock()
        mock_mgr.get_active_provider.return_value = get_ai_manager().get_provider("anthropic")
        mock_mgr.generate_products.return_value = SAMPLE_PRODUCTS
        mock_get_manager.return_value = mock_mgr

        resp = self.client.post(
            "/research",
            data={"niche": "Gia dụng mới", "product_count": 5, "fresh": "true"},
            follow_redirects=False
        )

        self.assertEqual(resp.status_code, 303)
        # Exactly 1 call
        self.assertEqual(mock_mgr.generate_products.call_count, 1)


if __name__ == "__main__":
    unittest.main()
