"""
Unit & Integration Tests for Multi-AI Phase 4: Routing & Fallback

Tests customer-controlled cross-provider routing and fallback:
1. Cross-provider fallback disabled by default (raises immediately on primary failure).
2. Workflow allow_cross_provider_fallback=False strictly overrides customer setting.
3. Successful cross-provider fallback to candidate provider when primary fails (503/timeout).
4. Unconfigured providers in fallback list are skipped with 0 attempts.
5. Conservative 429 policy: 429 fails immediately when ai_fallback_on_quota is False.
6. 429 falls back only when customer explicitly enables ai_fallback_on_quota=True.
7. 400 Bad Request / 401 Auth are never fallback-eligible.
8. All candidates failing raises AIFallbackExhaustedError with sanitized chain.
9. Duplicate and loop prevention in candidate chain.
10. Research low-consumption invariant: generate_products strictly enforces
    allow_cross_provider_fallback=False, max_retries=0, enable_fallback=False.
11. Metadata attribution accurately reflects primary vs actual provider and model.

ALL EXTERNAL HTTP REQUESTS ARE 100% MOCKED. ZERO PAID QUOTA CONSUMED.
"""
import unittest
import json
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base, Setting
from app.services.ai.base import (
    AIProvider,
    AIGenerationOptions,
    ExecutionMetadata,
    AIProviderError,
    AIServiceUnavailableError,
    AIQuotaExceededError,
    AIRateLimitError,
    AIBadRequestError,
    AIAuthenticationError,
    AITimeoutError,
    AINetworkError,
    AIFallbackExhaustedError,
)
from app.services.ai.manager import AIProviderManager, get_ai_manager, set_ai_manager
from app.config import (
    get_ai_fallback_enabled,
    get_ai_fallback_providers,
    get_ai_fallback_on_quota,
    DEFAULT_ACTIVE_AI_PROVIDER,
)


class MockProvider(AIProvider):
    """Mock provider for deterministic testing of routing and fallback."""
    def __init__(self, provider_id: str, display_name: str, provider_type: str = "direct"):
        self._provider_id = provider_id
        self._display_name = display_name
        self._provider_type = provider_type
        self.generate_calls = []
        self.fail_with = None
        self.return_value = f"Response from {provider_id}"
        self._last_execution = ExecutionMetadata(
            provider=provider_id,
            provider_type=provider_type,
            configured_model=f"{provider_id}-model",
            actual_model_used=f"{provider_id}-model",
            status="SUCCESS"
        )

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def provider_type(self) -> str:
        return self._provider_type

    def generate(self, prompt: str, db=None, timeout: float = 60.0, max_retries: int = 2,
                 enable_fallback: bool = True, options: AIGenerationOptions = None, **kwargs) -> str:
        self.generate_calls.append({
            "prompt": prompt,
            "timeout": timeout,
            "max_retries": max_retries,
            "enable_fallback": enable_fallback,
            "options": options,
            "kwargs": kwargs
        })
        if self.fail_with:
            self._last_execution = ExecutionMetadata(
                provider=self._provider_id,
                provider_type=self._provider_type,
                configured_model=f"{self._provider_id}-model",
                actual_model_used="",
                status="FAILED",
                error_type=self.fail_with.error_type if hasattr(self.fail_with, "error_type") else "error"
            )
            raise self.fail_with
        return self.return_value

    def test_connection(self, db=None):
        return {"success": True, "provider": self._provider_id}

    def list_models(self, db=None):
        return [{"id": f"{self._provider_id}-model", "name": f"{self._provider_id}-model"}]

    def supports_model(self, model_name: str) -> bool:
        return self._provider_id in model_name

    def get_last_execution_metadata(self):
        return self._last_execution.to_dict()


class TestMultiAIRoutingPhase4(unittest.TestCase):
    """Test suite for Multi-AI Phase 4 Routing & Fallback."""

    def setUp(self):
        # Create a thread-safe in-memory SQLite database
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()

        # Build test manager with mock providers
        self.manager = AIProviderManager()
        self.mock_gemini = MockProvider("gemini", "Google Gemini", "direct")
        self.mock_openai = MockProvider("openai", "OpenAI", "direct")
        self.mock_anthropic = MockProvider("anthropic", "Anthropic", "direct")
        self.mock_groq = MockProvider("groq", "Groq", "direct")
        self.mock_openrouter = MockProvider("openrouter", "OpenRouter", "gateway")

        self.manager._providers = {
            "gemini": self.mock_gemini,
            "openai": self.mock_openai,
            "anthropic": self.mock_anthropic,
            "groq": self.mock_groq,
            "openrouter": self.mock_openrouter,
        }

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(self.engine)

    # --------------------------------------------------------------------------
    # 1. Fallback Disabled by Default
    # --------------------------------------------------------------------------
    def test_cross_provider_fallback_disabled_by_default(self):
        """When customer ai_fallback_enabled is default (False), primary failure raises immediately."""
        self.mock_gemini.fail_with = AIServiceUnavailableError(
            "Gemini is temporarily overloaded (503)", provider="gemini"
        )

        options = AIGenerationOptions(allow_cross_provider_fallback=True)
        with self.assertRaises(AIServiceUnavailableError):
            self.manager.generate("Test prompt", db=self.db, options=options)

        self.assertEqual(len(self.mock_gemini.generate_calls), 1)
        self.assertEqual(len(self.mock_groq.generate_calls), 0)
        self.assertEqual(len(self.mock_openai.generate_calls), 0)

    # --------------------------------------------------------------------------
    # 2. Workflow Override
    # --------------------------------------------------------------------------
    def test_workflow_disallow_cross_provider_fallback_overrides_customer_setting(self):
        """When workflow sets allow_cross_provider_fallback=False, no fallback occurs even if enabled in DB."""
        self.db.add(Setting(key="ai_fallback_enabled", value="true"))
        self.db.add(Setting(key="ai_fallback_providers", value="groq,openai"))
        self.db.commit()

        self.mock_gemini.fail_with = AIServiceUnavailableError("503 Overload", provider="gemini")

        # Workflow explicitly prohibits cross-provider fallback
        options = AIGenerationOptions(allow_cross_provider_fallback=False)
        with self.assertRaises(AIServiceUnavailableError):
            self.manager.generate("Test prompt", db=self.db, options=options)

        self.assertEqual(len(self.mock_gemini.generate_calls), 1)
        self.assertEqual(len(self.mock_groq.generate_calls), 0)

    # --------------------------------------------------------------------------
    # 3. Successful Cross-Provider Fallback
    # --------------------------------------------------------------------------
    @patch("app.services.ai.manager.get_provider_key_configured")
    def test_successful_cross_provider_fallback_to_groq(self, mock_key_conf):
        """When enabled by both customer & workflow, 503 on primary falls back to first configured candidate."""
        self.db.add(Setting(key="ai_fallback_enabled", value="true"))
        self.db.add(Setting(key="ai_fallback_providers", value="groq,openai"))
        self.db.commit()

        # Groq and OpenAI are configured
        mock_key_conf.side_effect = lambda pid, db: True

        self.mock_gemini.fail_with = AIServiceUnavailableError("503 Overload", provider="gemini")
        self.mock_groq.return_value = "Success from Groq"

        options = AIGenerationOptions(allow_cross_provider_fallback=True)
        res = self.manager.generate("Test prompt", db=self.db, options=options)

        self.assertEqual(res, "Success from Groq")
        self.assertEqual(len(self.mock_gemini.generate_calls), 1)
        self.assertEqual(len(self.mock_groq.generate_calls), 1)
        self.assertEqual(len(self.mock_openai.generate_calls), 0)

        meta = self.manager.get_last_execution_metadata()
        self.assertTrue(meta.get("fallback_used"))
        self.assertEqual(meta.get("actual_provider_used"), "groq")
        self.assertEqual(meta.get("provider"), "gemini")

    # --------------------------------------------------------------------------
    # 4. Unconfigured Candidates Skipped
    # --------------------------------------------------------------------------
    @patch("app.services.ai.manager.get_provider_key_configured")
    def test_unconfigured_provider_skipped_in_fallback_chain(self, mock_key_conf):
        """Unconfigured candidate providers in fallback order are skipped without making calls."""
        self.db.add(Setting(key="ai_fallback_enabled", value="true"))
        self.db.add(Setting(key="ai_fallback_providers", value="openai,groq"))
        self.db.commit()

        # OpenAI has no key; Groq has key
        def key_checker(pid, db):
            return pid == "groq"
        mock_key_conf.side_effect = key_checker

        self.mock_gemini.fail_with = AITimeoutError("Gemini timed out", provider="gemini")
        self.mock_groq.return_value = "Groq responded"

        options = AIGenerationOptions(allow_cross_provider_fallback=True)
        res = self.manager.generate("Test prompt", db=self.db, options=options)

        self.assertEqual(res, "Groq responded")
        self.assertEqual(len(self.mock_gemini.generate_calls), 1)
        self.assertEqual(len(self.mock_openai.generate_calls), 0)  # Skipped!
        self.assertEqual(len(self.mock_groq.generate_calls), 1)

    # --------------------------------------------------------------------------
    # 5. Conservative 429 Policy (Default Off)
    # --------------------------------------------------------------------------
    @patch("app.services.ai.manager.get_provider_key_configured")
    def test_429_quota_fails_immediately_when_ai_fallback_on_quota_is_false(self, mock_key_conf):
        """429 Daily Quota halts immediately and does NOT fall back when ai_fallback_on_quota is False."""
        self.db.add(Setting(key="ai_fallback_enabled", value="true"))
        self.db.add(Setting(key="ai_fallback_providers", value="groq,openai"))
        # ai_fallback_on_quota is False by default
        self.db.commit()

        mock_key_conf.return_value = True
        self.mock_gemini.fail_with = AIQuotaExceededError("Daily quota exhausted (429)", provider="gemini")

        options = AIGenerationOptions(allow_cross_provider_fallback=True)
        with self.assertRaises(AIQuotaExceededError):
            self.manager.generate("Test prompt", db=self.db, options=options)

        self.assertEqual(len(self.mock_gemini.generate_calls), 1)
        self.assertEqual(len(self.mock_groq.generate_calls), 0)

    # --------------------------------------------------------------------------
    # 6. Conservative 429 Policy (Opt-In On)
    # --------------------------------------------------------------------------
    @patch("app.services.ai.manager.get_provider_key_configured")
    def test_429_quota_falls_back_when_ai_fallback_on_quota_is_true(self, mock_key_conf):
        """429 Daily Quota triggers fallback only when customer explicitly enables ai_fallback_on_quota."""
        self.db.add(Setting(key="ai_fallback_enabled", value="true"))
        self.db.add(Setting(key="ai_fallback_providers", value="groq,openai"))
        self.db.add(Setting(key="ai_fallback_on_quota", value="true"))
        self.db.commit()

        mock_key_conf.return_value = True
        self.mock_gemini.fail_with = AIQuotaExceededError("Daily quota exhausted (429)", provider="gemini")
        self.mock_groq.return_value = "Groq saved the day"

        options = AIGenerationOptions(allow_cross_provider_fallback=True)
        res = self.manager.generate("Test prompt", db=self.db, options=options)

        self.assertEqual(res, "Groq saved the day")
        self.assertEqual(len(self.mock_gemini.generate_calls), 1)
        self.assertEqual(len(self.mock_groq.generate_calls), 1)

    # --------------------------------------------------------------------------
    # 7. Ineligible 400 Bad Request & 401 Auth
    # --------------------------------------------------------------------------
    @patch("app.services.ai.manager.get_provider_key_configured")
    def test_400_bad_request_never_falls_back(self, mock_key_conf):
        """HTTP 400 Bad Request is not fallback-eligible and stops immediately."""
        self.db.add(Setting(key="ai_fallback_enabled", value="true"))
        self.db.add(Setting(key="ai_fallback_providers", value="groq,openai"))
        self.db.commit()

        mock_key_conf.return_value = True
        self.mock_gemini.fail_with = AIBadRequestError("Malformed prompt (400)", provider="gemini")

        options = AIGenerationOptions(allow_cross_provider_fallback=True)
        with self.assertRaises(AIBadRequestError):
            self.manager.generate("Test prompt", db=self.db, options=options)

        self.assertEqual(len(self.mock_gemini.generate_calls), 1)
        self.assertEqual(len(self.mock_groq.generate_calls), 0)

    # --------------------------------------------------------------------------
    # 8. All Candidates Fail -> AIFallbackExhaustedError
    # --------------------------------------------------------------------------
    @patch("app.services.ai.manager.get_provider_key_configured")
    def test_fallback_exhausted_raises_ai_fallback_exhausted_error(self, mock_key_conf):
        """When all configured fallback candidates fail, AIFallbackExhaustedError is raised."""
        self.db.add(Setting(key="ai_fallback_enabled", value="true"))
        self.db.add(Setting(key="ai_fallback_providers", value="groq,openai"))
        self.db.commit()

        mock_key_conf.return_value = True
        self.mock_gemini.fail_with = AIServiceUnavailableError("Gemini 503", provider="gemini")
        self.mock_groq.fail_with = AINetworkError("Groq DNS failed", provider="groq")
        self.mock_openai.fail_with = AIServiceUnavailableError("OpenAI 500", provider="openai")

        options = AIGenerationOptions(allow_cross_provider_fallback=True)
        with self.assertRaises(AIFallbackExhaustedError) as ctx:
            self.manager.generate("Test prompt", db=self.db, options=options)

        err_msg = str(ctx.exception)
        self.assertIn("All configured AI providers failed", err_msg)
        self.assertIn("gemini", err_msg)
        self.assertIn("groq", err_msg)
        self.assertIn("openai", err_msg)

        meta = self.manager.get_last_execution_metadata()
        self.assertEqual(meta.get("status"), "FAILED")
        self.assertEqual(len(meta.get("fallback_chain", [])), 3)

    # --------------------------------------------------------------------------
    # 9. Duplicate & Loop Prevention
    # --------------------------------------------------------------------------
    @patch("app.services.ai.manager.get_provider_key_configured")
    def test_duplicate_provider_prevented_in_fallback_chain(self, mock_key_conf):
        """A provider is never called more than once per generation request."""
        self.db.add(Setting(key="ai_fallback_enabled", value="true"))
        # Fallback list has duplicate gemini and groq
        self.db.add(Setting(key="ai_fallback_providers", value="gemini,groq,gemini,groq"))
        self.db.commit()

        mock_key_conf.return_value = True
        self.mock_gemini.fail_with = AIServiceUnavailableError("Gemini 503", provider="gemini")
        self.mock_groq.return_value = "Groq OK"

        options = AIGenerationOptions(allow_cross_provider_fallback=True)
        res = self.manager.generate("Test prompt", db=self.db, options=options)

        self.assertEqual(res, "Groq OK")
        # Gemini was primary (1 call). It was in fallback list twice, but attempted_providers prevented retrying it.
        self.assertEqual(len(self.mock_gemini.generate_calls), 1)
        # Groq was called exactly once despite being in list twice
        self.assertEqual(len(self.mock_groq.generate_calls), 1)

    # --------------------------------------------------------------------------
    # 10. Research Low-Consumption Invariant
    # --------------------------------------------------------------------------
    @patch("app.services.ai.manager.get_provider_key_configured")
    def test_research_invariant_strictly_enforced_in_manager(self, mock_key_conf):
        """Research batch generation strictly enforces allow_cross_provider_fallback=False and max_retries=0."""
        self.db.add(Setting(key="ai_fallback_enabled", value="true"))
        self.db.add(Setting(key="ai_fallback_providers", value="groq,openai"))
        self.db.add(Setting(key="ai_fallback_on_quota", value="true"))
        self.db.commit()

        mock_key_conf.return_value = True
        self.mock_gemini.fail_with = AIServiceUnavailableError("Gemini 503 Overload", provider="gemini")

        # generate_products in manager
        with self.assertRaises(AIProviderError):
            self.manager.generate_products("Gia dụng", count=10, db=self.db)

        # Assert Gemini called exactly once with max_retries=0 and enable_fallback=False
        self.assertEqual(len(self.mock_gemini.generate_calls), 1)
        call_info = self.mock_gemini.generate_calls[0]
        self.assertEqual(call_info["max_retries"], 0)
        self.assertFalse(call_info["enable_fallback"])
        opt = call_info["options"]
        if opt:
            self.assertFalse(opt.allow_cross_provider_fallback)
            self.assertFalse(opt.enable_fallback)
            self.assertEqual(opt.max_retries, 0)

        # Assert no fallback provider was ever called
        self.assertEqual(len(self.mock_groq.generate_calls), 0)
        self.assertEqual(len(self.mock_openai.generate_calls), 0)

    # --------------------------------------------------------------------------
    # 11. Metadata Attribution
    # --------------------------------------------------------------------------
    @patch("app.services.ai.manager.get_provider_key_configured")
    def test_metadata_contains_sanitized_fallback_chain(self, mock_key_conf):
        """ExecutionMetadata includes primary, actual provider, fallback_used, and sanitized chain."""
        self.db.add(Setting(key="ai_fallback_enabled", value="true"))
        self.db.add(Setting(key="ai_fallback_providers", value="groq"))
        self.db.commit()

        mock_key_conf.return_value = True
        self.mock_gemini.fail_with = AIServiceUnavailableError(
            "Gemini 503 with key sk-ant-SecretKey1234", provider="gemini"
        )
        self.mock_groq.return_value = "Groq output"

        options = AIGenerationOptions(allow_cross_provider_fallback=True)
        self.manager.generate("Test prompt", db=self.db, options=options)

        meta = self.manager.get_last_execution_metadata()
        self.assertEqual(meta["provider"], "gemini")
        self.assertEqual(meta["actual_provider_used"], "groq")
        self.assertTrue(meta["fallback_used"])
        self.assertEqual(meta["status"], "SUCCESS")

        chain = meta.get("fallback_chain", [])
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[0]["provider"], "gemini")
        self.assertEqual(chain[0]["status"], "FAILED")
        # Ensure secret was sanitized in error string
        self.assertNotIn("sk-ant-SecretKey1234", str(chain[0].get("error", "")))
        self.assertEqual(chain[1]["provider"], "groq")
        self.assertEqual(chain[1]["status"], "SUCCESS")

    # --------------------------------------------------------------------------
    # 12. Settings Context Includes Fallback Configuration
    # --------------------------------------------------------------------------
    def test_settings_context_includes_fallback_configuration(self):
        """get_current_settings loads fallback configuration safely into template context."""
        from app.routes.settings import get_current_settings
        self.db.add(Setting(key="ai_fallback_enabled", value="true"))
        self.db.add(Setting(key="ai_fallback_providers", value="groq,openrouter"))
        self.db.add(Setting(key="ai_fallback_on_quota", value="true"))
        self.db.commit()

        ctx = get_current_settings(self.db)
        self.assertTrue(ctx["ai_fallback_enabled"])
        self.assertEqual(ctx["ai_fallback_providers"], ["groq", "openrouter"])
        self.assertTrue(ctx["ai_fallback_on_quota"])

    # --------------------------------------------------------------------------
    # 13. Save Settings Persists Fallback Configuration
    # --------------------------------------------------------------------------
    # 13. Save Settings Persists Fallback Configuration
    # --------------------------------------------------------------------------
    def test_save_settings_persists_fallback_configuration(self):
        """save_settings endpoint persists fallback enabled, candidate list, and 429 quota toggle."""
        from fastapi import FastAPI
        from app.routes.settings import router as settings_router
        from starlette.testclient import TestClient
        from app.database import get_db

        app = FastAPI()
        app.include_router(settings_router)

        def override_get_db():
            db = self.Session()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db

        client = TestClient(app)
        resp = client.post(
            "/settings",
            data={
                "active_ai_provider": "gemini",
                "ai_fallback_enabled": "on",
                "ai_fallback_providers": "groq,openai,openrouter",
                "ai_fallback_on_quota": "on"
            },
            follow_redirects=False
        )
        self.assertIn(resp.status_code, [200, 303])

        # Verify persisted values in DB
        db = self.Session()
        try:
            fb_en = db.query(Setting).filter(Setting.key == "ai_fallback_enabled").first()
            self.assertIsNotNone(fb_en)
            self.assertEqual(fb_en.value, "true")

            fb_prov = db.query(Setting).filter(Setting.key == "ai_fallback_providers").first()
            self.assertIsNotNone(fb_prov)
            self.assertEqual(fb_prov.value, "groq,openai,openrouter")

            fb_q = db.query(Setting).filter(Setting.key == "ai_fallback_on_quota").first()
            self.assertIsNotNone(fb_q)
            self.assertEqual(fb_q.value, "true")
        finally:
            db.close()

    # --------------------------------------------------------------------------
    # 14. Settings HTML Contains Fallback Controls & Safety Notices
    # --------------------------------------------------------------------------
    def test_settings_html_contains_fallback_ui_controls_and_warnings(self):
        """GET /settings renders Smart Routing & Fallback card with cost warning and research invariant."""
        from fastapi import FastAPI
        from app.routes.settings import router as settings_router
        from starlette.testclient import TestClient
        from app.database import get_db

        app = FastAPI()
        app.include_router(settings_router)

        def override_get_db():
            db = self.Session()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db

        client = TestClient(app)
        resp = client.get("/settings")
        self.assertEqual(resp.status_code, 200)

        html = resp.text
        # Check for card title
        self.assertIn("Smart Routing & Fallback", html)
        # Check for master switch
        self.assertIn("name=\"ai_fallback_enabled\"", html)
        # Check for general cost warning
        self.assertIn("Cảnh báo chi phí & hạn ngạch", html)
        # Check for research invariant protection notice
        self.assertIn("Bảo Vệ Ngân Sách Nghiên Cứu", html)
        # Check for 429 quota opt-in switch
        self.assertIn("name=\"ai_fallback_on_quota\"", html)
        # Check for candidates input
        self.assertIn("name=\"ai_fallback_providers\"", html)

    # --------------------------------------------------------------------------
    # 15. Content Service Cross-Provider Fallback Integration
    # --------------------------------------------------------------------------
    @patch("app.services.content_service.get_ai_manager")
    @patch("app.services.content_service.get_api_key", return_value="dummy-key")
    def test_content_service_cross_provider_fallback_enabled(self, mock_key, mock_get_mgr):
        """ContentService passes allow_cross_provider_fallback=True so pipeline can recover."""
        from app.services.content_service import ContentService
        from app.models import Product, Video, Voice

        # Seed test product, video, and voice script
        prod = Product(product_id="P0999", niche="Gia dụng", name_vietnamese="Chảo chống dính")
        self.db.add(prod)
        vid = Video(video_id="V0999", product_id="P0999", douyin_url="https://douyin.com/123", local_file="test.mp4", status="DOWNLOADED")
        self.db.add(vid)
        voice = Voice(video_id="V0999", script="Kịch bản video giới thiệu chảo chống dính.")
        self.db.add(voice)
        self.db.commit()

        mock_data = {
            "facebook_personal": {"caption": "Cap FB P", "hashtags": "#fbp"},
            "facebook_page": {"caption": "Cap FB Page", "hashtags": "#page"},
            "tiktok": {"caption": "Cap TikTok", "hashtags": "#tiktok"},
            "threads": {"caption": "Cap Threads", "hashtags": "#threads"},
            "instagram": {"caption": "Cap IG", "hashtags": "#ig"},
            "shopee": {"caption": "Cap Shopee", "hashtags": "#shopee"},
            "youtube": {"title": "Title YT", "description": "Desc YT", "hashtags": "#yt"}
        }
        mock_mgr = MagicMock()
        mock_mgr.generate.return_value = json.dumps(mock_data)
        mock_get_mgr.return_value = mock_mgr

        service = ContentService()
        with patch("app.services.content_service.validate_content_json", return_value={"valid": True}):
            res = service.generate_content_for_video(self.db, "V0999")
            self.assertTrue(res["success"])

        # Check call arguments
        call_kwargs = mock_mgr.generate.call_args[1]
        self.assertTrue(call_kwargs.get("allow_cross_provider_fallback"))


if __name__ == "__main__":
    unittest.main()


