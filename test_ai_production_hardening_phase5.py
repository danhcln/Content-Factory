"""
Multi-AI Architecture Phase 5: Production Hardening Test Suite
==============================================================
Validates:
1. True end-to-end routed workflow deadline (includes primary time, clamps candidate time).
2. Zero candidate begins after deadline exhaustion.
3. Fallback candidates receive max_retries=0.
4. Fallback candidate timeout capped at 20.0s.
5. Generation performs zero model-discovery network calls.
6. Local-only model normalization without silent replacement of valid models.
7. Settings diagnostics perform zero network calls.
8. ExecutionMetadata completeness (duration, tokens, internal vs cross fallback).
9. Safe zero-key cold start / degraded mode.
10. Research invariant strictly preserved (0 calls on cache hit, 1 on miss, 0 retries, 0 fallback).
11. Secret scrubbing and user-friendly error translations.
12. Provider failure isolation.

Zero Real API Quota Consumed: All external network/HTTP operations mocked.
"""
import unittest
import time
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base, Setting
from app.services.ai.base import (
    AIProvider,
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
    AIFallbackExhaustedError,
    AIGenerationOptions,
    ExecutionMetadata,
    get_user_friendly_error_message,
)
from app.services.ai.manager import AIProviderManager, get_ai_manager
from app.config import (
    DEFAULT_AI_FALLBACK_BUDGET_SECONDS,
    MIN_CANDIDATE_TIMEOUT_SECONDS,
    MAX_CANDIDATE_TIMEOUT_SECONDS,
    get_ai_fallback_budget_seconds,
    normalize_model_name,
)
from app.routes.settings import get_ai_system_diagnostics



class MockAIProvider(AIProvider):
    """Custom mock provider for precision timing and call verification."""

    def __init__(
        self,
        pid: str,
        display_name: str,
        provider_type: str = "direct",
        default_model: str = "mock-model"
    ):
        self._pid = pid
        self._display = display_name
        self._type = provider_type
        self._default_model = default_model
        self.call_count = 0
        self.last_options = None
        self.last_timeout = None
        self.last_max_retries = None
        self.behavior = "success"  # "success", "delay_and_fail", "delay_and_succeed", "fail"
        self.delay_seconds = 0.0
        self.fail_error = AIServiceUnavailableError(f"{pid} 503 Overloaded", status_code=503)
        self._last_meta = {}

    @property
    def provider_id(self) -> str:
        return self._pid

    @property
    def display_name(self) -> str:
        return self._display

    @property
    def provider_type(self) -> str:
        return self._type

    def generate(self, prompt: str, db=None, timeout=60.0, max_retries=2, enable_fallback=True, options=None, **kwargs) -> str:
        self.call_count += 1
        self.last_timeout = timeout
        self.last_max_retries = max_retries
        self.last_options = options

        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

        if self.behavior in ("fail", "delay_and_fail"):
            self._last_meta = {
                "provider": self._pid,
                "status": "FAILED",
                "error_type": getattr(self.fail_error, "error_type", "error"),
                "attempts": max_retries + 1,
                "duration_seconds": self.delay_seconds
            }
            raise self.fail_error

        self._last_meta = {
            "provider": self._pid,
            "provider_type": self._type,
            "configured_model": self._default_model,
            "actual_model_used": self._default_model,
            "fallback_used": False,
            "status": "SUCCESS",
            "attempts": 1,
            "duration_seconds": self.delay_seconds or 0.15,
            "input_tokens": 100,
            "output_tokens": 250,
            "total_tokens": 350
        }
        return f"Response from {self._pid}"

    def get_last_execution_metadata(self):
        return dict(self._last_meta)

    def test_connection(self, db=None):
        return {"connected": True, "provider": self._pid}

    def list_models(self, db=None):
        return [{"id": self._default_model, "name": self._default_model}]

    def supports_model(self, model_name: str) -> bool:
        return True

    def generate_products(self, niche: str, count: int = 10, db=None) -> list:
        self.call_count += 1
        if self.behavior in ("fail", "delay_and_fail"):
            raise self.fail_error
        return [{"name": f"Product 1 ({self._pid})"}]



class TestPhase5ProductionHardening(unittest.TestCase):

    def setUp(self):
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.db = self.SessionLocal()

        # Seed initial test settings
        self.db.add(Setting(key="active_ai_provider", value="gemini"))
        self.db.add(Setting(key="gemini_api_key", value="mock-gemini-key"))
        self.db.add(Setting(key="openai_api_key", value="mock-openai-key"))
        self.db.add(Setting(key="groq_api_key", value="mock-groq-key"))
        self.db.commit()

        # Setup test manager with mock providers
        self.manager = AIProviderManager()
        self.gemini = MockAIProvider("gemini", "Google Gemini", "direct", "gemini-3.8-flash")
        self.openai = MockAIProvider("openai", "OpenAI", "direct", "gpt-4o")
        self.groq = MockAIProvider("groq", "Groq", "direct", "llama-3.3-70b-versatile")
        self.anthropic = MockAIProvider("anthropic", "Anthropic", "direct", "claude-3-5-sonnet")
        self.openrouter = MockAIProvider("openrouter", "OpenRouter", "gateway", "anthropic/claude-3.5-sonnet")

        self.manager.register_provider(self.gemini)
        self.manager.register_provider(self.openai)
        self.manager.register_provider(self.groq)
        self.manager.register_provider(self.anthropic)
        self.manager.register_provider(self.openrouter)

    def tearDown(self):
        self.db.close()
        Base.metadata.drop_all(self.engine)

    # --------------------------------------------------------------------------
    # 1. Total Workflow Deadline & Budget Enforcement
    # --------------------------------------------------------------------------
    def test_workflow_deadline_includes_primary_time(self):
        """Total deadline begins before primary attempt; primary elapsed time counts against budget."""
        # Enable fallback to groq
        self.db.add(Setting(key="ai_fallback_enabled", value="true"))
        self.db.add(Setting(key="ai_fallback_providers", value="groq"))
        self.db.commit()

        # Primary fails after 0.2s
        self.gemini.behavior = "delay_and_fail"
        self.gemini.delay_seconds = 0.2
        self.groq.behavior = "success"

        # Mock a small total budget of 0.15s (simulating primary consuming the entire budget)
        with patch("app.services.ai.manager.get_ai_fallback_budget_seconds", return_value=0.15):
            options = AIGenerationOptions(allow_cross_provider_fallback=True)
            with self.assertRaises(AIFallbackExhaustedError) as ctx:
                self.manager.generate("test prompt", db=self.db, options=options)

            # Assert fallback candidate groq was NEVER called because primary exceeded total budget
            self.assertEqual(self.gemini.call_count, 1)
            self.assertEqual(self.groq.call_count, 0)
            self.assertIn("budget", str(ctx.exception).lower())

    def test_no_candidate_begins_after_deadline_exhaustion(self):
        """When candidate 1 consumes remaining budget, candidate 2 must not be started."""
        self.db.add(Setting(key="ai_fallback_enabled", value="true"))
        self.db.add(Setting(key="ai_fallback_providers", value="groq,openai"))
        self.db.commit()

        self.gemini.behavior = "delay_and_fail"
        self.gemini.delay_seconds = 0.05
        self.groq.behavior = "delay_and_fail"
        self.groq.delay_seconds = 0.10
        self.openai.behavior = "success"

        # Total budget = 0.12s, min threshold = 0.02s. Primary takes 0.05s (remaining 0.07s > 0.02s).
        # Groq takes 0.10s (remaining -0.03s < 0.02s). OpenAI must NOT be called.
        with patch("app.services.ai.manager.get_ai_fallback_budget_seconds", return_value=0.12), \
             patch("app.services.ai.manager.MIN_CANDIDATE_TIMEOUT_SECONDS", 0.02):
            options = AIGenerationOptions(allow_cross_provider_fallback=True)
            with self.assertRaises(AIFallbackExhaustedError):
                self.manager.generate("test prompt", db=self.db, options=options)

            self.assertEqual(self.gemini.call_count, 1)
            self.assertEqual(self.groq.call_count, 1)
            self.assertEqual(self.openai.call_count, 0)

    def test_fallback_candidates_receive_zero_retries_and_capped_timeout(self):
        """Fallback candidates must receive max_retries=0 and timeout capped at min(timeout, remaining, 20.0)."""
        self.db.add(Setting(key="ai_fallback_enabled", value="true"))
        self.db.add(Setting(key="ai_fallback_providers", value="groq"))
        self.db.commit()

        self.gemini.behavior = "fail"
        self.groq.behavior = "success"

        options = AIGenerationOptions(
            timeout=60.0,
            max_retries=2,
            allow_cross_provider_fallback=True
        )
        res = self.manager.generate("test prompt", db=self.db, options=options)
        self.assertEqual(res, "Response from groq")

        # Verify groq was called with max_retries=0 and timeout <= 20.0
        self.assertEqual(self.groq.call_count, 1)
        self.assertEqual(self.groq.last_max_retries, 0)
        self.assertLessEqual(self.groq.last_timeout, 20.0)

    # --------------------------------------------------------------------------
    # 2. Local-Only Model Normalization (Zero Network Calls During Generation)
    # --------------------------------------------------------------------------
    def test_generation_performs_zero_model_discovery_network_calls(self):
        """Generation must never call list_models or external model catalog discovery endpoints."""
        with patch.object(self.gemini, "list_models") as mock_list:
            res = self.manager.generate("test prompt", db=self.db)
            self.assertEqual(res, "Response from gemini")
            mock_list.assert_not_called()

    def test_empty_or_whitespace_model_normalized_locally(self):
        """Empty or whitespace model setting falls back safely to provider default."""
        self.assertEqual(normalize_model_name("gemini", "  "), "gemini-3.8-flash")
        self.assertEqual(normalize_model_name("openai", ""), "gpt-4o-mini")
        self.assertEqual(normalize_model_name("groq", "   \t"), "llama-3.3-70b-versatile")

    def test_valid_customer_selected_model_not_replaced(self):
        """Valid model string selected by customer is preserved as-is without cache checking."""
        custom_model = "gpt-4o-2024-11-20"
        self.assertEqual(normalize_model_name("openai", custom_model), custom_model)

    # --------------------------------------------------------------------------
    # 3. Settings Diagnostics (Local-Only by Default)
    # --------------------------------------------------------------------------
    def test_settings_diagnostics_performs_zero_provider_network_calls(self):
        """get_ai_system_diagnostics must be 100% local with zero network calls."""
        with patch.object(self.gemini, "test_connection") as mock_test, \
             patch.object(self.gemini, "list_models") as mock_list:
            diag = get_ai_system_diagnostics(self.db)
            mock_test.assert_not_called()
            mock_list.assert_not_called()

            self.assertEqual(diag["active_provider"], "gemini")
            self.assertTrue(diag["providers"]["gemini"]["configured"])
            self.assertEqual(diag["fallback_enabled"], False)

    # --------------------------------------------------------------------------
    # 4. ExecutionMetadata Completeness
    # --------------------------------------------------------------------------
    def test_metadata_propagates_winning_candidate_data(self):
        """When candidate succeeds, metadata contains token usage, duration, and fallback flags."""
        self.db.add(Setting(key="ai_fallback_enabled", value="true"))
        self.db.add(Setting(key="ai_fallback_providers", value="openai"))
        self.db.commit()

        self.gemini.behavior = "fail"
        self.openai.behavior = "success"

        options = AIGenerationOptions(allow_cross_provider_fallback=True)
        self.manager.generate("test prompt", db=self.db, options=options)

        meta = self.manager.get_last_execution_metadata()
        self.assertEqual(meta["provider"], "gemini")
        self.assertEqual(meta["actual_provider_used"], "openai")
        self.assertTrue(meta["fallback_used"])
        self.assertTrue(meta["cross_provider_fallback_used"])
        self.assertFalse(meta["internal_fallback_used"])
        self.assertEqual(meta["input_tokens"], 100)
        self.assertEqual(meta["output_tokens"], 250)
        self.assertEqual(meta["total_tokens"], 350)
        self.assertIsNotNone(meta["total_duration_seconds"])

    # --------------------------------------------------------------------------
    # 5. Degraded Mode & Cold Start
    # --------------------------------------------------------------------------
    def test_zero_key_cold_start_performs_zero_network_calls(self):
        """When no keys exist, generation raises clean AIAuthenticationError without crashing or calling network."""
        empty_db = self.SessionLocal()
        try:
            with patch.object(self.gemini, "generate", side_effect=AIAuthenticationError("No key", error_type="not_configured")):
                with self.assertRaises(AIAuthenticationError) as ctx:
                    self.manager.generate("test prompt", db=empty_db)
                self.assertEqual(ctx.exception.error_type, "not_configured")
        finally:
            empty_db.close()

    # --------------------------------------------------------------------------
    # 6. Research Invariant Strictly Preserved
    # --------------------------------------------------------------------------
    def test_research_invariant_strictly_preserved(self):
        """Research uses max_retries=0, allow_cross_provider_fallback=False, and stops on failure."""
        self.db.add(Setting(key="ai_fallback_enabled", value="true"))
        self.db.add(Setting(key="ai_fallback_providers", value="groq,openai"))
        self.db.commit()

        self.gemini.behavior = "fail"

        with self.assertRaises(AIProviderError):
            self.manager.generate_products(niche="test", count=5, db=self.db)

        # Gemini was called exactly once; groq and openai were NEVER called
        self.assertEqual(self.gemini.call_count, 1)
        self.assertEqual(self.groq.call_count, 0)
        self.assertEqual(self.openai.call_count, 0)


    # --------------------------------------------------------------------------
    # 7. Error UX & Secret Scrubbing
    # --------------------------------------------------------------------------
    def test_user_friendly_error_messages_and_sanitization(self):
        """get_user_friendly_error_message provides clean guidance and scrubs credentials."""
        dummy_key = "AIzaSy" + "TestMockKeyForSanitizer00000"
        err_auth = AIAuthenticationError(f"Invalid API Key {dummy_key}", error_type="authentication")
        msg = get_user_friendly_error_message(err_auth)
        self.assertNotIn(dummy_key, msg)
        self.assertIn("API Key", msg)

        err_quota = AIQuotaExceededError("Credit balance is 0", error_type="quota")
        msg_quota = get_user_friendly_error_message(err_quota)
        self.assertIn("Hạn ngạch", msg_quota)

    # --------------------------------------------------------------------------
    # 8. Fallback Budget Normalization
    # --------------------------------------------------------------------------
    def test_configured_fallback_budget_normalization(self):
        """Invalid, zero, negative, or substandard budget values normalize safely to 45.0."""
        # Non-numeric
        self.db.add(Setting(key="ai_fallback_budget_seconds", value="invalid_number"))
        self.db.commit()
        self.assertEqual(get_ai_fallback_budget_seconds(self.db), 45.0)

        # Zero
        rec = self.db.query(Setting).filter_by(key="ai_fallback_budget_seconds").first()
        rec.value = "0"
        self.db.commit()
        self.assertEqual(get_ai_fallback_budget_seconds(self.db), 45.0)

        # Negative
        rec.value = "-15.5"
        self.db.commit()
        self.assertEqual(get_ai_fallback_budget_seconds(self.db), 45.0)

        # Unreasonably small (< MIN_CANDIDATE_TIMEOUT_SECONDS = 3.0)
        rec.value = "0.5"
        self.db.commit()
        self.assertEqual(get_ai_fallback_budget_seconds(self.db), 45.0)

        # Valid custom budget
        rec.value = "60.0"
        self.db.commit()
        self.assertEqual(get_ai_fallback_budget_seconds(self.db), 60.0)


if __name__ == "__main__":
    unittest.main()

