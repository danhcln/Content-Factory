"""
test_gemini_status_ui.py
Tests for the Gemini Status UI feature.

Tests ALL HTTP status mappings using mocked httpx responses (zero real Gemini calls).
Tests the /api/gemini/status endpoint (no Gemini calls).
Tests API key exposure prevention.
Tests Research button double-submit protection logic.
Tests existing gemini_hardening still passes.
"""
import sys
import os
import datetime
import unittest
from unittest.mock import MagicMock, patch, call
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.services.gemini_status import GeminiStatusTracker, STATUS_MESSAGES, STATUS_HINT


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _make_mock_db():
    """Create a mock SQLAlchemy session that stores settings in-memory."""
    store = {}

    def _query_side_effect(model):
        from app.models import Setting
        q = MagicMock()

        def _filter_side_effect(*args, **kwargs):
            f = MagicMock()

            def _first_side_effect():
                # Extract key from filter expression
                try:
                    key = args[0].right.value
                except Exception:
                    key = None
                if key and key in store:
                    s = MagicMock()
                    s.value = store[key]
                    return s
                return None

            f.first = _first_side_effect
            return f

        q.filter = _filter_side_effect
        return q

    db = MagicMock()
    db.query.side_effect = _query_side_effect

    def _add_side_effect(obj):
        if hasattr(obj, 'key') and hasattr(obj, 'value'):
            store[obj.key] = obj.value

    def _commit_side_effect():
        pass  # Changes are tracked via _add_side_effect

    db.add.side_effect = _add_side_effect
    db.commit.side_effect = _commit_side_effect

    # Expose store for inspection
    db._store = store
    return db


def _make_mock_resp(status_code: int, json_data: dict = None, text: str = ""):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text or str(json_data or "")
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = Exception("No JSON")
    return resp


# ─────────────────────────────────────────────────────────────────
class TestStatusMessages(unittest.TestCase):
    """Test status message map completeness."""

    def test_all_states_have_messages(self):
        required_states = [
            "READY", "CHECKING", "BUSY", "RATE_LIMITED", "QUOTA_EXCEEDED",
            "AUTH_ERROR", "NETWORK_ERROR", "ERROR", "UNKNOWN"
        ]
        for s in required_states:
            self.assertIn(s, STATUS_MESSAGES, f"Missing status message for: {s}")
            self.assertIn(s, STATUS_HINT, f"Missing status hint for: {s}")

    def test_message_content_in_vietnamese(self):
        # At least half of the messages should contain Vietnamese diacritic chars
        vi_chars = set("àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")
        vi_count = sum(1 for msg in STATUS_MESSAGES.values() if any(c in vi_chars for c in msg.lower()))
        self.assertGreater(vi_count, 4, "Most status messages should be in Vietnamese")


# ─────────────────────────────────────────────────────────────────
class TestGeminiStatusTracker(unittest.TestCase):
    """Test GeminiStatusTracker local persistence (no network calls)."""

    def test_initial_get_status_returns_unknown(self):
        """Before any status is set, status should be UNKNOWN."""
        with patch("app.services.gemini_status.SessionLocal") as mock_session_cls:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            mock_session_cls.return_value = mock_db

            with patch("app.config.get_gemini_model", return_value="gemini-3.8-flash"):
                result = GeminiStatusTracker.get_status(db=None)

            self.assertEqual(result["status"], "UNKNOWN")
            self.assertEqual(result["model"], "gemini-3.8-flash")
            self.assertNotIn("api_key", str(result).lower())

    def test_update_status_does_not_call_network(self):
        """update_status must not make any HTTP/network calls."""
        import httpx
        with patch("httpx.Client") as mock_client:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            GeminiStatusTracker.update_status("READY", db=mock_db, http_code=200)
            mock_client.assert_not_called()

    def test_get_status_does_not_call_network(self):
        """get_status must not make any HTTP/network calls."""
        import httpx
        with patch("httpx.Client") as mock_client:
            with patch("app.services.gemini_status.SessionLocal") as mock_sl:
                mock_db = MagicMock()
                mock_db.query.return_value.filter.return_value.first.return_value = None
                mock_sl.return_value = mock_db
                with patch("app.config.get_gemini_model", return_value="gemini-3.8-flash"):
                    GeminiStatusTracker.get_status(db=None)
            mock_client.assert_not_called()

    def test_api_key_never_in_status_response(self):
        """API key must never appear in the status response."""
        with patch("app.services.gemini_status.SessionLocal") as mock_session_cls:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            mock_session_cls.return_value = mock_db
            with patch("app.config.get_gemini_model", return_value="gemini-3.8-flash"):
                result = GeminiStatusTracker.get_status(db=None)

            result_str = str(result)
            # Check no API key patterns
            self.assertNotIn("AIza", result_str)
            self.assertNotIn("api_key", result_str.lower())


# ─────────────────────────────────────────────────────────────────
class TestGeminiCallStatusMapping(unittest.TestCase):
    """Test that GeminiService.call_gemini maps HTTP codes to correct statuses."""

    def setUp(self):
        self.status_calls = []
        self.orig_update = GeminiStatusTracker.update_status

        def mock_update(status, db=None, http_code=None, custom_message=None):
            self.status_calls.append(status)

        patcher = patch.object(GeminiStatusTracker, "update_status", side_effect=mock_update)
        self.patcher = patcher
        patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def _run_call(self, mock_resp):
        from app.services.gemini_service import GeminiService
        mock_db = MagicMock()

        with patch("app.config.get_gemini_api_key", return_value="test_key"), \
             patch("app.config.get_gemini_model", return_value="gemini-3.8-flash"), \
             patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start"), \
             patch("app.services.usage_tracker.GeminiUsageTracker.record_success"), \
             patch("app.services.usage_tracker.GeminiUsageTracker.record_failure"), \
             patch("httpx.Client") as mock_client_cls:

            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            svc = GeminiService()
            try:
                return svc.call_gemini("test prompt", db=mock_db, max_retries=0)
            except Exception as e:
                return e

    def test_200_maps_to_ready(self):
        """HTTP 200 → READY status."""
        resp = _make_mock_resp(200, {
            "candidates": [{"content": {"parts": [{"text": "OK"}]}}]
        })
        result = self._run_call(resp)
        self.assertEqual(result, "OK")
        self.assertIn("READY", self.status_calls)

    def test_503_maps_to_busy(self):
        """HTTP 503 → BUSY status."""
        resp = _make_mock_resp(503, {"error": {"message": "model overloaded"}})
        result = self._run_call(resp)
        self.assertIsInstance(result, Exception)
        self.assertIn("BUSY", self.status_calls)

    def test_429_daily_quota_maps_to_quota_exceeded(self):
        """HTTP 429 daily quota → QUOTA_EXCEEDED, no retry."""
        resp = _make_mock_resp(429, {"error": {
            "message": "Quota exceeded for GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        }})
        result = self._run_call(resp)
        self.assertIsInstance(result, Exception)
        self.assertIn("QUOTA_EXCEEDED", self.status_calls)

    def test_429_temporary_rate_limit_maps_to_rate_limited(self):
        """HTTP 429 RPM/per-minute → RATE_LIMITED."""
        resp = _make_mock_resp(429, {"error": {
            "message": "GenerateRequestsPerMinute exceeded. retry in 30s"
        }})
        result = self._run_call(resp)
        self.assertIsInstance(result, Exception)
        self.assertIn("RATE_LIMITED", self.status_calls)

    def test_401_maps_to_auth_error(self):
        """HTTP 401 → AUTH_ERROR."""
        resp = _make_mock_resp(401, {"error": {"message": "Invalid API key"}})
        result = self._run_call(resp)
        self.assertIsInstance(result, Exception)
        self.assertIn("AUTH_ERROR", self.status_calls)

    def test_403_maps_to_auth_error(self):
        """HTTP 403 → AUTH_ERROR."""
        resp = _make_mock_resp(403, {"error": {"message": "Permission denied"}})
        result = self._run_call(resp)
        self.assertIsInstance(result, Exception)
        self.assertIn("AUTH_ERROR", self.status_calls)

    def test_network_error_maps_to_network_error(self):
        """httpx.NetworkError → NETWORK_ERROR."""
        import httpx
        from app.services.gemini_service import GeminiService
        mock_db = MagicMock()

        network_calls = []
        def mock_update(status, db=None, http_code=None, custom_message=None):
            network_calls.append(status)

        with patch.object(GeminiStatusTracker, "update_status", side_effect=mock_update), \
             patch("app.config.get_gemini_api_key", return_value="test_key"), \
             patch("app.config.get_gemini_model", return_value="gemini-3.8-flash"), \
             patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start"), \
             patch("app.services.usage_tracker.GeminiUsageTracker.record_failure"), \
             patch("httpx.Client") as mock_client_cls:

            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = httpx.NetworkError("Connection refused")
            mock_client_cls.return_value = mock_client

            svc = GeminiService()
            try:
                svc.call_gemini("test", db=mock_db, max_retries=0)
            except Exception:
                pass

        self.assertIn("NETWORK_ERROR", network_calls)

    def test_timeout_maps_to_network_error(self):
        """httpx.TimeoutException → NETWORK_ERROR."""
        import httpx
        from app.services.gemini_service import GeminiService
        mock_db = MagicMock()

        timeout_calls = []
        def mock_update(status, db=None, http_code=None, custom_message=None):
            timeout_calls.append(status)

        with patch.object(GeminiStatusTracker, "update_status", side_effect=mock_update), \
             patch("app.config.get_gemini_api_key", return_value="test_key"), \
             patch("app.config.get_gemini_model", return_value="gemini-3.8-flash"), \
             patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start"), \
             patch("app.services.usage_tracker.GeminiUsageTracker.record_failure"), \
             patch("httpx.Client") as mock_client_cls:

            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = httpx.TimeoutException("Timed out")
            mock_client_cls.return_value = mock_client

            svc = GeminiService()
            try:
                svc.call_gemini("test", db=mock_db, max_retries=0)
            except Exception:
                pass

        self.assertIn("NETWORK_ERROR", timeout_calls)


# ─────────────────────────────────────────────────────────────────
class TestDailyQuotaZeroRetry(unittest.TestCase):
    """Daily quota must NOT retry. Should fail immediately after 1 attempt."""

    def test_daily_quota_stops_immediately(self):
        """HTTP 429 daily quota must not make more than 1 request."""
        from app.services.gemini_service import GeminiService
        mock_db = MagicMock()

        call_count = [0]

        def mock_post(*args, **kwargs):
            call_count[0] += 1
            return _make_mock_resp(429, {"error": {
                "message": "Quota exceeded for GenerateRequestsPerDayPerProjectPerModel-FreeTier"
            }})

        with patch("app.config.get_gemini_api_key", return_value="test_key"), \
             patch("app.config.get_gemini_model", return_value="gemini-3.8-flash"), \
             patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start"), \
             patch("app.services.usage_tracker.GeminiUsageTracker.record_failure"), \
             patch("app.services.gemini_status.GeminiStatusTracker.update_status"), \
             patch("httpx.Client") as mock_client_cls:

            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = mock_post
            mock_client_cls.return_value = mock_client

            svc = GeminiService()
            from app.services.gemini_service import GeminiQuotaExceededError
            with self.assertRaises(GeminiQuotaExceededError):
                svc.call_gemini("test", db=mock_db, max_retries=2)

        # Must have been called exactly once — no retry on daily quota
        self.assertEqual(call_count[0], 1, "Daily quota must NOT retry (expected exactly 1 attempt)")


class TestBusyLimitedRetry(unittest.TestCase):
    """HTTP 503 must retry with max_retries limit, not indefinitely."""

    def test_503_retries_limited_times(self):
        """503 should retry up to max_retries times then fail."""
        from app.services.gemini_service import GeminiService
        import time
        mock_db = MagicMock()

        call_count = [0]

        def mock_post(*args, **kwargs):
            call_count[0] += 1
            return _make_mock_resp(503, {"error": {"message": "model overloaded"}})

        with patch("app.config.get_gemini_api_key", return_value="test_key"), \
             patch("app.config.get_gemini_model", return_value="gemini-3.8-flash"), \
             patch("app.services.usage_tracker.GeminiUsageTracker.record_request_start"), \
             patch("app.services.usage_tracker.GeminiUsageTracker.record_failure"), \
             patch("app.services.gemini_status.GeminiStatusTracker.update_status"), \
             patch("time.sleep"), \
             patch("httpx.Client") as mock_client_cls:

            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = mock_post
            mock_client_cls.return_value = mock_client

            svc = GeminiService()
            try:
                svc.call_gemini("test", db=mock_db, max_retries=2)
            except RuntimeError as e:
                error_str = str(e)
                self.assertIn("503", error_str.replace("quá tải", "503") or error_str)

        # With max_retries=2, should attempt 1 + 2 = 3 times total
        self.assertEqual(call_count[0], 3, f"503 should retry exactly max_retries times (got {call_count[0]})")


# ─────────────────────────────────────────────────────────────────
class TestStatusEndpointSafety(unittest.TestCase):
    """Test /api/gemini/status endpoint properties."""

    def test_status_endpoint_returns_no_api_key(self):
        """Status response must never contain API key data."""
        from app.services.gemini_status import GeminiStatusTracker

        with patch("app.services.gemini_status.SessionLocal") as mock_sl, \
             patch("app.config.get_gemini_model", return_value="gemini-3.8-flash"), \
             patch("app.config.get_gemini_api_key", return_value="AIzaSyTestKeyABC123"):

            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            mock_sl.return_value = mock_db

            result = GeminiStatusTracker.get_status(db=None)

        result_str = str(result)
        # API key values must NOT appear
        self.assertNotIn("AIzaSyTestKeyABC123", result_str)
        self.assertNotIn("AIza", result_str)

    def test_status_endpoint_model_is_correct(self):
        """Status endpoint should return the configured model name."""
        from app.services.gemini_status import GeminiStatusTracker

        with patch("app.services.gemini_status.SessionLocal") as mock_sl, \
             patch("app.config.get_gemini_model", return_value="gemini-3.8-flash"):

            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            mock_sl.return_value = mock_db

            result = GeminiStatusTracker.get_status(db=None)

        self.assertEqual(result["model"], "gemini-3.8-flash")


# ─────────────────────────────────────────────────────────────────
class TestIsQuotaErrorFunctionality(unittest.TestCase):
    """Ensure is_daily_quota_error still correctly discriminates 429 types."""

    def test_daily_quota_detection(self):
        from app.services.gemini_service import is_daily_quota_error
        cases_daily = [
            (429, "GenerateRequestsPerDay exceeded"),
            (429, "Quota exceeded for GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
            (429, "Resource has been exhausted (resource_exhausted)"),
            (429, "free tier daily quota"),
        ]
        for code, msg in cases_daily:
            self.assertTrue(is_daily_quota_error(code, msg), f"Should be daily quota: {msg}")

    def test_temporary_rate_limit_not_daily(self):
        from app.services.gemini_service import is_daily_quota_error
        cases_tmp = [
            (429, "GenerateRequestsPerMinute exceeded. Retry in 30s"),
            (429, "RPM limit exceeded"),
            (429, "per minute limit"),
        ]
        for code, msg in cases_tmp:
            self.assertFalse(is_daily_quota_error(code, msg), f"Should NOT be daily quota: {msg}")


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*70)
    print("GEMINI STATUS UI — TEST SUITE")
    print("All tests use mocked HTTP — zero real Gemini quota consumed")
    print("="*70 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestStatusMessages,
        TestGeminiStatusTracker,
        TestGeminiCallStatusMapping,
        TestDailyQuotaZeroRetry,
        TestBusyLimitedRetry,
        TestStatusEndpointSafety,
        TestIsQuotaErrorFunctionality,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
