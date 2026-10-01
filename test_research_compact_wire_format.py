"""
Unit and Integration Tests for Compact Wire-Format Optimization in Research.
Tests:
A. Compact Prompt: Requests nv, nc, dk, ca, h and exact count.
B. Local Hydration: Compact keys mapped to canonical fields.
C. Backward Compatibility: Canonical keys remain accepted.
D. Collision Precedence: Canonical keys override compact keys when both present.
E. Exact Count Enforcement: 50 succeeds, 49/51/52 fail without second AI call.
F. Full 50-Product Persistence: 50 compact products normalized & persisted via 1 POST.
G. Wire Size: Compact wire representation is materially smaller (chars & bytes).
H. HTTP 524 Handling: Upstream Cloudflare 524 stops with Retry-After and Ray ID, no retry.
I. Provider Diagnostic: Telemetry logs Provider=mwapi instead of Provider=unknown.

Zero Real External API Calls.
"""
import unittest
from unittest.mock import patch, MagicMock, ANY
import json
import logging
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.testclient import TestClient

from app.database import Base, get_db
from app.main import app
from app.models import Product, Setting
from app.services.ai.base import (
    AIProvider,
    AIGenerationOptions,
    AIInvalidResponseError,
    AIServiceUnavailableError,
    hydrate_research_product,
    get_research_max_output_tokens,
    get_research_httpx_timeout,
)
from app.services.ai.providers.mwapi import MWAPIProvider
from app.services.ai.providers.common import classify_http_error

from sqlalchemy.pool import StaticPool

# In-memory SQLite for test isolation
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


class DummyTestProvider(AIProvider):
    """Minimal test provider subclass to test generate_products domain helper."""
    provider_id = "test_provider"
    provider_name = "Test Provider"
    provider_type = "cloud"

    def __init__(self, mock_generate_return: str = ""):
        self.mock_generate_return = mock_generate_return
        self.generate_calls = []

    @property
    def display_name(self) -> str:
        return "Test Provider"

    def generate(self, prompt: str, db=None, timeout=60.0, max_retries=0, enable_fallback=False, options=None, **kwargs) -> str:
        self.generate_calls.append({
            "prompt": prompt,
            "timeout": timeout,
            "max_retries": max_retries,
            "enable_fallback": enable_fallback,
            "options": options,
            "kwargs": kwargs
        })
        return self.mock_generate_return

    def get_supported_models(self, db=None):
        return []

    def get_default_model(self, db=None):
        return "test-model"

    def normalize_model_name(self, model: str) -> str:
        return model

    def validate_credentials(self, db=None):
        return {"configured": True, "valid": True}

    def list_models(self, db=None):
        return []

    def supports_model(self, model: str, db=None) -> bool:
        return True

    def test_connection(self, db=None):
        return {"success": True, "message": "OK"}

    def get_last_execution_metadata(self):
        return {}


class TestResearchCompactWireFormat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)

    @classmethod
    def tearDownClass(cls):
        Base.metadata.drop_all(bind=engine)

    def setUp(self):
        self.db = TestingSessionLocal()
        self.db.query(Product).delete()
        self.db.query(Setting).delete()
        self.db.commit()
        self.client = TestClient(app)

    def tearDown(self):
        self.db.close()

    # --------------------------------------------------------------------------
    # A. COMPACT PROMPT
    # --------------------------------------------------------------------------
    def test_compact_prompt_structure_and_exact_count(self):
        """Verify prompt requests nv, nc, dk, ca, h, specifies exact count, and forbids fences."""
        mock_50 = json.dumps([
            {"nv": f"SP {i}", "nc": f"CP {i}", "dk": "DK", "ca": "CA", "h": "H"}
            for i in range(1, 51)
        ])
        dummy = DummyTestProvider(mock_generate_return=mock_50)
        dummy.generate_products(niche="đồ gia dụng", count=50, db=self.db)

        self.assertEqual(len(dummy.generate_calls), 1)
        prompt = dummy.generate_calls[0]["prompt"]

        # Check compact wire keys are requested
        self.assertIn('"nv"', prompt)
        self.assertIn('"nc"', prompt)
        self.assertIn('"dk"', prompt)
        self.assertIn('"ca"', prompt)
        self.assertIn('"h"', prompt)

        # Check exact count requirement
        self.assertIn("50", prompt)
        self.assertIn("exactly 50", prompt.lower())

        # Check no fences instruction
        self.assertIn("No markdown fences", prompt)
        self.assertIn("raw JSON array", prompt)

        # Confirm options contain 6000 token ceiling and explicit HTTPX timeout
        opts = dummy.generate_calls[0]["options"]
        self.assertIsNotNone(opts)
        self.assertEqual(opts.max_output_tokens, 6000)
        self.assertEqual(opts.max_retries, 0)
        self.assertFalse(opts.enable_fallback)

    # --------------------------------------------------------------------------
    # B. HYDRATION (Compact -> Canonical)
    # --------------------------------------------------------------------------
    def test_hydration_compact_keys_to_canonical(self):
        """Verify hydrate_research_product maps nv, nc, dk, ca, h to canonical schema."""
        compact_item = {
            "nv": "Nồi chiên không dầu 5L",
            "nc": "空气炸锅5L",
            "dk": "空气炸锅测评 居家好物 懒人料理神器",
            "ca": "Nấu ăn không dầu mỡ tiện lợi cho gia đình",
            "h": "Đừng dùng dầu ăn chiên rán nữa nếu bạn có món này!"
        }
        hydrated = hydrate_research_product(compact_item)
        self.assertIsNotNone(hydrated)
        self.assertEqual(hydrated["name_vietnamese"], "Nồi chiên không dầu 5L")
        self.assertEqual(hydrated["name_chinese"], "空气炸锅5L")
        self.assertEqual(hydrated["douyin_keywords"], "空气炸锅测评 居家好物 懒人料理神器")
        self.assertEqual(hydrated["content_angle"], "Nấu ăn không dầu mỡ tiện lợi cho gia đình")
        self.assertEqual(hydrated["hook"], "Đừng dùng dầu ăn chiên rán nữa nếu bạn có món này!")

    # --------------------------------------------------------------------------
    # C. BACKWARD COMPATIBILITY (Canonical Keys)
    # --------------------------------------------------------------------------
    def test_backward_compatibility_canonical_keys(self):
        """Verify hydrate_research_product accepts full/canonical keys seamlessly."""
        full_item = {
            "name_vietnamese": "Máy hút bụi UV",
            "name_chinese": "除螨仪",
            "douyin_keywords": "除螨仪测评 好物推荐",
            "content_angle": "Diệt khuẩn giường nệm sạch sẽ",
            "hook": "Giường bạn có thể bẩn hơn bạn nghĩ!"
        }
        hydrated = hydrate_research_product(full_item)
        self.assertIsNotNone(hydrated)
        self.assertEqual(hydrated["name_vietnamese"], "Máy hút bụi UV")
        self.assertEqual(hydrated["name_chinese"], "除螨仪")
        self.assertEqual(hydrated["douyin_keywords"], "除螨仪测评 好物推荐")
        self.assertEqual(hydrated["content_angle"], "Diệt khuẩn giường nệm sạch sẽ")
        self.assertEqual(hydrated["hook"], "Giường bạn có thể bẩn hơn bạn nghĩ!")

    # --------------------------------------------------------------------------
    # D. COLLISION PRECEDENCE (Canonical Wins)
    # --------------------------------------------------------------------------
    def test_collision_precedence_canonical_wins(self):
        """Verify that when both canonical and compact keys exist, canonical wins."""
        collision_item = {
            "name_vietnamese": "CANONICAL_VN",
            "nv": "COMPACT_VN",
            "name_chinese": "CANONICAL_CN",
            "nc": "COMPACT_CN",
            "douyin_keywords": "CANONICAL_DK",
            "dk": "COMPACT_DK",
            "content_angle": "CANONICAL_CA",
            "ca": "COMPACT_CA",
            "hook": "CANONICAL_H",
            "h": "COMPACT_H"
        }
        hydrated = hydrate_research_product(collision_item)
        self.assertIsNotNone(hydrated)
        self.assertEqual(hydrated["name_vietnamese"], "CANONICAL_VN")
        self.assertEqual(hydrated["name_chinese"], "CANONICAL_CN")
        self.assertEqual(hydrated["douyin_keywords"], "CANONICAL_DK")
        self.assertEqual(hydrated["content_angle"], "CANONICAL_CA")
        self.assertEqual(hydrated["hook"], "CANONICAL_H")

    # --------------------------------------------------------------------------
    # E. COUNT RESILIENCE & OVER-GENERATION TRIMMING
    # --------------------------------------------------------------------------
    def test_count_resilience_and_overgeneration_trimming(self):
        """Verify count=50 fails on <50 (48, 49) and deterministically trims >=50 (50, 51, 52) to 50."""
        def make_items(n):
            return json.dumps([
                {"nv": f"Sản phẩm {i}", "nc": f"产品 {i}", "dk": "关键词", "ca": "Góc quay", "h": "Hook"}
                for i in range(1, n + 1)
            ])

        # 48 items -> FAIL
        dummy_48 = DummyTestProvider(mock_generate_return=make_items(48))
        with self.assertRaises(AIInvalidResponseError) as ctx_48:
            dummy_48.generate_products(niche="test", count=50, db=self.db)
        self.assertIn("48/50", str(ctx_48.exception))
        self.assertEqual(len(dummy_48.generate_calls), 1)  # Zero retry

        # 49 items -> FAIL
        dummy_49 = DummyTestProvider(mock_generate_return=make_items(49))
        with self.assertRaises(AIInvalidResponseError) as ctx_49:
            dummy_49.generate_products(niche="test", count=50, db=self.db)
        self.assertIn("49/50", str(ctx_49.exception))
        self.assertEqual(len(dummy_49.generate_calls), 1)  # Zero retry

        # 50 items -> SUCCESS (exactly 50)
        dummy_50 = DummyTestProvider(mock_generate_return=make_items(50))
        results_50 = dummy_50.generate_products(niche="test", count=50, db=self.db)
        self.assertEqual(len(results_50), 50)
        self.assertEqual(len(dummy_50.generate_calls), 1)
        self.assertEqual(results_50[0]["name_vietnamese"], "Sản phẩm 1")
        self.assertEqual(results_50[49]["name_vietnamese"], "Sản phẩm 50")

        # 51 items -> SUCCESS (locally trimmed to 50, item 51 dropped)
        dummy_51 = DummyTestProvider(mock_generate_return=make_items(51))
        results_51 = dummy_51.generate_products(niche="test", count=50, db=self.db)
        self.assertEqual(len(results_51), 50)
        self.assertEqual(len(dummy_51.generate_calls), 1)
        self.assertEqual(results_51[0]["name_vietnamese"], "Sản phẩm 1")
        self.assertEqual(results_51[49]["name_vietnamese"], "Sản phẩm 50")

        # 52 items -> SUCCESS (locally trimmed to 50, items 51 & 52 dropped)
        dummy_52 = DummyTestProvider(mock_generate_return=make_items(52))
        results_52 = dummy_52.generate_products(niche="test", count=50, db=self.db)
        self.assertEqual(len(results_52), 50)
        self.assertEqual(len(dummy_52.generate_calls), 1)
        self.assertEqual(results_52[0]["name_vietnamese"], "Sản phẩm 1")
        self.assertEqual(results_52[49]["name_vietnamese"], "Sản phẩm 50")

    def test_malformed_candidates_validated_before_trimming(self):
        """Verify invalid/malformed candidate objects are discarded before trimming occurs."""
        # 50 valid items + 3 malformed items interspersed (total 53 in JSON array)
        mixed_53 = []
        for i in range(1, 51):
            mixed_53.append({"nv": f"Hợp lệ {i}", "nc": f"产品 {i}", "dk": "DK", "ca": "CA", "h": "H"})
            if i in (10, 25, 40):
                # Inject malformed candidates: missing name, non-dict, empty dict
                mixed_53.append({"nc": "Không có tên", "dk": "DK"})  # Missing nv/name_vietnamese
        mixed_53.append("Not a dictionary string")
        mixed_53.append({})

        dummy = DummyTestProvider(mock_generate_return=json.dumps(mixed_53, ensure_ascii=False))
        results = dummy.generate_products(niche="test", count=50, db=self.db)
        self.assertEqual(len(results), 50)
        self.assertEqual(results[0]["name_vietnamese"], "Hợp lệ 1")
        self.assertEqual(results[49]["name_vietnamese"], "Hợp lệ 50")

        # 49 valid items + 3 malformed items (total 52 in JSON array) -> Insufficient valid items -> FAIL
        mixed_52 = [
            {"nv": f"Hợp lệ {i}", "nc": f"产品 {i}", "dk": "DK", "ca": "CA", "h": "H"}
            for i in range(1, 50)
        ]
        mixed_52.extend([
            {"nc": "Thiếu tên VN"},
            "Chuỗi rác",
            {}
        ])
        dummy_under = DummyTestProvider(mock_generate_return=json.dumps(mixed_52, ensure_ascii=False))
        with self.assertRaises(AIInvalidResponseError) as ctx_under:
            dummy_under.generate_products(niche="test", count=50, db=self.db)
        self.assertIn("49/50", str(ctx_under.exception))

    # --------------------------------------------------------------------------
    # F. FULL 50-PRODUCT PERSISTENCE (1 POST, Canonical In DB)
    # --------------------------------------------------------------------------
    @patch("app.services.ai.providers.mwapi.httpx.Client")
    def test_full_50_product_response_and_persistence(self, mock_client_cls):
        """Mock exactly 50 compact objects, verify 1 POST, exactly 50 persisted with canonical fields."""
        # Set active provider to mwapi
        self.db.add(Setting(key="active_ai_provider", value="mwapi"))
        self.db.add(Setting(key="mwapi_api_key", value="test-dummy-key-00000000"))
        self.db.add(Setting(key="mwapi_model", value="claude-sonnet-4-6"))
        self.db.commit()

        compact_50 = [
            {
                "nv": f"Sản phẩm gia dụng thông minh {i}",
                "nc": f"智能家居好物{i}",
                "dk": f"好物推荐 居家神器 测评{i}",
                "ca": f"Giải pháp tiện lợi số {i} cho căn nhà hiện đại",
                "h": f"Bạn sẽ tiếc nếu không biết món đồ số {i} này sớm hơn!"
            }
            for i in range(1, 51)
        ]
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(compact_50, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 400, "completion_tokens": 3800, "total_tokens": 4200},
            "model": "claude-sonnet-4-6"
        }
        mock_resp.headers = {}
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        response = self.client.post(
            "/research",
            data={"niche": "đồ gia dụng mới", "product_count": "50", "fresh": "1"},
            follow_redirects=False
        )
        self.assertEqual(response.status_code, 303)

        # Invariant: exactly 1 POST
        self.assertEqual(mock_client.post.call_count, 1)

        # Database verification: exactly 50 products persisted
        prods = self.db.query(Product).filter(Product.niche == "đồ gia dụng mới").all()
        self.assertEqual(len(prods), 50)

        # Verify all canonical fields are present in database
        first_prod = prods[0]
        self.assertEqual(first_prod.name_vietnamese, "Sản phẩm gia dụng thông minh 1")
        self.assertEqual(first_prod.name_chinese, "智能家居好物1")
        self.assertEqual(first_prod.douyin_keywords, "好物推荐 居家神器 测评1")
        self.assertEqual(first_prod.content_angle, "Giải pháp tiện lợi số 1 cho căn nhà hiện đại")
        self.assertEqual(first_prod.hook, "Bạn sẽ tiếc nếu không biết món đồ số 1 này sớm hơn!")

    @patch("app.services.ai.providers.mwapi.httpx.Client")
    def test_overgeneration_51_persistence_trimming(self, mock_client_cls):
        """When AI returns 51 products for count=50, exactly 50 are persisted and product 51 is dropped."""
        self.db.add(Setting(key="active_ai_provider", value="mwapi"))
        self.db.add(Setting(key="mwapi_api_key", value="test-dummy-key-00000000"))
        self.db.add(Setting(key="mwapi_model", value="claude-sonnet-4-6"))
        self.db.commit()

        compact_51 = [
            {
                "nv": f"Sản phẩm thừa 51_{i}",
                "nc": f"产品 {i}",
                "dk": f"好物 {i}",
                "ca": f"Góc quay {i}",
                "h": f"Hook {i}"
            }
            for i in range(1, 52)
        ]
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(compact_51, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 400, "completion_tokens": 3850, "total_tokens": 4250},
            "model": "claude-sonnet-4-6"
        }
        mock_resp.headers = {}
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        response = self.client.post(
            "/research",
            data={"niche": "niche thừa 51", "product_count": "50", "fresh": "1"},
            follow_redirects=False
        )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(mock_client.post.call_count, 1)

        prods = self.db.query(Product).filter(Product.niche == "niche thừa 51").order_by(Product.id.asc()).all()
        self.assertEqual(len(prods), 50, "Must persist exactly 50 products when 51 were returned")
        self.assertEqual(prods[0].name_vietnamese, "Sản phẩm thừa 51_1")
        self.assertEqual(prods[49].name_vietnamese, "Sản phẩm thừa 51_50")

        # Verify product 51 is not in DB
        prod_51 = self.db.query(Product).filter(Product.name_vietnamese == "Sản phẩm thừa 51_51").first()
        self.assertIsNone(prod_51)

    @patch("app.services.ai.providers.mwapi.httpx.Client")
    def test_overgeneration_52_persistence_trimming(self, mock_client_cls):
        """When AI returns 52 products for count=50, exactly 50 are persisted and products 51 & 52 are dropped."""
        self.db.add(Setting(key="active_ai_provider", value="mwapi"))
        self.db.add(Setting(key="mwapi_api_key", value="test-dummy-key-00000000"))
        self.db.add(Setting(key="mwapi_model", value="claude-sonnet-4-6"))
        self.db.commit()

        compact_52 = [
            {
                "nv": f"Sản phẩm thừa 52_{i}",
                "nc": f"产品 {i}",
                "dk": f"好物 {i}",
                "ca": f"Góc quay {i}",
                "h": f"Hook {i}"
            }
            for i in range(1, 53)
        ]
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(compact_52, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 400, "completion_tokens": 3900, "total_tokens": 4300},
            "model": "claude-sonnet-4-6"
        }
        mock_resp.headers = {}
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        response = self.client.post(
            "/research",
            data={"niche": "niche thừa 52", "product_count": "50", "fresh": "1"},
            follow_redirects=False
        )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(mock_client.post.call_count, 1)

        prods = self.db.query(Product).filter(Product.niche == "niche thừa 52").order_by(Product.id.asc()).all()
        self.assertEqual(len(prods), 50, "Must persist exactly 50 products when 52 were returned")
        self.assertEqual(prods[0].name_vietnamese, "Sản phẩm thừa 52_1")
        self.assertEqual(prods[49].name_vietnamese, "Sản phẩm thừa 52_50")

        # Verify product 51 and 52 are not in DB
        prod_51 = self.db.query(Product).filter(Product.name_vietnamese == "Sản phẩm thừa 52_51").first()
        prod_52 = self.db.query(Product).filter(Product.name_vietnamese == "Sản phẩm thừa 52_52").first()
        self.assertIsNone(prod_51)
        self.assertIsNone(prod_52)

    @patch("app.services.ai.providers.mwapi.httpx.Client")
    def test_undergeneration_zero_persistence(self, mock_client_cls):
        """When AI returns 49 products for count=50, it fails locally with zero persistence and zero retry."""
        self.db.add(Setting(key="active_ai_provider", value="mwapi"))
        self.db.add(Setting(key="mwapi_api_key", value="test-dummy-key-00000000"))
        self.db.add(Setting(key="mwapi_model", value="claude-sonnet-4-6"))
        self.db.commit()

        compact_49 = [
            {
                "nv": f"Sản phẩm thiếu {i}",
                "nc": f"产品 {i}",
                "dk": f"好物 {i}",
                "ca": f"Góc quay {i}",
                "h": f"Hook {i}"
            }
            for i in range(1, 50)
        ]
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(compact_49, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 400, "completion_tokens": 3700, "total_tokens": 4100},
            "model": "claude-sonnet-4-6"
        }
        mock_resp.headers = {}
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        response = self.client.post(
            "/research",
            data={"niche": "niche thiếu 49", "product_count": "50", "fresh": "1"},
            follow_redirects=False
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(mock_client.post.call_count, 1)

        prods = self.db.query(Product).filter(Product.niche == "niche thiếu 49").all()
        self.assertEqual(len(prods), 0, "Must persist zero products on under-generation")

    # --------------------------------------------------------------------------
    # G. WIRE SIZE DETERMINISTIC COMPARISON
    # --------------------------------------------------------------------------
    def test_wire_size_reduction_deterministic(self):
        """Verify compact JSON format is deterministically smaller than canonical JSON for 50 items."""
        sample_50_canonical = [
            {
                "name_vietnamese": f"Súng bắn bong bóng tự động {i}",
                "name_chinese": f"全自动泡泡枪{i}",
                "douyin_keywords": f"泡泡枪测评 好物推荐{i} 网红玩具",
                "content_angle": f"Trẻ em mê mẩn không rời tay hàng ngàn bong bóng mỗi phút {i}",
                "hook": f"Đứa trẻ nào nhìn thấy món đồ số {i} này cũng thích mê!"
            }
            for i in range(1, 51)
        ]
        sample_50_compact = [
            {
                "nv": f"Súng bắn bong bóng tự động {i}",
                "nc": f"全自动泡泡枪{i}",
                "dk": f"泡泡枪测评 好物推荐{i} 网红玩具",
                "ca": f"Trẻ em mê mẩn không rời tay hàng ngàn bong bóng mỗi phút {i}",
                "h": f"Đứa trẻ nào nhìn thấy món đồ số {i} này cũng thích mê!"
            }
            for i in range(1, 51)
        ]

        canon_json = json.dumps(sample_50_canonical, ensure_ascii=False)
        compact_json = json.dumps(sample_50_compact, ensure_ascii=False)

        canon_chars = len(canon_json)
        compact_chars = len(compact_json)
        char_diff = canon_chars - compact_chars

        canon_bytes = len(canon_json.encode("utf-8"))
        compact_bytes = len(compact_json.encode("utf-8"))
        byte_diff = canon_bytes - compact_bytes

        pct_reduction = (char_diff / canon_chars) * 100.0

        # Assert at least 2,000 characters and > 12% reduction on 50 products
        self.assertGreater(char_diff, 2000)
        self.assertGreater(byte_diff, 2000)
        self.assertGreater(pct_reduction, 12.0)

    # --------------------------------------------------------------------------
    # H. HTTP 524 ERROR HANDLING
    # --------------------------------------------------------------------------
    def test_http_524_handling(self):
        """Verify upstream HTTP 524 is classified with Retry-After and Ray ID, stops with 0 retries."""
        headers = {
            "retry-after": "120",
            "cf-ray": "a431e9179b498545-HKG"
        }
        err = classify_http_error(
            status_code=524,
            response_text="<!DOCTYPE html><html><title>524 A timeout occurred</title></html>",
            provider="mwapi",
            model="claude-sonnet-4-6",
            response_headers=headers,
            duration_seconds=125.67
        )
        self.assertIsInstance(err, AIServiceUnavailableError)
        self.assertEqual(err.status_code, 524)
        self.assertEqual(err.retry_after, 120)
        self.assertEqual(err.request_id, "a431e9179b498545-HKG")
        self.assertEqual(err.provider, "mwapi")
        self.assertEqual(err.model, "claude-sonnet-4-6")
        self.assertEqual(err.duration_seconds, 125.67)

    # --------------------------------------------------------------------------
    # I. PROVIDER DIAGNOSTIC METADATA
    # --------------------------------------------------------------------------
    @patch("app.services.ai.providers.mwapi.httpx.Client")
    def test_provider_diagnostic_reports_mwapi_on_overload(self, mock_client_cls):
        """Verify overload logging logs Provider=mwapi rather than Provider=unknown."""
        self.db.add(Setting(key="active_ai_provider", value="mwapi"))
        self.db.add(Setting(key="mwapi_api_key", value="test-dummy-key-00000000"))
        self.db.add(Setting(key="mwapi_model", value="claude-sonnet-4-6"))
        self.db.commit()

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 524
        mock_resp.text = "<!DOCTYPE html>524 A timeout occurred"
        mock_resp.headers = {"retry-after": "120", "cf-ray": "a431e9179b498545-HKG"}
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        with self.assertLogs("app.routes.research", level="WARNING") as log_capture:
            response = self.client.post(
                "/research",
                data={"niche": "niche quá tải", "product_count": "50", "fresh": "1"},
                follow_redirects=False
            )
            self.assertEqual(response.status_code, 503)

            # Assert log contains Provider=mwapi
            log_output = "\n".join(log_capture.output)
            self.assertIn("[RESEARCH 503 OVERLOAD]", log_output)
            self.assertIn("Provider=mwapi", log_output)
            self.assertNotIn("Provider=unknown", log_output)
            self.assertIn("RequestID=a431e9179b498545-HKG", log_output)
            self.assertIn("RetryAfter=120s", log_output)

        # Zero partial persistence
        prods = self.db.query(Product).filter(Product.niche == "niche quá tải").all()
        self.assertEqual(len(prods), 0)


if __name__ == "__main__":
    unittest.main()
