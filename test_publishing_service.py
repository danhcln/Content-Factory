import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.database import SessionLocal
from app.models import Video, Product, Publishing, SocialAccount
from app.services.publishing.manager import PublishingManager
from app.services.publishing.base import BasePublisherProvider


class TestPublishingService(unittest.TestCase):
    def setUp(self):
        self.db = SessionLocal()
        self.manager = PublishingManager()

    def tearDown(self):
        self.db.close()

    def test_01_six_providers_registered(self):
        """Verify that exactly 6 target platforms are registered (Facebook Personal completely removed)."""
        providers = self.manager.providers
        self.assertEqual(len(providers), 6)
        platform_keys = list(providers.keys())
        self.assertIn("facebook_page", platform_keys)
        self.assertIn("instagram", platform_keys)
        self.assertIn("tiktok", platform_keys)
        self.assertIn("threads", platform_keys)
        self.assertIn("youtube", platform_keys)
        self.assertIn("shopee", platform_keys)
        # Strict removal check
        self.assertNotIn("facebook_personal", platform_keys)
        self.assertIsNone(self.manager.get_provider("facebook_personal"))

    def test_02_manual_required_shopee_provider(self):
        """Verify Shopee Video is accurately designated as MANUAL_REQUIRED and Facebook Personal is absent."""
        shopee = self.manager.get_provider("shopee")
        self.assertIsNotNone(shopee)

        status_shopee = shopee.get_status(self.db)
        self.assertEqual(status_shopee["status"], "MANUAL_REQUIRED")
        self.assertTrue(status_shopee.get("requires_manual", False))

        dummy_video = Path("downloads/final/V0172.mp4")
        res_shopee = shopee.publish_video(dummy_video, "Test Caption")
        self.assertEqual(res_shopee["status"], "MANUAL_REQUIRED")
        self.assertIn("message", res_shopee)

        # Confirm Facebook Personal returns None
        self.assertIsNone(self.manager.get_provider("facebook_personal"))

    def test_03_api_providers_connection_check(self):
        """Verify official API providers (5 platforms) report accurate NOT_CONNECTED when credentials are not configured."""
        api_platforms = ["facebook_page", "instagram", "tiktok", "threads", "youtube"]
        for key in api_platforms:
            provider = self.manager.get_provider(key)
            self.assertIsNotNone(provider, f"Provider for {key} should exist")
            status = provider.get_status(self.db)
            self.assertEqual(status["status"], "NOT_CONNECTED", f"{key} should be NOT_CONNECTED without config")

    def test_04_media_validation(self):
        """Verify video media validation checks file existence, size, and format."""
        provider = self.manager.get_provider("facebook_page")
        # Non-existent file
        val_bad = provider.validate_media(Path("non_existent_file.mp4"))
        self.assertFalse(val_bad["valid"])

        # Real V0172 file
        val_good = provider.validate_media(Path("downloads/final/V0172.mp4"))
        self.assertTrue(val_good["valid"])
        self.assertGreater(val_good["size_bytes"], 1000000)

    def test_05_publish_to_all_six_platforms(self):
        """
        Verify publish_to_all processes exactly 6 platforms without crashing,
        records each platform's status in DB, and excludes Facebook Personal.
        """
        report = self.manager.publish_to_all("V0172", db=self.db)
        self.assertTrue(report["success"])
        self.assertEqual(report["video_id"], "V0172")
        self.assertIn("report", report)
        self.assertEqual(len(report["report"]), 6)

        # Confirm Facebook Personal is NOT present in the publish report
        self.assertNotIn("facebook_personal", report["report"])

        # Shopee should show MANUAL_REQUIRED
        self.assertEqual(report["report"]["shopee"]["status"], "MANUAL_REQUIRED")

        # Unconfigured platforms should show NOT_CONNECTED
        self.assertEqual(report["report"]["facebook_page"]["status"], "NOT_CONNECTED")
        self.assertEqual(report["report"]["instagram"]["status"], "NOT_CONNECTED")
        self.assertEqual(report["report"]["tiktok"]["status"], "NOT_CONNECTED")
        self.assertEqual(report["report"]["threads"]["status"], "NOT_CONNECTED")
        self.assertEqual(report["report"]["youtube"]["status"], "NOT_CONNECTED")

        # Verify database Publishing record was updated for V0172
        pub_rec = self.db.query(Publishing).filter(Publishing.video_id == "V0172").first()
        self.assertIsNotNone(pub_rec)
        self.assertEqual(pub_rec.video_id, "V0172")
        self.assertIsNotNone(pub_rec.publish_status)


if __name__ == "__main__":
    unittest.main()
