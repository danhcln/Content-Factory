"""
Gemini Provider Adapter for Multi-AI Architecture (Phase 1)

Wraps and delegates to the hardened, proven GeminiService implementation.
Preserves existing retry/backoff, model discovery, smart fallback, error classification,
and test_connection behaviors without duplicating networking code.
"""
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from app.services.ai.base import (
    AIProvider,
    AIGenerationOptions,
    ExecutionMetadata,
    AIProviderError
)
def get_api_key(db: Optional[Session] = None) -> Optional[str]:
    from app.services.gemini_service import get_api_key as _get_key
    return _get_key(db)


def discover_available_models(api_key: str, timeout: float = 10.0) -> List[str]:
    from app.services.gemini_service import discover_available_models as _discover
    return _discover(api_key, timeout=timeout)


class GeminiProvider(AIProvider):
    """
    Adapter implementing the AIProvider interface by delegating to GeminiService.
    """

    def __init__(self, gemini_service: Optional[Any] = None):
        if gemini_service is None:
            from app.services.gemini_service import GeminiService
            self._gemini_service = GeminiService()
        else:
            self._gemini_service = gemini_service

    @property
    def provider_id(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Google Gemini"

    @property
    def provider_type(self) -> str:
        return "direct"

    def generate(
        self,
        prompt: str,
        db: Optional[Session] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        enable_fallback: bool = True,
        options: Optional[AIGenerationOptions] = None,
        **kwargs
    ) -> str:
        """
        Generate content using GeminiService with exponential backoff and smart model fallback.
        """
        max_output_tokens = kwargs.get("max_output_tokens")
        if options is not None:
            timeout = options.timeout
            max_retries = options.max_retries
            enable_fallback = options.enable_fallback
            if options.max_output_tokens is not None:
                max_output_tokens = options.max_output_tokens

        return self._gemini_service.call_gemini(
            prompt=prompt,
            db=db,
            timeout=timeout,
            max_retries=max_retries,
            enable_fallback=enable_fallback,
            max_output_tokens=max_output_tokens
        )

    def test_connection(self, db: Optional[Session] = None) -> Dict[str, Any]:
        """
        Test Gemini API connection using the configured model and report diagnostics.
        """
        report = self._gemini_service.test_connection(db=db)
        if isinstance(report, dict):
            res_dict = dict(report)
            res_dict["provider"] = self.provider_id
            res_dict["provider_type"] = self.provider_type
            res_dict["configured"] = bool(get_api_key(db))
            res_dict["connected"] = bool(report.get("success", False))
            res_dict["configured_model"] = report.get("model", "")
            res_dict["message"] = report.get("message", report.get("error", ""))
            return res_dict
        return report

    def list_models(self, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        """
        Discover available models via Gemini catalog discovery (with 1-hour cache).
        Normalizes results into standard dictionary structure.
        """
        api_key = get_api_key(db)
        if not api_key:
            return []
        try:
            raw_models = discover_available_models(api_key)
            if not isinstance(raw_models, list):
                return []
            return [
                {
                    "id": m if isinstance(m, str) else m.get("id", ""),
                    "name": m if isinstance(m, str) else m.get("name", ""),
                    "provider": self.provider_id,
                    "provider_type": self.provider_type,
                    "available": True
                }
                for m in raw_models
                if (m if isinstance(m, str) else m.get("id"))
            ]
        except Exception:
            return []

    def supports_model(self, model_name: str) -> bool:
        """Check if model belongs to Gemini family."""
        if not model_name:
            return False
        return model_name.strip().lower().startswith("gemini-")

    def get_last_execution_metadata(self) -> Dict[str, Any]:
        """
        Extract secret-free execution metadata from the last Gemini call.
        """
        raw = getattr(self._gemini_service, "last_execution", {})
        if not isinstance(raw, dict):
            raw = {}

        actual_model = raw.get("actual_model_used", raw.get("primary_model", ""))
        configured_model = raw.get("primary_model", "")
        fallback_used = bool(raw.get("fallback_used", False))
        primary_failure = raw.get("primary_failure")

        meta = ExecutionMetadata(
            provider=self.provider_id,
            provider_type=self.provider_type,
            configured_model=configured_model,
            actual_model_used=actual_model,
            fallback_used=fallback_used,
            status="SUCCESS" if actual_model else "FAILED",
            error_type=str(primary_failure) if primary_failure else None,
            attempts=raw.get("attempts", 1)
        )
        return meta.to_dict()

    # ==========================================================================
    # DOMAIN DELEGATIONS (Preserving mature batch research & script workflows)
    # ==========================================================================

    def generate_products(
        self,
        niche: str,
        count: int = 10,
        db: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        return self._gemini_service.generate_products(niche=niche, count=count, db=db)

    def generate_timed_script(
        self,
        video_duration: float,
        segments: List[Dict[str, Any]],
        product_info: Optional[Dict[str, Any]] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        return self._gemini_service.generate_timed_script(
            video_duration=video_duration,
            segments=segments,
            product_info=product_info,
            db=db
        )

    def rewrite_failed_segments(
        self,
        failed_segments: List[Dict[str, Any]],
        round_num: int = 1,
        max_rounds: int = 2,
        db: Optional[Session] = None,
        product_name: str = ""
    ) -> Dict[str, Any]:
        return self._gemini_service.rewrite_failed_segments(
            failed_segments=failed_segments,
            round_num=round_num,
            max_rounds=max_rounds,
            db=db,
            product_name=product_name
        )

    def build_rewrite_prompt(
        self,
        failed_segments: List[Dict[str, Any]],
        round_num: int = 1,
        max_rounds: int = 2,
        product_name: str = ""
    ) -> str:
        return self._gemini_service.build_rewrite_prompt(
            failed_segments=failed_segments,
            round_num=round_num,
            max_rounds=max_rounds,
            product_name=product_name
        )
