"""
AI Provider Manager for Multi-AI Architecture (Phase 1)

Orchestrates AI providers. In Phase 1, automatically registers GeminiProvider as the default
and only active provider. Provides a unified, provider-neutral interface for business services.
"""
import os
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from app.services.ai.base import (
    AIProvider,
    AIGenerationOptions,
    AIProviderError
)
from app.services.ai.providers.gemini import GeminiProvider
from app.services.ai.providers.openai import OpenAIProvider
from app.services.ai.providers.anthropic import AnthropicProvider
from app.services.ai.providers.groq import GroqProvider
from app.services.ai.providers.openrouter import OpenRouterProvider


class AIProviderManager:
    """
    Central manager for discovering, registering, and delegating to AI providers.
    In Phase 2, registers five cloud providers:
    - gemini (direct, default)
    - openai (direct)
    - anthropic (direct)
    - groq (direct)
    - openrouter (gateway)
    """

    DEFAULT_PROVIDER_ID = "gemini"

    def __init__(self):
        self._providers: Dict[str, AIProvider] = {}
        # Register the 5 official cloud providers
        self.register_provider(GeminiProvider())
        self.register_provider(OpenAIProvider())
        self.register_provider(AnthropicProvider())
        self.register_provider(GroqProvider())
        self.register_provider(OpenRouterProvider())

    def register_provider(self, provider: AIProvider) -> None:
        """Register an AI provider instance."""
        if not isinstance(provider, AIProvider):
            raise TypeError("Provider must implement AIProvider interface.")
        self._providers[provider.provider_id] = provider

    def get_provider(self, provider_id: Optional[str] = None) -> AIProvider:
        """
        Look up provider by ID.
        Raises clean controlled AIProviderError if unknown provider is requested.
        """
        pid = (provider_id or self.DEFAULT_PROVIDER_ID).strip().lower()
        if pid not in self._providers:
            raise AIProviderError(
                f"AI Provider '{pid}' is not registered or supported.",
                status_code=400,
                provider=pid
            )
        return self._providers[pid]

    def get_active_provider(self, db: Optional[Session] = None) -> AIProvider:
        """
        Resolve the active provider.
        Priority:
        1. SQLite Setting 'active_ai_provider' (runtime source).
        2. Environment variable ACTIVE_AI_PROVIDER (bootstrap/fallback).
        3. Default 'gemini'.
        """
        from app.config import get_active_ai_provider
        active_id = get_active_ai_provider(db=db)
        if active_id not in self._providers:
            active_id = self.DEFAULT_PROVIDER_ID
        return self.get_provider(active_id)

    def list_registered_providers(self) -> List[Dict[str, Any]]:
        """List all currently registered providers with provider_id, display_name, and provider_type."""
        return [
            {
                "provider_id": p.provider_id,
                "display_name": p.display_name,
                "provider_type": p.provider_type
            }
            for p in self._providers.values()
        ]

    def generate(
        self,
        prompt: str,
        provider_id: Optional[str] = None,
        db: Optional[Session] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        enable_fallback: bool = True,
        options: Optional[AIGenerationOptions] = None,
        **kwargs
    ) -> str:
        """
        Delegate content generation to the active or requested AI provider.
        """
        provider = self.get_provider(provider_id) if provider_id else self.get_active_provider(db=db)
        return provider.generate(
            prompt=prompt,
            db=db,
            timeout=timeout,
            max_retries=max_retries,
            enable_fallback=enable_fallback,
            options=options,
            **kwargs
        )

    def test_connection(
        self,
        provider_id: Optional[str] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """Delegate connection testing to the active or requested AI provider."""
        provider = self.get_provider(provider_id) if provider_id else self.get_active_provider(db=db)
        return provider.test_connection(db=db)

    def list_models(
        self,
        provider_id: Optional[str] = None,
        db: Optional[Session] = None
    ) -> List[str]:
        """Delegate model listing to the active or requested AI provider."""
        provider = self.get_provider(provider_id) if provider_id else self.get_active_provider(db=db)
        return provider.list_models(db=db)

    def get_last_execution_metadata(
        self,
        provider_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve execution metadata from the active or requested AI provider."""
        provider = self.get_provider(provider_id) if provider_id else self.get_active_provider()
        return provider.get_last_execution_metadata()

    # ==========================================================================
    # DOMAIN DELEGATION HELPERS
    # ==========================================================================

    def generate_products(
        self,
        niche: str,
        count: int = 10,
        provider_id: Optional[str] = None,
        db: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        provider = self.get_provider(provider_id) if provider_id else self.get_active_provider(db=db)
        return provider.generate_products(niche=niche, count=count, db=db)

    def generate_timed_script(
        self,
        video_duration: float,
        segments: List[Dict[str, Any]],
        product_info: Optional[Dict[str, Any]] = None,
        provider_id: Optional[str] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        provider = self.get_provider(provider_id) if provider_id else self.get_active_provider(db=db)
        return provider.generate_timed_script(
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
        product_name: str = "",
        provider_id: Optional[str] = None
    ) -> Dict[str, Any]:
        provider = self.get_provider(provider_id) if provider_id else self.get_active_provider(db=db)
        return provider.rewrite_failed_segments(
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
        product_name: str = "",
        provider_id: Optional[str] = None
    ) -> str:
        provider = self.get_provider(provider_id) if provider_id else self.get_active_provider()
        return provider.build_rewrite_prompt(
            failed_segments=failed_segments,
            round_num=round_num,
            max_rounds=max_rounds,
            product_name=product_name
        )


# Global singleton instance
_ai_manager_instance: Optional[AIProviderManager] = None


def get_ai_manager() -> AIProviderManager:
    """Retrieve or create the global AIProviderManager singleton instance."""
    global _ai_manager_instance
    if _ai_manager_instance is None:
        _ai_manager_instance = AIProviderManager()
    return _ai_manager_instance


def set_ai_manager(manager: Optional[AIProviderManager]) -> None:
    """Set or reset the global AIProviderManager instance (useful for unit testing)."""
    global _ai_manager_instance
    _ai_manager_instance = manager
