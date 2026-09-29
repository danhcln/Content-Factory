"""
Core Provider Abstraction & Neutral Contracts for Multi-AI Architecture (Phase 1)

This module defines provider-neutral interfaces, options, execution metadata,
and exception hierarchies. It contains ZERO provider-specific implementation details.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session


# ==============================================================================
# PROVIDER-NEUTRAL EXCEPTIONS
# ==============================================================================

class AIProviderError(RuntimeError):
    """Base exception for all AI provider operations, fully compatible with RuntimeError."""
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        upstream_message: str = ""
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.provider = provider
        self.model = model
        self.upstream_message = upstream_message


class AIAuthenticationError(AIProviderError):
    """Raised when authentication fails (HTTP 401 / invalid API key)."""
    pass


class AIPermissionError(AIProviderError):
    """Raised when permission is denied / resource forbidden (HTTP 403)."""
    pass


class AIModelNotFoundError(AIProviderError):
    """Raised when requested model does not exist or is inaccessible (HTTP 404)."""
    pass


class AIQuotaExceededError(AIProviderError):
    """Raised when daily or billing quota is exhausted. Must NOT be retried."""
    pass


class AIRateLimitError(AIProviderError):
    """Raised when temporary rate limit (RPM/TPM) is hit. Retryable."""
    pass


class AIServiceUnavailableError(AIProviderError):
    """Raised when upstream server is overloaded or temporarily unavailable (HTTP 503)."""
    pass


class AITimeoutError(AIProviderError):
    """Raised when request times out."""
    pass


class AINetworkError(AIProviderError):
    """Raised on network connection / DNS failure."""
    pass


# ==============================================================================
# PROVIDER-NEUTRAL OPTIONS & METADATA
# ==============================================================================

@dataclass
class AIGenerationOptions:
    """Provider-neutral configuration options for content generation."""
    timeout: float = 60.0
    max_retries: int = 2
    enable_fallback: bool = True
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    system_instruction: Optional[str] = None
    extra_params: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionMetadata:
    """Provider-neutral execution metadata for audits and monitoring. Never contains secrets."""
    provider: str
    configured_model: str
    actual_model_used: str
    fallback_used: bool = False
    status: str = "SUCCESS"
    error_type: Optional[str] = None
    attempts: int = 1
    duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "configured_model": self.configured_model,
            "actual_model_used": self.actual_model_used,
            "fallback_used": self.fallback_used,
            "status": self.status,
            "error_type": self.error_type,
            "attempts": self.attempts,
            "duration_seconds": self.duration_seconds
        }


# ==============================================================================
# AI PROVIDER ABSTRACT CONTRACT
# ==============================================================================

class AIProvider(ABC):
    """
    Abstract Base Class defining the contract for all AI Providers.
    Concrete providers (Gemini in Phase 1; OpenAI/Claude/etc. in Phase 2) implement this interface.
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique provider identifier (e.g., 'gemini')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable provider name (e.g., 'Google Gemini')."""
        pass

    @abstractmethod
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
        """Generate text response from the provider."""
        pass

    @abstractmethod
    def test_connection(self, db: Optional[Session] = None) -> Dict[str, Any]:
        """Test connection/credentials with the provider."""
        pass

    @abstractmethod
    def list_models(self, db: Optional[Session] = None) -> List[str]:
        """List available models supported/accessible for this provider."""
        pass

    @abstractmethod
    def supports_model(self, model_name: str) -> bool:
        """Check whether the model identifier belongs to or is supported by this provider."""
        pass

    @abstractmethod
    def get_last_execution_metadata(self) -> Dict[str, Any]:
        """Retrieve execution metadata from the most recent operation. Guaranteed secret-free."""
        pass

    # Domain helpers (optional provider-level delegations)
    def generate_products(
        self,
        niche: str,
        count: int = 10,
        db: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError(f"generate_products not implemented by {self.provider_id}")

    def generate_timed_script(
        self,
        video_duration: float,
        segments: List[Dict[str, Any]],
        product_info: Optional[Dict[str, Any]] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError(f"generate_timed_script not implemented by {self.provider_id}")

    def rewrite_failed_segments(
        self,
        failed_segments: List[Dict[str, Any]],
        round_num: int = 1,
        max_rounds: int = 2,
        db: Optional[Session] = None,
        product_name: str = ""
    ) -> Dict[str, Any]:
        raise NotImplementedError(f"rewrite_failed_segments not implemented by {self.provider_id}")

    def build_rewrite_prompt(
        self,
        failed_segments: List[Dict[str, Any]],
        round_num: int = 1,
        max_rounds: int = 2,
        product_name: str = ""
    ) -> str:
        raise NotImplementedError(f"build_rewrite_prompt not implemented by {self.provider_id}")
