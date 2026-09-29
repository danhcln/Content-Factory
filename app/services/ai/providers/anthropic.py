"""
Anthropic Claude Provider Adapter for Multi-AI Architecture (Phase 2)

Implements Direct BYOK support for Anthropic using native Messages REST API:
- Base: https://api.anthropic.com
- Generation: POST /v1/messages
- Model Discovery: GET /v1/models
- Headers: x-api-key, anthropic-version: 2023-06-01
"""
import time
import json
import logging
from typing import Optional, Dict, Any, List
import httpx
from sqlalchemy.orm import Session

from app.services.ai.base import (
    AIProvider,
    AIGenerationOptions,
    ExecutionMetadata,
    AIProviderError,
    AIAuthenticationError,
    AIQuotaExceededError,
    AITimeoutError,
    AINetworkError,
    AIInvalidResponseError,
)
from app.services.ai.providers.common import (
    sanitize_secrets,
    global_model_cache,
    classify_http_error,
)
from app.config import get_anthropic_api_key, get_anthropic_model, normalize_model_name

logger = logging.getLogger("app.services.ai.providers.anthropic")


class AnthropicProvider(AIProvider):
    """Direct BYOK Provider for Anthropic Claude."""

    BASE_URL = "https://api.anthropic.com"
    API_VERSION = "2023-06-01"

    def __init__(self):
        self._last_execution: Optional[ExecutionMetadata] = None

    @property
    def provider_id(self) -> str:
        return "anthropic"

    @property
    def display_name(self) -> str:
        return "Anthropic Claude"

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
        Generate content using native Anthropic Messages API.
        Strictly obeys max_retries (max_retries=0 for Research => exactly 1 HTTP call).
        """
        api_key = get_anthropic_api_key(db)
        model_name = normalize_model_name(self.provider_id, kwargs.get("model") or get_anthropic_model(db))

        max_output_tokens = kwargs.get("max_output_tokens")
        temperature = kwargs.get("temperature")
        system_instruction = kwargs.get("system_instruction")

        if options is not None:
            timeout = options.timeout
            max_retries = options.max_retries
            enable_fallback = options.enable_fallback
            if options.max_output_tokens is not None:
                max_output_tokens = options.max_output_tokens
            if options.temperature is not None:
                temperature = options.temperature
            if options.system_instruction is not None:
                system_instruction = options.system_instruction

        if not api_key:
            err = AIAuthenticationError(
                "Anthropic API key is not configured. Please add your key in Settings or .env.",
                provider=self.provider_id,
                model=model_name
            )
            self._last_execution = ExecutionMetadata(
                provider=self.provider_id,
                provider_type=self.provider_type,
                configured_model=model_name,
                actual_model_used="",
                status="FAILED",
                error_type="not_configured",
                attempts=0
            )
            raise err

        # Anthropic Messages API requires max_tokens
        limit_tokens = max_output_tokens if (max_output_tokens is not None and max_output_tokens > 0) else 4096

        payload: Dict[str, Any] = {
            "model": model_name,
            "max_tokens": limit_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        if system_instruction:
            payload["system"] = system_instruction
        if temperature is not None:
            payload["temperature"] = temperature

        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json"
        }

        url = f"{self.BASE_URL}/v1/messages"
        total_attempts = max_retries + 1
        start_time = time.time()

        for attempt in range(total_attempts):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)

                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except Exception as je:
                        raise AIInvalidResponseError(
                            f"Anthropic returned unparseable JSON: {je}",
                            provider=self.provider_id,
                            model=model_name
                        )

                    content_blocks = data.get("content", [])
                    if not content_blocks or not isinstance(content_blocks, list):
                        raise AIInvalidResponseError(
                            "Anthropic response contains no content blocks.",
                            provider=self.provider_id,
                            model=model_name
                        )

                    text_parts = [
                        block.get("text", "")
                        for block in content_blocks
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    content = "".join(text_parts).strip()

                    usage = data.get("usage", {})
                    in_tok = usage.get("input_tokens")
                    out_tok = usage.get("output_tokens")
                    tot_tok = (in_tok + out_tok) if (in_tok is not None and out_tok is not None) else None

                    actual_model = data.get("model", model_name)
                    dur = round(time.time() - start_time, 3)

                    self._last_execution = ExecutionMetadata(
                        provider=self.provider_id,
                        provider_type=self.provider_type,
                        configured_model=model_name,
                        actual_model_used=actual_model,
                        fallback_used=False,
                        status="SUCCESS",
                        attempts=attempt + 1,
                        duration_seconds=dur,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        total_tokens=tot_tok
                    )
                    return content

                err = classify_http_error(
                    status_code=resp.status_code,
                    response_text=resp.text,
                    provider=self.provider_id,
                    model=model_name,
                    api_key=api_key
                )

                is_retryable = resp.status_code in (429, 500, 502, 503, 504) and not isinstance(err, AIQuotaExceededError)
                if not is_retryable or attempt >= max_retries:
                    self._last_execution = ExecutionMetadata(
                        provider=self.provider_id,
                        provider_type=self.provider_type,
                        configured_model=model_name,
                        actual_model_used="",
                        status="FAILED",
                        error_type=err.error_type,
                        attempts=attempt + 1,
                        duration_seconds=round(time.time() - start_time, 3)
                    )
                    raise err

                logger.warning(
                    f"Anthropic transient error ({resp.status_code}) on attempt {attempt + 1}/{total_attempts}, retrying..."
                )
                time.sleep(min(2.0 ** attempt, 8.0))

            except httpx.TimeoutException as te:
                err = AITimeoutError(
                    f"Anthropic request timed out after {timeout}s: {sanitize_secrets(str(te), api_key)}",
                    provider=self.provider_id,
                    model=model_name
                )
                if attempt >= max_retries:
                    self._last_execution = ExecutionMetadata(
                        provider=self.provider_id,
                        provider_type=self.provider_type,
                        configured_model=model_name,
                        actual_model_used="",
                        status="FAILED",
                        error_type="timeout",
                        attempts=attempt + 1,
                        duration_seconds=round(time.time() - start_time, 3)
                    )
                    raise err
                time.sleep(1.0)

            except httpx.RequestError as re:
                err = AINetworkError(
                    f"Anthropic network error: {sanitize_secrets(str(re), api_key)}",
                    provider=self.provider_id,
                    model=model_name
                )
                if attempt >= max_retries:
                    self._last_execution = ExecutionMetadata(
                        provider=self.provider_id,
                        provider_type=self.provider_type,
                        configured_model=model_name,
                        actual_model_used="",
                        status="FAILED",
                        error_type="network",
                        attempts=attempt + 1,
                        duration_seconds=round(time.time() - start_time, 3)
                    )
                    raise err
                time.sleep(1.0)

        raise AIProviderError("Anthropic generation failed after all attempts.", provider=self.provider_id, model=model_name)

    def test_connection(self, db: Optional[Session] = None) -> Dict[str, Any]:
        """
        Validate Anthropic credentials via lightweight GET /v1/models check.
        Consumes ZERO generation tokens.
        """
        api_key = get_anthropic_api_key(db)
        model_name = get_anthropic_model(db)

        if not api_key:
            return {
                "provider": self.provider_id,
                "provider_type": self.provider_type,
                "configured": False,
                "connected": False,
                "configured_model": model_name,
                "message": "Anthropic API key is not configured.",
                "error_type": "not_configured"
            }

        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json"
        }
        url = f"{self.BASE_URL}/v1/models"

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(url, headers=headers)

            if resp.status_code == 200:
                return {
                    "provider": self.provider_id,
                    "provider_type": self.provider_type,
                    "configured": True,
                    "connected": True,
                    "configured_model": model_name,
                    "message": f"Connected successfully to Anthropic ({model_name}).",
                    "error_type": None
                }

            err = classify_http_error(
                status_code=resp.status_code,
                response_text=resp.text,
                provider=self.provider_id,
                model=model_name,
                api_key=api_key
            )
            return {
                "provider": self.provider_id,
                "provider_type": self.provider_type,
                "configured": True,
                "connected": False,
                "configured_model": model_name,
                "message": err.message,
                "error_type": err.error_type
            }

        except httpx.TimeoutException:
            return {
                "provider": self.provider_id,
                "provider_type": self.provider_type,
                "configured": True,
                "connected": False,
                "configured_model": model_name,
                "message": "Anthropic connection timed out after 15s.",
                "error_type": "timeout"
            }
        except Exception as e:
            return {
                "provider": self.provider_id,
                "provider_type": self.provider_type,
                "configured": True,
                "connected": False,
                "configured_model": model_name,
                "message": sanitize_secrets(str(e), api_key),
                "error_type": "network"
            }

    def list_models(self, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        """
        Dynamically list Anthropic Claude models with 60-minute in-memory caching.
        """
        api_key = get_anthropic_api_key(db)
        if not api_key:
            return []

        cached = global_model_cache.get(self.provider_id, api_key)
        if cached is not None:
            return cached

        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json"
        }
        url = f"{self.BASE_URL}/v1/models"

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(url, headers=headers)

            if resp.status_code != 200:
                logger.warning(f"Failed to discover Anthropic models: HTTP {resp.status_code}")
                return []

            data = resp.json()
            raw_list = data.get("data", [])
            if not isinstance(raw_list, list):
                return []

            models = []
            for item in raw_list:
                mid = str(item.get("id", "")).strip()
                if not mid:
                    continue
                display = str(item.get("display_name", mid)).strip()

                models.append({
                    "id": mid,
                    "name": display or mid,
                    "provider": self.provider_id,
                    "provider_type": self.provider_type,
                    "available": True
                })

            global_model_cache.set(self.provider_id, api_key, models)
            return models

        except Exception as e:
            logger.warning(f"Exception listing Anthropic models: {sanitize_secrets(str(e), api_key)}")
            return []

    def supports_model(self, model_name: str) -> bool:
        """Check if model belongs to Anthropic Claude family."""
        if not model_name:
            return False
        return model_name.strip().lower().startswith("claude-")

    def get_last_execution_metadata(self) -> Dict[str, Any]:
        """Return execution metadata from last operation without secrets."""
        if self._last_execution:
            return self._last_execution.to_dict()
        return ExecutionMetadata(provider=self.provider_id, provider_type=self.provider_type).to_dict()
