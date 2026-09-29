"""
OpenRouter Provider Adapter for Multi-AI Architecture (Phase 2)

Implements Cloud AI Gateway support via OpenRouter:
- Base: https://openrouter.ai/api/v1
- Generation: POST /chat/completions
- Model Discovery: GET /models
- Gateway Routing Safety:
  * Strict single-model submission (never sends 'models' fallback array).
  * Sets provider.allow_fallbacks = False when enable_fallback=False (Research mode).
  * Exposes provider_type = 'gateway'.
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
from app.config import get_openrouter_api_key, get_openrouter_model

logger = logging.getLogger("app.services.ai.providers.openrouter")


class OpenRouterProvider(AIProvider):
    """Cloud AI Gateway Provider for OpenRouter."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self):
        self._last_execution: Optional[ExecutionMetadata] = None

    @property
    def provider_id(self) -> str:
        return "openrouter"

    @property
    def display_name(self) -> str:
        return "OpenRouter"

    @property
    def provider_type(self) -> str:
        return "gateway"

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
        Generate content using OpenRouter Chat Completions endpoint.
        Strictly obeys max_retries (max_retries=0 for Research => exactly 1 HTTP call).
        Strictly submits EXACTLY ONE model (never models[] array).
        Configures provider.allow_fallbacks = False when enable_fallback is False.
        """
        api_key = get_openrouter_api_key(db)
        model_name = kwargs.get("model") or get_openrouter_model(db)

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
                "OpenRouter API key is not configured. Please add your key in Settings or .env.",
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

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        # CRITICAL SAFETY: Single model string only. Never models: [...] array.
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages
        }
        if max_output_tokens is not None and max_output_tokens > 0:
            payload["max_tokens"] = max_output_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        # If fallback is explicitly disabled (e.g., Research mode),
        # instruct OpenRouter not to failover between provider backends
        if not enable_fallback:
            payload["provider"] = {
                "allow_fallbacks": False
            }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://contentfactory.local",
            "X-Title": "Content Factory"
        }

        url = f"{self.BASE_URL}/chat/completions"
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
                            f"OpenRouter returned unparseable JSON: {je}",
                            provider=self.provider_id,
                            model=model_name
                        )

                    choices = data.get("choices", [])
                    if not choices or not isinstance(choices, list):
                        raise AIInvalidResponseError(
                            "OpenRouter response contains no choices.",
                            provider=self.provider_id,
                            model=model_name
                        )

                    msg_obj = choices[0].get("message", {})
                    content = msg_obj.get("content", "")
                    if content is None:
                        content = ""

                    usage = data.get("usage", {})
                    in_tok = usage.get("prompt_tokens")
                    out_tok = usage.get("completion_tokens")
                    tot_tok = usage.get("total_tokens")

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
                    f"OpenRouter transient error ({resp.status_code}) on attempt {attempt + 1}/{total_attempts}, retrying..."
                )
                time.sleep(min(2.0 ** attempt, 8.0))

            except httpx.TimeoutException as te:
                err = AITimeoutError(
                    f"OpenRouter request timed out after {timeout}s: {sanitize_secrets(str(te), api_key)}",
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
                    f"OpenRouter network error: {sanitize_secrets(str(re), api_key)}",
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

        raise AIProviderError("OpenRouter generation failed after all attempts.", provider=self.provider_id, model=model_name)

    def test_connection(self, db: Optional[Session] = None) -> Dict[str, Any]:
        """
        Validate OpenRouter credentials via lightweight GET /models check.
        Consumes ZERO generation tokens.
        """
        api_key = get_openrouter_api_key(db)
        model_name = get_openrouter_model(db)

        if not api_key:
            return {
                "provider": self.provider_id,
                "provider_type": self.provider_type,
                "configured": False,
                "connected": False,
                "configured_model": model_name,
                "message": "OpenRouter API key is not configured.",
                "error_type": "not_configured"
            }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        url = f"{self.BASE_URL}/models"

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
                    "message": f"Connected successfully to OpenRouter ({model_name}).",
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
                "message": "OpenRouter connection timed out after 15s.",
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
        Dynamically list OpenRouter models with 60-minute in-memory caching.
        """
        api_key = get_openrouter_api_key(db)
        if not api_key:
            return []

        cached = global_model_cache.get(self.provider_id, api_key)
        if cached is not None:
            return cached

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        url = f"{self.BASE_URL}/models"

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(url, headers=headers)

            if resp.status_code != 200:
                logger.warning(f"Failed to discover OpenRouter models: HTTP {resp.status_code}")
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
                display = str(item.get("name", mid)).strip()

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
            logger.warning(f"Exception listing OpenRouter models: {sanitize_secrets(str(e), api_key)}")
            return []

    def supports_model(self, model_name: str) -> bool:
        """OpenRouter supports vendor/model scoped strings or any model in its catalog."""
        if not model_name:
            return False
        clean = model_name.strip().lower()
        return "/" in clean

    def get_last_execution_metadata(self) -> Dict[str, Any]:
        """Return execution metadata from last operation without secrets."""
        if self._last_execution:
            return self._last_execution.to_dict()
        return ExecutionMetadata(provider=self.provider_id, provider_type=self.provider_type).to_dict()
