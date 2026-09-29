"""
AI Provider Manager for Multi-AI Architecture (Phase 1)

Orchestrates AI providers. In Phase 1, automatically registers GeminiProvider as the default
and only active provider. Provides a unified, provider-neutral interface for business services.
"""
import os
import logging
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from app.services.ai.base import (
    AIProvider,
    AIGenerationOptions,
    ExecutionMetadata,
    AIProviderError,
    AIBadRequestError,
    AIAuthenticationError,
    AIPermissionError,
    AIQuotaExceededError,
    AIRateLimitError,
    AIFallbackExhaustedError,
)
from app.services.ai.providers.gemini import GeminiProvider
from app.services.ai.providers.openai import OpenAIProvider
from app.services.ai.providers.anthropic import AnthropicProvider
from app.services.ai.providers.groq import GroqProvider
from app.services.ai.providers.openrouter import OpenRouterProvider
from app.config import (
    get_ai_fallback_enabled,
    get_ai_fallback_providers,
    get_ai_fallback_on_quota,
    get_ai_fallback_budget_seconds,
    MIN_CANDIDATE_TIMEOUT_SECONDS,
    MAX_CANDIDATE_TIMEOUT_SECONDS,
)

logger = logging.getLogger("app.services.ai.manager")


def get_provider_key_configured(provider_id: str, db: Optional[Session] = None) -> bool:
    """Check if the given provider has a configured API key."""
    from app.config import (
        get_gemini_api_key,
        get_openai_api_key,
        get_anthropic_api_key,
        get_groq_api_key,
        get_openrouter_api_key,
    )
    pid = (provider_id or "").strip().lower()
    if pid == "gemini":
        return bool(get_gemini_api_key(db))
    elif pid == "openai":
        return bool(get_openai_api_key(db))
    elif pid == "anthropic":
        return bool(get_anthropic_api_key(db))
    elif pid == "groq":
        return bool(get_groq_api_key(db))
    elif pid == "openrouter":
        return bool(get_openrouter_api_key(db))
    return False


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
        self._last_execution_meta: Optional[Dict[str, Any]] = None
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
        allow_cross_provider_fallback: Optional[bool] = None,
        options: Optional[AIGenerationOptions] = None,
        **kwargs
    ) -> str:
        """
        Delegate content generation to the active or requested AI provider.
        Cross-provider fallback is engaged ONLY when:
        1. Customer has explicitly enabled ai_fallback_enabled in Settings (or .env).
        2. Workflow explicitly allows allow_cross_provider_fallback == True.
        3. Caller did not specify an explicit targeted provider_id.
        """
        workflow_allow = False
        if allow_cross_provider_fallback is not None:
            workflow_allow = allow_cross_provider_fallback
        elif options is not None and hasattr(options, "allow_cross_provider_fallback"):
            workflow_allow = options.allow_cross_provider_fallback

        customer_fallback_enabled = get_ai_fallback_enabled(db)
        should_cross_fallback = customer_fallback_enabled and workflow_allow and (provider_id is None)

        if not should_cross_fallback:
            provider = self.get_provider(provider_id) if provider_id else self.get_active_provider(db=db)
            res = provider.generate(
                prompt=prompt,
                db=db,
                timeout=timeout,
                max_retries=max_retries,
                enable_fallback=enable_fallback,
                options=options,
                **kwargs
            )
            raw_meta = provider.get_last_execution_metadata()
            if isinstance(raw_meta, dict):
                self._last_execution_meta = dict(raw_meta)
                self._last_execution_meta["actual_provider_used"] = provider.provider_id
                self._last_execution_meta["internal_fallback_used"] = raw_meta.get("fallback_used", False)
                self._last_execution_meta["cross_provider_fallback_used"] = False
                self._last_execution_meta["fallback_used"] = raw_meta.get("fallback_used", False)
                self._last_execution_meta["total_duration_seconds"] = raw_meta.get("duration_seconds")
            return res

        return self._execute_cross_provider_fallback(
            prompt=prompt,
            db=db,
            timeout=timeout,
            max_retries=max_retries,
            enable_fallback=enable_fallback,
            options=options,
            **kwargs
        )

    def _execute_cross_provider_fallback(
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
        Execute deterministic customer-controlled cross-provider fallback.
        Phase 5 Hardening Guarantees:
        - True end-to-end workflow deadline: starts BEFORE primary provider attempt.
        - Primary + retries + fallback candidates all stay within DEFAULT_AI_FALLBACK_BUDGET_SECONDS.
        - Primary and candidate timeouts are clamped to remaining workflow budget.
        - Fallback candidates strictly use max_retries=0 and candidate timeout <= 20.0s.
        - Halts immediately when remaining budget is below MIN_CANDIDATE_TIMEOUT_SECONDS.
        - Preserves Research invariant (Research permanently sets allow_cross_provider_fallback=False).
        - Propagates full execution metadata (tokens, duration, internal vs cross fallback).
        - Sanitizes all secrets in error messages and metadata.
        """
        import time
        from app.services.ai.providers.common import sanitize_secrets

        workflow_start = time.monotonic()
        total_budget = get_ai_fallback_budget_seconds(db)

        primary_provider = self.get_active_provider(db=db)
        primary_id = primary_provider.provider_id
        attempted_providers = {primary_id}
        fallback_chain: List[Dict[str, Any]] = []

        # 1. Attempt Primary Provider inside overall workflow deadline
        rem_before_primary = total_budget - (time.monotonic() - workflow_start)
        if rem_before_primary <= 0:
            raise AIFallbackExhaustedError(
                f"Workflow latency budget ({total_budget}s) expired before primary attempt.",
                provider=primary_id
            )

        primary_timeout = min(timeout, rem_before_primary)
        primary_options = options
        if options:
            primary_options = AIGenerationOptions(
                timeout=primary_timeout,
                max_retries=options.max_retries,
                enable_fallback=options.enable_fallback,
                allow_cross_provider_fallback=options.allow_cross_provider_fallback,
                temperature=options.temperature,
                max_output_tokens=options.max_output_tokens,
                system_instruction=options.system_instruction,
                extra_params=options.extra_params
            )

        try:
            res = primary_provider.generate(
                prompt=prompt,
                db=db,
                timeout=primary_timeout,
                max_retries=max_retries,
                enable_fallback=enable_fallback,
                options=primary_options,
                **kwargs
            )
            raw_meta = primary_provider.get_last_execution_metadata()
            dur = round(time.monotonic() - workflow_start, 3)
            if isinstance(raw_meta, dict):
                self._last_execution_meta = dict(raw_meta)
                self._last_execution_meta["actual_provider_used"] = primary_id
                self._last_execution_meta["internal_fallback_used"] = raw_meta.get("fallback_used", False)
                self._last_execution_meta["cross_provider_fallback_used"] = False
                self._last_execution_meta["fallback_used"] = raw_meta.get("fallback_used", False)
                self._last_execution_meta["fallback_attempts"] = 0
                self._last_execution_meta["total_duration_seconds"] = dur
                self._last_execution_meta["fallback_chain"] = [{
                    "provider": primary_id,
                    "status": "SUCCESS",
                    "model": raw_meta.get("actual_model_used", "")
                }]
            return res
        except Exception as primary_err:
            # Check error eligibility for fallback
            # 400 Bad Request, 401 Auth, 403 Permission are NEVER eligible
            if isinstance(primary_err, (AIBadRequestError, AIAuthenticationError, AIPermissionError)):
                logger.warning(
                    f"Primary provider '{primary_id}' failed with non-fallback error "
                    f"({type(primary_err).__name__}). Aborting without cross-provider fallback."
                )
                raise primary_err

            # 429 Quota / Rate Limit: only eligible if customer explicitly enabled ai_fallback_on_quota
            if isinstance(primary_err, (AIQuotaExceededError, AIRateLimitError)):
                if not get_ai_fallback_on_quota(db):
                    logger.warning(
                        f"Primary provider '{primary_id}' hit 429 quota/rate limit, but "
                        f"ai_fallback_on_quota is disabled. Aborting without cross-provider fallback."
                    )
                    raise primary_err

            # Eligible failure on primary -> record in chain
            err_type = getattr(primary_err, "error_type", "error")
            clean_err = sanitize_secrets(str(primary_err))
            fallback_chain.append({
                "provider": primary_id,
                "status": "FAILED",
                "error_type": err_type,
                "error": clean_err
            })
            logger.warning(
                f"[CROSS-PROVIDER FALLBACK] Primary provider '{primary_id}' failed "
                f"({err_type}: {clean_err}). Evaluating fallback candidates..."
            )

        # 2. Check remaining workflow budget after primary failure
        elapsed_after_primary = time.monotonic() - workflow_start
        remaining_budget = total_budget - elapsed_after_primary
        if remaining_budget < MIN_CANDIDATE_TIMEOUT_SECONDS:
            logger.warning(
                f"[CROSS-PROVIDER FALLBACK] Workflow budget ({total_budget}s) exhausted after "
                f"primary attempt ({elapsed_after_primary:.2f}s elapsed). Aborting fallback."
            )
            self._last_execution_meta = {
                "provider": primary_id,
                "actual_provider_used": "",
                "fallback_used": True,
                "internal_fallback_used": False,
                "cross_provider_fallback_used": True,
                "fallback_attempts": 0,
                "fallback_chain": fallback_chain,
                "status": "FAILED",
                "error_type": "fallback_exhausted",
                "duration_seconds": round(elapsed_after_primary, 3),
                "total_duration_seconds": round(elapsed_after_primary, 3)
            }
            chain_desc = "; ".join(f"{c['provider']}: {c.get('error_type', 'failed')}" for c in fallback_chain)
            raise AIFallbackExhaustedError(
                f"All configured AI providers failed. Workflow budget ({total_budget}s) exhausted. Attempts: {chain_desc}",
                provider=primary_id
            )

        # 3. Retrieve and filter fallback candidates
        candidate_ids = get_ai_fallback_providers(db)
        eligible_candidates = []
        for cid in candidate_ids:
            if cid not in attempted_providers and cid in self._providers:
                if get_provider_key_configured(cid, db):
                    eligible_candidates.append(cid)
                else:
                    logger.info(
                        f"[CROSS-PROVIDER FALLBACK] Candidate '{cid}' is in fallback list "
                        f"but has no configured API key. Skipping."
                    )

        if not eligible_candidates:
            self._last_execution_meta = {
                "provider": primary_id,
                "actual_provider_used": "",
                "fallback_used": True,
                "internal_fallback_used": False,
                "cross_provider_fallback_used": True,
                "fallback_attempts": 0,
                "fallback_chain": fallback_chain,
                "status": "FAILED",
                "error_type": "fallback_exhausted",
                "duration_seconds": round(time.monotonic() - workflow_start, 3),
                "total_duration_seconds": round(time.monotonic() - workflow_start, 3)
            }
            chain_desc = "; ".join(f"{c['provider']}: {c.get('error_type', 'failed')}" for c in fallback_chain)
            raise AIFallbackExhaustedError(
                f"All configured AI providers failed. Attempts: {chain_desc}",
                provider=primary_id
            )

        # 4. Iterate candidates deterministically inside remaining budget
        for cand_id in eligible_candidates:
            rem_budget = total_budget - (time.monotonic() - workflow_start)
            if rem_budget < MIN_CANDIDATE_TIMEOUT_SECONDS:
                logger.warning(
                    f"[CROSS-PROVIDER FALLBACK] Remaining budget ({rem_budget:.1f}s) is below "
                    f"minimum viable threshold ({MIN_CANDIDATE_TIMEOUT_SECONDS}s). Halting fallback."
                )
                break

            attempted_providers.add(cand_id)
            cand_provider = self.get_provider(cand_id)
            cand_timeout = min(timeout, rem_budget, MAX_CANDIDATE_TIMEOUT_SECONDS)

            cand_options = None
            if options:
                cand_options = AIGenerationOptions(
                    timeout=cand_timeout,
                    max_retries=0,  # Fallback candidates never retry
                    enable_fallback=False,
                    allow_cross_provider_fallback=False,
                    temperature=options.temperature,
                    max_output_tokens=options.max_output_tokens,
                    system_instruction=options.system_instruction,
                    extra_params=options.extra_params
                )
            else:
                cand_options = AIGenerationOptions(
                    timeout=cand_timeout,
                    max_retries=0,
                    enable_fallback=False,
                    allow_cross_provider_fallback=False
                )

            try:
                logger.info(f"[CROSS-PROVIDER FALLBACK] Routing request to candidate provider '{cand_id}' (timeout: {cand_timeout:.1f}s)...")
                cand_res = cand_provider.generate(
                    prompt=prompt,
                    db=db,
                    timeout=cand_timeout,
                    max_retries=0,
                    enable_fallback=False,
                    options=cand_options,
                    **kwargs
                )
                raw_cand_meta = cand_provider.get_last_execution_metadata()
                actual_model = raw_cand_meta.get("actual_model_used", "") if isinstance(raw_cand_meta, dict) else ""
                fallback_chain.append({
                    "provider": cand_id,
                    "status": "SUCCESS",
                    "model": actual_model
                })

                total_dur = round(time.monotonic() - workflow_start, 3)
                cand_dur = raw_cand_meta.get("duration_seconds", cand_timeout) if isinstance(raw_cand_meta, dict) else cand_timeout

                self._last_execution_meta = {
                    "provider": primary_id,
                    "provider_type": primary_provider.provider_type,
                    "actual_provider_used": cand_id,
                    "actual_model_used": actual_model,
                    "fallback_used": True,
                    "internal_fallback_used": False,
                    "cross_provider_fallback_used": True,
                    "fallback_attempts": len(fallback_chain) - 1,
                    "fallback_chain": fallback_chain,
                    "status": "SUCCESS",
                    "error_type": None,
                    "attempts": len(fallback_chain),
                    "duration_seconds": cand_dur,
                    "total_duration_seconds": total_dur,
                    "input_tokens": raw_cand_meta.get("input_tokens") if isinstance(raw_cand_meta, dict) else None,
                    "output_tokens": raw_cand_meta.get("output_tokens") if isinstance(raw_cand_meta, dict) else None,
                    "total_tokens": raw_cand_meta.get("total_tokens") if isinstance(raw_cand_meta, dict) else None,
                }
                logger.info(
                    f"[CROSS-PROVIDER FALLBACK] Succeeded with candidate provider '{cand_id}' "
                    f"(model: '{actual_model}') in {total_dur}s total."
                )
                return cand_res
            except Exception as cand_err:
                cand_err_type = getattr(cand_err, "error_type", "error")
                clean_cand_err = sanitize_secrets(str(cand_err))
                fallback_chain.append({
                    "provider": cand_id,
                    "status": "FAILED",
                    "error_type": cand_err_type,
                    "error": clean_cand_err
                })
                logger.warning(
                    f"[CROSS-PROVIDER FALLBACK] Candidate '{cand_id}' failed "
                    f"({cand_err_type}: {clean_cand_err}). Continuing chain..."
                )

        # 5. If all candidates fail or deadline reached
        total_fail_dur = round(time.monotonic() - workflow_start, 3)
        self._last_execution_meta = {
            "provider": primary_id,
            "actual_provider_used": "",
            "fallback_used": True,
            "internal_fallback_used": False,
            "cross_provider_fallback_used": True,
            "fallback_attempts": len(fallback_chain) - 1,
            "fallback_chain": fallback_chain,
            "status": "FAILED",
            "error_type": "fallback_exhausted",
            "duration_seconds": total_fail_dur,
            "total_duration_seconds": total_fail_dur,
        }
        chain_desc = "; ".join(f"{c['provider']}: {c.get('error_type', 'failed')}" for c in fallback_chain)
        raise AIFallbackExhaustedError(
            f"All configured AI providers failed. Workflow budget: {total_budget}s. Attempts: {chain_desc}",
            provider=primary_id
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
        """Retrieve execution metadata from the active or requested AI provider or recent fallback operation."""
        if provider_id is None and self._last_execution_meta is not None:
            return dict(self._last_execution_meta)
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
