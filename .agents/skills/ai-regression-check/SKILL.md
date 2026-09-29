---
name: ai-regression-check
description: Automated offline regression test runner for Content Factory Multi-AI architecture and Research optimization. Verifies provider isolation, token consumption, and error handling without consuming real API quota.
---

<!-- Content Factory Custom Tooling -->

# AI Regression Check Protocol

Use this skill to execute the complete Content Factory AI test suite offline before any checkpoint.

## Core Rules
1. **100% Mocked External HTTP:** Every external network call must be intercepted by mock fixtures.
2. **Zero Paid Quota:** Automated regression runs must NEVER consume real API quota or spend money.
3. **Deterministic Results:** Tests must pass reliably and independently of external service availability.

## Standard Test Suite Execution

Run all core regression suites using the project Python virtual environment:

### 1. Phase 1 Architecture Verification
```powershell
.\.venv\Scripts\python.exe test_ai_architecture_phase1.py
```
- Verifies `AIProviderManager`, provider registry, base abstract classes, and decoupled business logic.

### 2. Phase 2 Multi-AI Provider Verification
```powershell
.\.venv\Scripts\python.exe test_ai_providers_phase2.py
```
- Verifies provider adapters for Gemini, OpenAI, Anthropic, Groq, and OpenRouter with mocked responses.

### 3. Research Low-Consumption Verification
```powershell
.\.venv\Scripts\python.exe test_research_efficiency.py
```
- Verifies the strict 1-call limit on cache miss, 0-call limit on cache hit, token budget limits, and absence of autonomous retry loops.

### 4. Gemini Reliability & Hardening Verification
```powershell
.\.venv\Scripts\python.exe test_gemini_reliability.py
.\.venv\Scripts\python.exe test_gemini_status_ui.py
.\.venv\Scripts\python.exe test_gemini_hardening.py
```
- Verifies Gemini fallback, diagnostics, status UI routes, and error resilience.

## Reporting Format

Report execution results with exact pass/fail counts:
```text
==================================================
AI REGRESSION TEST SUMMARY:
- Architecture (Phase 1): X / X passed
- Providers (Phase 2):    X / X passed
- Research Efficiency:    X / X passed
- Gemini Reliability:     X / X passed
- Gemini Status UI:       X / X passed
- Gemini Hardening:       X / X passed

Total Tests Passed: Y
Total Failures: 0
Real API Calls: 0
Estimated API Cost: $0.00
STATUS: ALL REGRESSION TESTS PASSED
==================================================
```

If any test fails:
**STOP immediately.** Report the failure traceback and do not proceed to checkpointing until the regression is resolved.
