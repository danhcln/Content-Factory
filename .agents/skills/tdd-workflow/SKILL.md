---
name: tdd-workflow
description: Test-driven development discipline adapted for Python and FastAPI in Content Factory. Emphasizes writing focused offline mock tests first, implementing minimum clean code, and verifying with regression suites.
---

<!-- Derived from: affaan-m/ECC (tdd-guide / tdd-workflow) -->

# Test-Driven Development (TDD) Workflow

Follow this discipline when adding features, provider adapters, routes, or bug fixes in Content Factory.

## Core Principle
**Write focused offline tests before writing implementation code. Mock all external APIs.**

## Practical TDD Cycle

### 1. Understand Desired Behavior
- Identify the exact input, expected output, and edge-case error conditions for the target function or route.
- Clarify error sanitization requirements and invariant constraints.

### 2. Write Focused Mocked Test First
- Create or update a standalone test file (e.g., `test_<feature_name>.py`).
- Use Python's `unittest.mock` (e.g., `patch`, `MagicMock`, `AsyncMock`) to mock network I/O and external SDKs:
  ```python
  from unittest.mock import patch, MagicMock
  ```
- **Zero Real API Calls:** Never allow tests to hit live Google, OpenAI, Anthropic, Groq, or OpenRouter endpoints.
- Avoid real API keys in fixtures; use dummy values (e.g., `"test-dummy-key-00000000"`).

### 3. Observe Test Failure (Red)
- Run the focused test using the workspace virtual environment:
  ```powershell
  .\.venv\Scripts\python.exe test_<feature_name>.py
  ```
- Confirm the test fails cleanly for the expected reason (e.g., `NotImplementedError`, missing method, assertion failure).

### 4. Implement Minimum Code (Green)
- Write only the code necessary to satisfy the test requirements.
- Adhere to architectural boundaries (e.g., provider abstractions in `app/services/ai/base.py`).
- Sanitize error messages and prevent key leakage in exceptions.

### 5. Verify Focused Test Passes
- Re-run the focused test:
  ```powershell
  .\.venv\Scripts\python.exe test_<feature_name>.py
  ```
- Confirm all assertions pass without error.

### 6. Run Regression Suites
- Run existing regression suites to prevent breaking existing functionality:
  ```powershell
  .\.venv\Scripts\python.exe test_ai_architecture_phase1.py
  .\.venv\Scripts\python.exe test_ai_providers_phase2.py
  .\.venv\Scripts\python.exe test_research_efficiency.py
  ```
- If any regression fails, resolve it before proceeding.

### 7. Review Git Diff
- Run `git diff` on modified files.
- Ensure no accidental edits, debug print statements, or leaked test secrets exist.
