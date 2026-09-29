# Content Factory — Core Project Invariants
# Permanent Rule for Antigravity Agent
# Scope: Project-wide invariants across all development phases.

## 1. Research Invariant (Strict Low-Consumption)
- **Cache Hit:** Exactly 0 external AI calls.
- **Cache Miss / Fresh Research:**
  - Exactly ONE selected provider.
  - Exactly ONE selected model.
  - Exactly ONE outbound generation request.
  - Local Python validation only.
  - Save to cache or return error. STOP.
- **Hard Prohibitions in Research:**
  - NO automatic retry.
  - NO second generation attempt.
  - NO fallback model or cross-provider fallback.
  - NO AI-based JSON repair or AI quality retry.
  - NO automatic web search or autonomous agent loop.
  - NO implicit `list_models()`, `test_connection()`, or health checks on user actions.
- **Output Token Budgets:**
  - 5 hooks/points -> 800 tokens.
  - 10 hooks/points -> 1500 tokens.
  - 20 hooks/points -> 2800 tokens.
  - 30 hooks/points -> 4000 tokens.
  - 50 hooks/points -> 6000 tokens.

## 2. Multi-AI Architecture Invariant
- **Direct BYOK:** Google Gemini, OpenAI, Anthropic, Groq.
- **Gateway BYOK:** OpenRouter.
- **Default Provider:** Google Gemini.
- **Core Exclusions:** NO Ollama, NO local AI models, NO 9Router.
- **Future Custom Gateways:** Evaluated separately via explicit phases.
- **Cross-Provider Fallback:** Disallowed unless explicitly designed in future phases.

## 3. Security & Secret Protection Invariant
- NEVER commit `.env` or track it in Git (`git ls-files .env` must be empty).
- NEVER print, log, or expose raw API keys, `Authorization` headers, `x-api-key`, or Bearer tokens.
- NEVER include real API keys in tests, mock fixtures, documentation, or `.env.example`.
- `.env.example` must contain empty placeholders only.
- Upstream provider errors MUST be sanitized to strip keys and tokens before surfacing.

## 4. Git & Checkpoint Safety Invariant
- **Before Modifying Code:** Verify branch, HEAD commit hash, and clean working tree.
- **Before Checkpoint:** Run focused tests, run regression suite, inspect `git diff`, scan for secrets.
- **Safety Locks:** NEVER force push, rebase, drop stashes, pop stashes, or apply old stashes automatically.
- Commit and push ONLY upon explicit user instruction.

## 5. Production Data Protection Invariant
- NEVER casually delete, truncate, or overwrite:
  - SQLite database files (`content.db`, `data/*.db`).
  - Asset directories (`downloads/`, `media/`, original/final videos, audio assets).
  - Production Product or Video records.
- Preserve existing user data and media integrity across all operations.

## 6. Testing & Quota Invariant
- All external AI provider calls MUST be mocked in automated tests.
- Automated test runs must incur ZERO real paid API quota or cost.
- Live external API smoke tests require explicit, advance user authorization.
