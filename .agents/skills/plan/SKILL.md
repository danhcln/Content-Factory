---
name: plan
description: Structured implementation planning before code modifications. Restates requirements, evaluates architecture boundaries and project invariants, identifies affected files and tests, and produces a step-by-step implementation plan without writing code.
---

<!-- Derived from: affaan-m/ECC (planner / plan) -->

# Implementation Planning Skill

Use this skill when starting a new phase, implementing a feature, or planning an architectural refactoring in Content Factory.

## Core Principle
**Plan first. Never modify code during the planning phase.**

## Workflow Steps

### 1. Inspect Current Repository State
- Verify the current Git branch: `git branch --show-current`.
- Verify the current HEAD hash: `git rev-parse HEAD`.
- Verify clean working tree: `git status`.
- Check active stashes: `git stash list` (ensure `stash@{0}` is preserved).

### 2. Understand Requested Objective
- Restate the functional and technical goals in clear, unambiguous terms.
- Confirm scope limits: clarify what is explicitly IN scope and OUT of scope.

### 3. Check Architecture Boundaries & Invariants
- Read `.agents/rules/00-content-factory-invariants.md`.
- Ensure strict compliance:
  - **Research Invariant:** Max 1 AI request on cache miss, 0 on cache hit. No retries, no fallback models, no agent loops.
  - **Multi-AI Invariant:** Preserve `AIProviderManager` abstraction. Respect Direct BYOK (Gemini, OpenAI, Anthropic, Groq) and Gateway BYOK (OpenRouter). No Ollama/Local AI.
  - **Security Invariant:** Never log raw credentials or commit `.env`.
  - **Data Invariant:** Never delete SQLite database or media assets.

### 4. Ground in Existing Codebase Patterns
- Inspect existing patterns in:
  - Provider adapters: `app/services/ai/providers/`
  - Manager interface: `app/services/ai/manager.py`
  - Routes & templates: `app/routes/`, `app/templates/`
  - Test suites: `test_ai_architecture_phase1.py`, `test_ai_providers_phase2.py`

### 5. Identify Affected Files & New Components
- Explicitly enumerate files to create, modify, or delete.
- Prevent shotgun editing across unrelated modules.

### 6. Specify Test Strategy (Mocked by Default)
- Enumerate required unit and integration tests.
- Ensure all external HTTP / AI requests are mocked (0 paid API quota).
- Identify existing regression test suites to run after completion.

### 7. Risk & Dependency Assessment
- Identify potential breaking changes, backward compatibility concerns, and edge cases.
- Detail rollback strategy if implementation encounters blockers.

### 8. Output Structured Plan
Produce a clear markdown plan with numbered phases:
```markdown
## Phase Overview
## Architectural Design & Pattern Grounding
## Affected Files
## Testing Strategy (Offline / Mocked)
## Risk Mitigation
## Verification Checklist
```

### 9. Wait for Confirmation
**STOP.** Do not write or modify application code until the user explicitly reviews and approves the plan.
