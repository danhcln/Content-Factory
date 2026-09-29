---
name: code-review
description: Objective, fresh-context code review for Content Factory. Evaluates correctness, regression risks, architecture boundaries, error sanitization, and project invariants with prioritized findings (CRITICAL, HIGH, MEDIUM, LOW).
---

<!-- Derived from: affaan-m/ECC (code-reviewer) -->

# Code Review Skill

Use this skill to perform an objective quality, security, and architectural audit of code changes before creating checkpoints.

## Core Principle
**Inspect actual diffs objectively. Prioritize real architectural and security risks. Do not invent artificial nitpicks.**

## Review Dimensions

### 1. Correctness & Error Handling
- Are exceptions caught gracefully?
- Are upstream API error messages sanitized so keys, tokens, and sensitive headers are never exposed to clients or written to application logs?
- Are return types consistent with provider base classes (`AIProvider`, `AIResponse`)?

### 2. Project Invariant Compliance
Verify against `.agents/rules/00-content-factory-invariants.md`:
- **Research Invariant:** Is Research constrained to exactly 1 outbound call on cache miss and 0 on cache hit? Are automatic retries, cross-provider fallbacks, or autonomous loops strictly prevented?
- **Multi-AI Architecture:** Does the code maintain clean provider isolation through `AIProviderManager`? Are disallowed providers (Ollama, local models) excluded from the core?
- **Production Data:** Are database schemas, media files, and production assets safeguarded from deletion or accidental truncation?

### 3. Secret Exposure & Security
- Are API keys, Authorization headers, or `x-api-key` values exposed or hardcoded?
- Are tests using dummy tokens rather than real keys?
- Is `.env` ignored?

### 4. Test Coverage & Mocking Discipline
- Are all external HTTP or API requests mocked?
- Do tests run cleanly in the offline virtual environment?
- Are assertion checks rigorous enough to catch edge-case regressions?

### 5. Architectural Boundaries & Simplicity
- Does the change introduce unnecessary dependencies or bloated abstractions?
- Is code duplicated where an existing utility should be used?

## Finding Severities

When reporting issues, classify each finding under one of the following headings:

- **[CRITICAL]:** Flaws that break project invariants, leak credentials, cause production data loss, or introduce live API loops. (Must be fixed before commit).
- **[HIGH]:** Missing error sanitization, unhandled network exceptions, broken regression tests, or architecture boundary breaches.
- **[MEDIUM]:** Code duplication, suboptimal token budgeting, or missing offline unit test edge cases.
- **[LOW]:** Cosmetic formatting, non-blocking docstring updates, or minor naming suggestions.

## Clean Audit Statement
If the code satisfies all invariants and quality standards without issue, state clearly:
`AUDIT PASSED: No critical, high, or medium issues detected.`
