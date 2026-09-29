---
description: Perform fresh-context code review and security audit of modified files before checkpointing.
argument-hint: "[optional target files or commit ref]"
---

# Code Review Workflow

Provides an objective, fresh-context quality and security audit of working tree changes in Content Factory.

## Critical Constraint
This workflow **NEVER** modifies code automatically. It reports findings and recommendations, awaiting user decision.

## Procedure

1. **Invoke Review Skills:**
   - Execute `.agents/skills/code-review/SKILL.md`.
   - Execute `.agents/skills/secret-scanner/SKILL.md`.

2. **Inspect Current Diff:**
   - Run `git diff` against the base checkpoint.
   - Review all modified and newly created files.

3. **Audit Against Content Factory Invariants:**
   - Verify compliance with `.agents/rules/00-content-factory-invariants.md`.
   - Check for Research single-request compliance, Multi-AI architectural abstraction, error message sanitization, and test mocking.

4. **Classify Findings:**
   Organize all observations by severity:
   - **CRITICAL:** Security breaches, API key leakage, silent fallbacks, broken invariants, data deletion risks.
   - **HIGH:** Missing test coverage, unhandled upstream exceptions, improper error masking.
   - **MEDIUM:** Architecture leakage, unnecessary complexity, formatting or style inconsistencies.
   - **LOW:** Minor refactoring ideas, cosmetic docstring improvements.

5. **Acknowledge Clean State:**
   - If no issues are found, state: `AUDIT PASSED: No issues identified.`
   - Do NOT invent artificial issues.
