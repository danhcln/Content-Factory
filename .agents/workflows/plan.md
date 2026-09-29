---
description: Plan an architectural change or feature implementation for Content Factory without touching code.
argument-hint: "[phase or feature description]"
---

# Plan Workflow

Execute structured implementation planning for Content Factory before making any code modifications.

## Procedure

1. **Invoke the Planning Skill:**
   Activate and follow the instructions in `.agents/skills/plan/SKILL.md`.

2. **Ground in Current Repository State:**
   - Verify active branch, HEAD commit, and working tree status.
   - Inspect existing architecture patterns in `app/services/ai/`, `app/routes/`, and `app/templates/`.

3. **Check Project Invariants:**
   - Review `.agents/rules/00-content-factory-invariants.md`.
   - Ensure the proposed feature does NOT violate Research single-call rules, Multi-AI BYOK constraints, or data protection rules.

4. **Output Implementation Plan:**
   - Restate requirements and scope boundaries.
   - Detail step-by-step phases, affected files, and required mock tests.
   - Identify risks, edge cases, and quota implications.

5. **Stop & Await Confirmation:**
   - DO NOT modify any code.
   - Ask for user review and approval before proceeding to implementation.
