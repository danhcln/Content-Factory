---
name: safe-checkpoint
description: Comprehensive pre-commit verification protocol for Content Factory. Audits git branch, working tree, diff, test suites, secret exposure, database safety, and stash preservation without auto-committing.
---

<!-- Content Factory Custom Tooling -->

# Safe Checkpoint Protocol

Use this skill before creating any Git commit or checkpoint. It acts as an automated safety gate.

## Critical Rule
**This skill NEVER runs `git commit` or `git push` automatically.** It verifies readiness and reports whether the repository is ready for a checkpoint.

## 10-Point Verification Checklist

Execute the following checks in order:

### 1. Verify Active Branch
```powershell
git branch --show-current
```
- Confirm output matches the intended working branch (e.g., `feature/oauth-social-connect`).
- Never commit directly to an unintended branch.

### 2. Verify Base HEAD
```powershell
git rev-parse HEAD
git rev-parse origin/feature/oauth-social-connect
```
- Verify relationship with the starting checkpoint.

### 3. Verify Git Status & Working Tree
```powershell
git status
```
- Confirm only intended new or modified files are present.
- Confirm untracked temporary files, virtualenvs, or IDE files are excluded.

### 4. Inspect Full Git Diff
```powershell
git diff --stat
git diff
```
- Review every modified line for unexpected edits, commented-out dead code, or unintended side-effects.

### 5. Run Relevant Focused Tests
```powershell
.\.venv\Scripts\python.exe <test_relevant_to_phase>.py
```
- Confirm 100% pass rate on newly written or updated tests.

### 6. Run Full AI Regression Suite
Execute the core offline regression tests:
```powershell
.\.venv\Scripts\python.exe test_ai_architecture_phase1.py
.\.venv\Scripts\python.exe test_ai_providers_phase2.py
.\.venv\Scripts\python.exe test_research_efficiency.py
.\.venv\Scripts\python.exe test_gemini_reliability.py
```
- Every test must pass with ZERO live API calls.

### 7. Scan for Credential / Secret Leaks
Invoke the `.agents/skills/secret-scanner/SKILL.md` procedure:
- Scan git diff for raw API keys (`AIza`, `sk-`, `gsk_`), Bearer tokens, and private headers.

### 8. Confirm `.env` Is Not Tracked
```powershell
git ls-files .env
```
- Output **MUST BE EMPTY**. If `.env` is tracked, STOP immediately and unstage/remove from Git index.

### 9. Verify Production Asset Integrity
- Verify that SQLite databases (`content.db`) are NOT staged or corrupted.
- Verify asset folders (`downloads/`, `media/`, audio/video folders) are preserved.

### 10. Confirm Stash Preservation
```powershell
git stash list
```
- Ensure previous stashes (e.g. `stash@{0}`) remain intact and were not dropped or popped.

## Conclusion Report

Conclude the audit with a clear determination:
```text
==================================================
CHECKPOINT AUDIT RESULT:
- Branch: OK
- Tests: ALL PASSED (0 live API calls)
- Secrets: NONE DETECTED
- .env untracked: CONFIRMED
- Assets intact: CONFIRMED
- Stash preserved: CONFIRMED

READY FOR CHECKPOINT: YES / NO
==================================================
```
If **YES**, wait for the user to provide the exact commit message and authorization to commit.
