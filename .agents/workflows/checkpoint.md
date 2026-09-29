---
description: Execute safe pre-checkpoint verification (tests, diff, secret scan, stash check) without auto-committing.
argument-hint: "[optional checkpoint note]"
---

# Safe Checkpoint Workflow

Performs a rigorous pre-commit audit and verification check for Content Factory.

## Critical Constraint
This workflow **NEVER** runs `git commit` or `git push` automatically. It audits readiness and concludes with `READY FOR CHECKPOINT: YES/NO`.

## Procedure

1. **Invoke Core Verification Skills:**
   - Execute `.agents/skills/safe-checkpoint/SKILL.md`.
   - Execute `.agents/skills/secret-scanner/SKILL.md` to scan unstaged/staged diffs.
   - Execute `.agents/skills/ai-regression-check/SKILL.md` to verify all regression test suites pass offline.

2. **Verify Repository State:**
   - Confirm branch matches intended feature branch (`feature/oauth-social-connect`).
   - Confirm `git ls-files .env` returns nothing.
   - Verify `git stash list` confirms `stash@{0}` remains intact.
   - Confirm production database (`content.db`) and media folders are untouched.

3. **Inspect Git Diff:**
   - Run `git diff --stat` and review exact lines changed.
   - Ensure only intentional implementation files are present.

4. **Report Verification Summary:**
   - Provide test pass/fail counts.
   - State secret audit result (Pass/Fail).
   - Conclude clearly:
     ```text
     ==================================================
     READY FOR CHECKPOINT: YES / NO
     ==================================================
     ```
   - Await explicit user authorization before creating any Git commit.
