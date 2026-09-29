---
name: secret-scanner
description: Secret and credential scanner designed for Content Factory under Windows PowerShell. Scans git diff, staged files, and workspace changes for API keys, bearer tokens, private keys, and tracked .env files.
---

<!-- Content Factory Custom Tooling -->

# Secret Scanner Protocol

Use this skill to scan code changes for accidentally committed API keys, tokens, or private credentials before creating a checkpoint.

## Target Secret Signatures

The scanner searches for common cloud AI provider key patterns and credential formats:
- **Google Gemini:** `AIza[0-9A-Za-z_-]{35}`
- **OpenAI:** `sk-[A-Za-z0-9]{20,}` or `sk-proj-[A-Za-z0-9_-]{20,}`
- **Anthropic Claude:** `sk-ant-[A-Za-z0-9_-]{20,}`
- **Groq:** `gsk_[A-Za-z0-9_-]{20,}`
- **OpenRouter:** `sk-or-[A-Za-z0-9_-]{20,}`
- **Generic Headers:** `Authorization:\s*Bearer\s+[A-Za-z0-9_.-]+`
- **API Key Headers:** `x-api-key:\s*[A-Za-z0-9_.-]+`
- **Private Keys:** `-----BEGIN\s+(RSA|EC|DSA|OPENSSH|PRIVATE)\s+KEY-----`
- **JSON Web Tokens:** `eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}`

## Audit Procedure

Run the following checks from the project root (`d:\Content_Factory`):

### 1. Verify `.env` File Is Untracked
```powershell
git ls-files .env
```
- **Requirement:** Output MUST be completely blank. If `.env` appears, it is tracked in Git and must be removed from the index immediately (`git rm --cached .env`).

### 2. Check for Tracked Secret Files
```powershell
git ls-files "*secret*" "*credential*" "*.pem" "*.key"
```
- Confirm no private key or credential store files are tracked.

### 3. Scan Git Diff for Secret Patterns
Search the current uncommitted changes (or staged changes) for provider key signatures:
```powershell
git diff -U0 | Select-String -Pattern 'AIza[0-9A-Za-z_-]{20,}', 'sk-[a-zA-Z0-9]{20,}', 'gsk_[a-zA-Z0-9]{20,}', 'sk-ant-[a-zA-Z0-9]{20,}', 'sk-or-[a-zA-Z0-9]{20,}'
```
- Ensure any match is verified. If a line represents a mock test placeholder (e.g. `test-dummy-key`), verify it is explicitly fake and not an actual active credential.

### 4. Scan Headers and Bearer Tokens
```powershell
git diff -U0 | Select-String -Pattern 'Bearer\s+[a-zA-Z0-9_\-\.]{20,}', 'x-api-key'
```
- Verify that authorization headers are parameterized via configuration/env variables, never hardcoded.

## Reporting & Redaction Guidelines
- **NEVER print full detected secrets in chat or logs.**
- Redact findings using mask notation: `sk-proj-...[REDACTED]...4f` or `AIzaSy...[REDACTED]`.
- Report file path and line number only:
  ```text
  [ALERT] Potential secret found in app/services/ai/providers/foo.py:Line 42
  Pattern: OpenAI Key Signature (sk-proj-...)
  Action Required: Replace with environment variable reference.
  ```
- If clean, state:
  `SECRET AUDIT PASSED: No credentials or tracked .env files detected.`
