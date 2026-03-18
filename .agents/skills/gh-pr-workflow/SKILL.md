---
name: gh-pr-workflow
description: Safe GitHub CLI workflow for this repository. Covers branch checks, PR creation, PR inspection, checks, and review follow-up while blocking merge and other high-risk GitHub operations by default.
user-invocable: true
disable-model-invocation: false
---

# GitHub PR Workflow

Use this skill when the user wants GitHub CLI help for pull requests in this repository.

## Allowed Commands
Use these by default when needed:

```bash
gh auth status
gh pr status
gh pr view <pr-number>
gh pr checks <pr-number>
gh pr create --title "<JIRA-KEY>: <summary>" --body-file <file>
gh pr edit <pr-number> --body-file <file>
gh pr comment <pr-number> --body-file <file>
```

## Disallowed Commands
Do not run these unless the user explicitly asks:

```bash
gh pr merge
gh repo delete
gh repo archive
gh secret *
gh workflow run
gh api <write operation>
```

Do not force-push unless the user explicitly asks.
Do not create or edit release artifacts as part of PR handling.

## PR Creation Rules
Before creating a PR:
1. Ensure the current branch is not `main`.
2. Ensure the working tree is clean enough to describe exactly what is being proposed.
3. Ensure at least one relevant validation command has been run.
4. Use `.github/pull_request_template.md` as the body source.
5. Reference the Jira key in the title and body.

If validation failed, stop and ask whether the user wants to proceed anyway.

## Review Follow-Up
When the user asks to address review comments:
1. Read unresolved review comments first.
2. Apply only the requested changes.
3. Re-run focused validation.
4. Push updates to the same branch.
5. Update the PR body only if the scope or validation section changed materially.

If the user explicitly asks to work through review comments, prefer the `gh-address-comments` skill if available.
If CI is failing and the user wants help, prefer the `gh-fix-ci` skill if available.
