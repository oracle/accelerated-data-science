# ADS Ticket-to-PR Agent Instructions

## Purpose
Implement Jira tickets in this repository with a validate -> code -> test -> PR workflow.

This repository uses a human-reviewed merge model:
- The agent may create branches, commits, pushes, and pull requests.
- The agent must not merge pull requests.
- The agent must not transition Jira issues to `Done` unless the user explicitly asks.

## Skills To Use
Use the smallest relevant set:
1. `jira-ticket-to-pr` for ticket-driven implementation.
2. `gh-pr-workflow` for safe GitHub CLI usage in this repository.
3. `gh-address-comments` when the user asks to address PR feedback.
4. `gh-fix-ci` when the user asks to diagnose or fix failing CI.

If one of these skills is unavailable in the runtime, state that briefly and continue with equivalent staged behavior.

## Read Order
For ticket-driven work, always read in this order:
1. `AGENTS.md`
2. `docs/jira-ticket-template.md`
3. `.github/pull_request_template.md`
4. `CONTRIBUTING.md`
5. `README-development.md`
6. The Jira issue and any linked local files the ticket explicitly depends on
7. The target code and adjacent tests

## Starting Conditions
Before changing code:
- Confirm the Jira issue key and summary.
- Validate the issue against `docs/jira-ticket-template.md`.
- Check `git status --short`.
- If the working tree contains unrelated user changes, stop and ask before proceeding.
- Branch from `main` using `<jira-key>-<short-slug>` in lowercase.

## Ticket Validation Contract
Treat the ticket as ready only if all required sections are present or can be mapped clearly:
- Summary
- Problem statement
- Desired outcome
- Acceptance criteria
- Test plan

The following sections are optional but should be used when present:
- Scope / non-goals
- Implementation notes
- Risks / rollback
- Dependencies / linked issues

If required information is missing, stop and report the exact gaps. Do not guess product requirements.

## Implementation Workflow
When the user asks to implement a Jira ticket:
1. Read the Jira issue through the Jira MCP.
2. Validate the issue against the ticket template.
3. Summarize the implementation plan and assumptions.
4. Create a feature branch.
5. Implement the smallest correct change.
6. Add or update the most relevant unit tests.
7. Run the most targeted validation first.
8. Expand validation only as needed for confidence.
9. Commit with `--signoff`.
10. Push the branch and create the pull request automatically when validation passes.

If validation fails:
- Do not create the pull request unless the user explicitly asks to proceed anyway.
- Report the failing command and the likely reason.

## Test Expectations
Prefer the narrowest useful test command first.
Typical defaults in this repository:
- `python3 -m pytest tests/unitary/default_setup` for default-setup changes
- `python3 -m pytest tests/unitary/<target>` for focused unit changes
- `python3 -m pytest tests/unitary/with_extras/aqua/<target>` for AQUA changes

Use broader suites only when the change crosses modules or the narrow test surface is not sufficient.
Do not run unrelated integration suites by default.

## Git And PR Rules
Required branch and PR conventions:
- Branch name: `<jira-key>-<short-slug>`
- Commit message subject: `<JIRA-KEY>: <imperative summary>`
- PR title: `<JIRA-KEY>: <ticket summary>`

PR body must include:
- Jira reference
- What changed
- Validation run
- Risks or follow-ups

Always use `git commit --signoff` because this repository requires signed-off commits.
Never push directly to `main`.
Never use force-push unless the user explicitly asks.

## Safe GitHub CLI Scope
Allowed by default:
- `gh auth status`
- `gh pr status`
- `gh pr view`
- `gh pr checks`
- `gh pr create`
- `gh pr edit`
- `gh pr comment`

Disallowed unless the user explicitly asks:
- `gh pr merge`
- `gh repo delete`
- `gh repo archive`
- `gh secret *`
- `gh workflow run`
- write operations through `gh api`

## Review Follow-Up
When the user asks to address PR comments:
- Read unresolved review comments first.
- Apply only the requested changes.
- Re-run the most relevant validation.
- Push updates to the existing branch.
- Comment on the PR only if the user asks or the workflow clearly benefits.

## Refusal / Stop Conditions
Stop and ask the user when:
- the Jira ticket is underspecified,
- the working tree is unexpectedly dirty,
- required credentials for Jira, git, or GitHub are missing,
- the change requires a destructive git operation,
- the safest test command fails for an unrelated pre-existing reason.
