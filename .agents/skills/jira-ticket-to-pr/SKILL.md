---
name: jira-ticket-to-pr
description: Implement a Jira ticket in this repository by validating the ticket against the local template, making the code change, adding unit tests, running targeted validation, and creating a pull request automatically when checks pass.
user-invocable: true
disable-model-invocation: false
---

# Jira Ticket To PR

Use this repo-local skill when the user provides a Jira issue key and wants the change implemented in this repository.

## Read Order
1. `AGENTS.md`
2. `docs/jira-ticket-template.md`
3. `.github/pull_request_template.md`
4. `CONTRIBUTING.md`
5. `README-development.md`
6. Jira issue details from the Jira MCP
7. The affected code and adjacent tests

## Workflow
1. Read the Jira issue via the Jira MCP.
2. Validate the issue against `docs/jira-ticket-template.md`.
3. If required fields are missing, stop and report the exact gaps.
4. Check `git status --short` and stop if unrelated changes are present.
5. Create a branch from `main` named `<jira-key>-<short-slug>`.
6. Implement the smallest correct fix.
7. Add or update the most relevant unit tests.
8. Run the narrowest relevant pytest command first.
9. Commit with `git commit --signoff`.
10. Push the branch.
11. Create the PR automatically using the repository PR template.

## Ticket Validation Rules
Required ticket content:
- Summary
- Problem statement
- Desired outcome
- Acceptance criteria
- Test plan

Optional but useful:
- Scope / non-goals
- Implementation notes
- Risks / rollback
- Dependencies / links

Do not invent missing product behavior. If the ticket is underspecified, stop.

## Validation Rules
Default to focused validation. Good examples in this repository:

```bash
python3 -m pytest tests/unitary/default_setup
python3 -m pytest tests/unitary/<target>
python3 -m pytest tests/unitary/with_extras/aqua/<target>
```

Broaden the test surface only if the change clearly warrants it.
Do not run unrelated integration suites by default.

## Branch, Commit, And PR Conventions
Use these exact formats:

```text
branch: <jira-key>-<short-slug>
commit: <JIRA-KEY>: <imperative summary>
pr title: <JIRA-KEY>: <ticket summary>
```

The PR body should include:
- Jira reference
- What changed
- Validation run
- Risks / follow-ups

## Stop Conditions
Stop and ask the user when:
- the ticket is missing required detail,
- local git state is unexpectedly dirty,
- credentials for Jira, git, or GitHub are missing,
- validation fails for a likely unrelated pre-existing reason,
- the user asks for a merge or other protected action.
