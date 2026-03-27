# Using The Jira Ticket To PR Agent In Codex

## Prerequisites
- Start Codex from the repository root.
- Ensure Jira MCP is configured and authenticated.
- Ensure `gh auth status` succeeds.
- Restart Codex after installing any new global skills so they are discovered in a fresh session.

## Typical Flow
1. Open the ADS repository in Codex.
2. Provide the Jira issue key and ask Codex to implement it.
3. Codex reads `AGENTS.md`, validates the ticket against `docs/jira-ticket-template.md`, and stops if required sections are missing.
4. If the ticket is ready, Codex creates a branch, implements the change, adds or updates unit tests, and runs targeted validation.
5. When validation passes, Codex commits with `--signoff`, pushes the branch, and creates the PR automatically.
6. Later, ask Codex to address PR comments or inspect failing checks when needed.

## Example Prompts
```text
Use jira-ticket-to-pr and implement ADS-12345.

Read ADS-12345, validate it against the local ticket template, implement the change, add the right unit tests, run focused pytest coverage, and open the PR if validation passes.

Use gh-pr-workflow to inspect the checks on PR 1356.

Use gh-address-comments to address the unresolved review comments on PR 1356.

Use gh-fix-ci to investigate the failing GitHub checks on PR 1356.
```

## Additional Examples
- Sample implementation-ready ticket: `.agents/examples/python-3.14-support-ticket.md`

## Expected Stop Conditions
Codex should stop and ask instead of guessing when:
- the Jira ticket is missing required implementation detail,
- the local worktree is unexpectedly dirty,
- validation fails for a likely unrelated pre-existing reason,
- credentials for Jira, git, or GitHub are missing,
- a destructive git or GitHub action would be required.
