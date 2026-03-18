# Sample Jira Ticket: Python 3.14 Support Planning

This example follows `docs/jira-ticket-template.md` and is intended as a copyable starting point for implementation-ready Jira issues.

## Summary
Plan and implement ADS support for Python 3.14 across packaging, CI, and targeted runtime compatibility validation.

## Problem Statement
ADS currently documents and validates a bounded set of Python versions. As Python 3.14 becomes available, the repository will need an explicit readiness ticket that captures packaging updates, CI coverage, dependency compatibility checks, and validation scope. Without a structured ticket, the implementation effort risks missing compatibility gaps or producing incomplete validation.

## Desired Outcome
ADS has an implementation-ready Jira issue that clearly scopes Python 3.14 support work, defines validation expectations, and identifies the areas that need compatibility review before the version can be claimed as supported.

## Acceptance Criteria
- Packaging metadata and documented supported Python versions are reviewed and updated where necessary.
- CI or local validation scope for Python 3.14 is identified explicitly.
- Dependency compatibility checks are called out for packages likely to lag new Python support.
- The ticket is specific enough that an engineering agent can implement the work without guessing missing requirements.

## Test Plan
- Run the most relevant unit test suites under Python 3.14 after environment setup succeeds.
- Validate package install and editable install flows with Python 3.14.
- Verify any version-specific CI or local workflow changes proposed by the implementation.

## Scope / Non-Goals
- This example ticket does not itself add Python 3.14 support.
- This example ticket does not guarantee every optional dependency is immediately compatible.
- This example ticket does not replace deeper dependency triage if third-party packages block adoption.

## Implementation Notes
- Review `pyproject.toml`, test workflows under `.github/workflows/`, and `README-development.md`.
- Check for version-pinned dependencies that may block Python 3.14.
- Keep the first implementation slice small and validation-focused.

## Risks / Rollback
- Third-party dependencies may not support Python 3.14 on the desired timeline.
- CI image availability may lag interpreter release timing.
- If compatibility blockers are found, revert support claims and narrow the ticket to investigation only.

## Dependencies / Links
- `docs/jira-ticket-template.md`
- `.agents/examples/jira-ticket-to-pr.md`
- `pyproject.toml`
- `.github/workflows/`
