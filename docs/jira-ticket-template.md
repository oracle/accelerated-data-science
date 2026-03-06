# Jira Ticket Template For Agent-Driven Changes

Use this template for tickets that the Codex agent is expected to implement.

## Required Fields

### Summary
One-line description of the bug fix or enhancement.

### Problem Statement
Describe the current behavior and why it is wrong.
Include concrete reproduction details when available.

### Desired Outcome
Describe the intended behavior after the fix.

### Acceptance Criteria
- Criterion 1
- Criterion 2
- Criterion 3

### Test Plan
Describe how the change should be validated.
Prefer concrete commands, paths, or scenarios.

## Optional Fields

### Scope / Non-Goals
Clarify what is intentionally out of scope.

### Implementation Notes
Relevant modules, classes, APIs, or design constraints.

### Risks / Rollback
Known risks, compatibility concerns, and rollback notes.

### Dependencies / Links
Related Jira issues, PRs, docs, or design references.

## Ready-For-Agent Checklist
- The ticket identifies the affected behavior clearly.
- The expected outcome is testable.
- Acceptance criteria are specific enough to validate.
- The initial validation path is described in the test plan.
- Links to dependent context are attached when needed.

## Example Skeleton
```md
Summary
Fix AQUA deployment shape recommendation fallback for missing GPU metadata.

Problem Statement
`recommend_shape` fails with a generic exception when a model record lacks GPU metadata in the fallback path.

Desired Outcome
The fallback path should return a clear validation error and keep the default recommendation path working.

Acceptance Criteria
- Missing GPU metadata returns a descriptive error.
- Existing valid model records still produce recommendations.
- Regression coverage exists for the fallback path.

Test Plan
- `python3 -m pytest tests/unitary/with_extras/aqua/test_recommend.py -k fallback`
- Run the relevant happy-path recommendation test.

Scope / Non-Goals
No UI changes. No new recommendation heuristics.

Implementation Notes
Focus on `ads/aqua/...` and adjacent tests.

Risks / Rollback
Low risk. Revert the fallback validation change if downstream parsing breaks.

Dependencies / Links
- ODSC-12345
- Design note: <link>
```
