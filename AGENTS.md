# AGENTS.md

## Project Overview
- This repository is the Oracle Accelerated Data Science SDK.
- Packaging name: `oracle_ads`
- Import and CLI module: `ads`
- It is a large Python SDK for OCI Data Science workflows, including jobs, pipelines, model lifecycle, deployments, operators, feature store, AQUA/LLM tooling, and supporting data-science utilities.

## Repository Map
- `ads/`: main Python package.
- `ads/common`, `ads/config`, `ads/cli.py`: shared auth, config, logging, serialization, OCI helpers, and CLI entrypoints.
- `ads/model`, `ads/jobs`, `ads/pipeline`, `ads/opctl`, `ads/aqua`, `ads/feature_store`, `ads/llm`: major OCI-facing product areas.
- `ads/feature_engineering`, `ads/evaluations`, `ads/hpo`, `ads/data_labeling`, `ads/text_dataset`, `ads/type_discovery`: core SDK utilities and ML workflows.
- `tests/unitary/default_setup`: minimum-dependency unit tests.
- `tests/unitary/with_extras`: unit tests requiring extras.
- `tests/integration`: integration tests for workflows and services.
- `tests/operators`: low-code operator test suites.
- `docs/source`: Sphinx source.
- `docs/build`, `docs/docs_html`, `dist`: generated outputs; do not treat them as source.
- `notebooks/`, `.demo/`, `skills/`: examples and auxiliary assets, not the primary package surface.

## Build, Install, And Test
- Base editable install: `python -m pip install -e .`
- Minimum test setup: `python -m pip install -r test-requirements.txt`
- Full development test setup: `python -m pip install -r dev-requirements.txt`
- Operator test setup: `python -m pip install -r test-requirements-operators.txt`
- Build distribution: `python -m build` or `make dist`
- Build docs: `make -C docs html`
- Live docs: `make -C docs livehtml`

## Validation Commands
- Run the smallest relevant test slice first.
- Default package validation: `python -m pytest tests/unitary/default_setup`
- Full unit validation: `python -m pytest tests/unitary`
- Non-opctl integration validation: `python -m pytest tests/integration --ignore=tests/integration/opctl`
- Operators validation: `python -m pytest tests/operators --ignore=tests/operators/forecast`
- Forecast operator validation: `python -m pytest tests/operators/forecast --ignore=tests/operators/forecast/test_explainers.py`
- Forecast explainer validation: `python -m pytest tests/operators/forecast/test_explainers.py`
- Lint and formatting gate: `pre-commit run --all-files`
- Default `pytest` config already excludes marker-based Oracle DB and thick-client tests unless explicitly requested.

## CI And Packaging Constraints
- Packaging is defined in `pyproject.toml` with Flit. `setup.py` is obsolete and should not be used for changes.
- CI covers `tests/unitary/default_setup` on Python 3.9-3.12.
- CI covers broader unit suites on Python 3.10-3.12.
- CI includes a Python 3.9 coverage job.
- CI runs separate operator and forecast workflows.
- Docs are built via Sphinx with `docs/source/conf.py`, and that config imports `ads` at build time.
- If dependencies change in `pyproject.toml`, update `THIRD_PARTY_LICENSES.txt`.

## Coding Conventions
- Preserve Python 3.8 compatibility even though CI also runs newer versions.
- Avoid 3.10+-only syntax such as `match`, `X | Y`, and built-in generics like `list[str]`.
- Follow existing package boundaries before introducing new abstractions.
- Keep public APIs typed and documented in the style already used in the package.
- Keep optional functionality behind extras and existing runtime dependency patterns; do not add unconditional imports for extra-only libraries.
- Preserve existing CLI command names and user-facing behavior unless the task explicitly requires a breaking change.
- Keep OCI auth, config, and environment-variable behavior backward compatible unless the task specifically targets that behavior.
- New Python and shell files should keep the repository’s standard header and copyright style.

## Safe Modification Rules
- Do not modify unrelated files.
- Do not hand-edit generated artifacts in `docs/build/`, `docs/docs_html/`, `dist/`, coverage output, or cache directories.
- Be careful with import-time side effects, because they can break both runtime behavior and documentation builds.
- Avoid wide refactors across multiple product areas in one patch.
- Respect existing user changes in the worktree; do not revert them unless explicitly asked.

<!-- BEGIN BEADS INTEGRATION v:1 profile:full hash:f65d5d33 -->
## Issue Tracking With bd

**IMPORTANT**: This project uses **bd (beads)** for all issue tracking. Do not use Markdown TODO lists, ad hoc task lists, or external trackers for repository work.

### Quick Start
- Check ready work: `bd ready --json`
- Create issue: `bd create "Issue title" --description="Detailed context" -t bug|feature|task|epic|chore -p 0-4 --json`
- Create discovered follow-up: `bd create "Issue title" --description="Context" -p 1 --deps discovered-from:bd-123 --json`
- Claim work: `bd update <id> --claim --json`
- Close work: `bd close <id> --reason "Completed" --json`

### Issue Types
- `bug`: something broken
- `feature`: new functionality
- `task`: implementation, refactor, tests, docs
- `epic`: large feature with subtasks
- `chore`: maintenance or tooling

### Priorities
- `0`: critical
- `1`: high
- `2`: medium
- `3`: low
- `4`: backlog

### Quality And Hygiene
- Use `--acceptance` and `--design` when creating issues if the task needs explicit success criteria.
- Use `--validate` when creating or editing issues if completeness is in doubt.
- Use `bd defer`, `bd supersede`, `bd stale`, `bd orphans`, and `bd lint` for issue hygiene when relevant.
- Use `bd human <id>` when a decision must be handed to a person.

### Auto-Sync
- `bd` writes auto-commit to Dolt history.
- Use `bd dolt pull` and `bd dolt push` when syncing issue state with the remote.
- After changing the status of any task, run `bd export --no-memories -o .beads/issues.jsonl` to refresh the exported issue snapshot.

### Non-Negotiable Rules
- Use `bd` for all task tracking.
- Always use `--json` for programmatic `bd` usage.
- Link follow-up work with `discovered-from`.
- Do not create duplicate Markdown task trackers.
- After any task status change, run `bd export --no-memories -o .beads/issues.jsonl`.
<!-- END BEADS INTEGRATION -->

## Recommended Change Workflow
1. Work on one task at a time.
2. Claim the corresponding `bd` issue before editing.
3. Read the nearest implementation, tests, and docs for the area being changed.
4. Make the smallest reviewable change that solves the task.
5. Update or add tests in the closest matching suite.
6. Run targeted validation before finishing.
7. If a finished task results in a coherent, reviewable change set, create a commit for it instead of batching it with unrelated completed work.
8. Run broader validation only when shared code changed.
9. Stop after a meaningful milestone and record follow-up work in `bd` if needed.

## Execution Rules For Codex
- Work on one task at a time.
- Make small, reviewable changes.
- Run tests before completing a task.
- Do not modify unrelated files.
- Stop after meaningful milestones.
- Commit after each finished task when it makes sense to do so as a clean, reviewable unit.
- Use `git commit --signoff` to match `CONTRIBUTING.md`.

## Session Completion
1. File `bd` issues for remaining work.
2. Run relevant quality gates.
3. Create a commit for each finished task when the resulting diff is coherent and reviewable; avoid batching unrelated finished tasks together.
4. Update and close `bd` issues as appropriate.
5. After each task status change, run `bd export --no-memories -o .beads/issues.jsonl`.
6. Sync issue state with `bd dolt push`.
7. If the task includes commit/push responsibilities, ensure git work is actually pushed and verify status afterward.
