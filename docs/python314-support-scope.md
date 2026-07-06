# ADS Python 3.14 Support Scope

This document records the initial support boundary for ODSC-88493. It is a
planning artifact for validation work only; it does not advertise Python 3.14
support in package metadata.

## In Scope

- Core ADS package installation from `pyproject.toml` on Python 3.14, including
  source distribution and wheel build/install validation.
- Base SDK workflows covered by `tests/unitary/default_setup`, including
  common configuration, authentication helpers, OCI client construction, data
  helpers, model metadata, jobs, pipelines, telemetry, feature engineering, and
  other modules that import without optional extras.
- Development and test dependencies required to run default and full unit test
  suites on Python 3.14.
- Optional dependency groups declared in `[project.optional-dependencies]` that
  are part of current ADS service-facing usage:
  - `data`
  - `viz`
  - `notebook`
  - `opctl`
  - `onnx`
  - `tensorflow`
  - `torch`
  - `boosted`
  - `huggingface`
  - `text`
  - `spark`
  - `bds`
  - `llm`
  - `aqua`
- Low-code operator extras and their runtime requirements:
  - `forecast`
  - `anomaly`
  - `recommender`
  - `regression`
  - `pii`
- Model artifact and model runtime workflows, including `runtime.yaml`
  generation, introspection, serialization, local artifact validation, and
  framework-specific model classes for generic, scikit-learn, XGBoost, LightGBM,
  TensorFlow, PyTorch, ONNX, Hugging Face, and Spark pipeline models.
- Job, pipeline, and Data Flow runtime builders that reference service conda or
  custom conda environments.
- Notebook tooling needed by ADS notebooks, job notebook runtimes, and AQUA
  Jupyter server integration.
- GitHub Actions coverage for Python 3.14, or a staged CI validation note when
  runner images or dependency availability block immediate enablement.
- License metadata review for any dependency changes made to support Python
  3.14.

## Service Conda Validation Targets

Python 3.14 support should be validated against the ADS package surface expected
in new OCI Data Science service conda environments. At minimum, later validation
Beads should map each required service conda pack to one of these buckets:

- Core ADS only.
- Notebook and visualization tooling.
- General ML and model serialization/runtime tooling.
- ONNX runtime and conversion tooling.
- TensorFlow runtime tooling.
- PyTorch runtime tooling.
- Spark/Data Flow tooling.
- Forecast and anomaly operator tooling.
- PII, recommender, and regression operator tooling.
- AQUA and LLM tooling.

If a required service conda pack cannot be made installable on Python 3.14 during
this work, record the blocker and the follow-up ticket instead of silently
dropping the pack from support.

## Explicitly Out of Scope

- Adding the `Programming Language :: Python :: 3.14` classifier or otherwise
  advertising Python 3.14 support before install, build, CI, unit test, and
  runtime validation are complete.
- Guaranteeing third-party ML framework functionality that does not publish
  Python 3.14-compatible releases or wheels. These cases should be tracked as
  deferred support with explicit follow-up tickets.
- Validating every historical service conda slug referenced in examples or
  documentation. Validation should focus on current and planned service conda
  environments required for Python 3.14.
- End-to-end OCI service integration tests that require live tenancy resources,
  unless they are already part of the agreed validation environment.
- Changing public ADS APIs solely for Python 3.14 support unless a later
  compatibility audit identifies a required code change.

## Validation Handoff

Later Beads should use this scope to decide whether dependency changes,
metadata updates, CI matrix changes, and runtime validation are complete. Known
incompatibilities should be captured with the affected extra or workflow, the
blocking dependency or behavior, and the follow-up ticket when support is
deferred.

## Dependency Audit Notes

Audit date: 2026-07-06. Current PyPI metadata was checked for high-risk
compiled packages and exact ADS pins or caps.

### Core Install

- `numpy`, `pandas`, `scipy`, `matplotlib`, and `scikit-learn` publish current
  Python 3.14-compatible releases and wheels, but ADS still allows very old
  lower bounds. Python 3.14 validation should constrain the resolver to modern
  versions instead of relying on broad lower bounds.
- `pandas>=2.2.0,<3.0.0` blocks current `pandas` 3.x, even though current
  pandas publishes Python 3.14 wheels. Keep the cap until ADS pandas behavior is
  validated, but explicitly test the latest compatible pandas 2.x resolver on
  Python 3.14.
- `oci`, `ocifs`, `pydantic`, `httpx`, `requests`, `PyYAML`, and other pure
  Python or widely maintained core dependencies do not show an immediate
  metadata-level Python 3.14 blocker. They still need install validation in the
  target service conda environment.

### Extras Requiring Version Changes or Markers

- `boosted`, `notebook`, and `onnx` cap `scikit-learn` below 1.6. Current
  `scikit-learn` Python 3.14 wheels are available only in newer releases. Add a
  Python 3.14 marker that selects a validated newer scikit-learn version, or
  defer those extras with a follow-up ticket.
- `onnx` selects `onnx~=1.17.0` and `onnxruntime~=1.22.0` for Python 3.12+.
  Those exact ranges do not provide Python 3.14 wheels. Current `onnx` and
  `onnxruntime` releases do, so this extra needs Python 3.14-specific version
  markers after model serialization tests pass.
- `onnx` also pins `xgboost<=1.7`, which excludes current Python 3.14-ready
  xgboost metadata. Relax or split this cap for Python 3.14 after ONNX
  conversion tests confirm compatibility.
- `tensorflow` does not currently publish Python 3.14 wheels. Keep TensorFlow
  model support out of the Python 3.14 advertised scope until upstream wheels
  are available or a service-conda-provided build is documented.
- `text` requires `spacy>=3.4.2,<3.8`; `pii` pins `spacy==3.6.1` and
  `spacy-transformers==1.2.5`. Current spaCy supports Python 3.14 only in the
  newer 3.8 line, so `text` and `pii` need marker-based upgrades or deferred
  support tickets.
- `data` caps `sqlalchemy<=1.4.46`, which predates Python 3.14 wheel coverage.
  Validate ADS database access against SQLAlchemy 2.x for Python 3.14, or add a
  follow-up if the API migration is too large for this support pass.
- `geo` uses `geopandas<1.0.0` and `fiona<=1.9.6`. The current cap predates
  Python 3.14 wheels for Fiona. Treat `geo` as blocked until newer Fiona and
  GeoPandas constraints are validated together.
- `aqua` and `testsuite` pin Notebook 6.x (`notebook>=6.4,<=6.6` and
  `notebook==6.4.12`). Current Notebook 7.x is the Python 3.14-ready line, so
  AQUA and tests need either Notebook 7 validation or a staged follow-up.
- `optuna==2.9.0` is old relative to the current Optuna line. Validate resolver
  behavior on Python 3.14 before including the `optuna` extra.

### Operator Extras Requiring Follow-up

- `forecast` pins `numpy<2.0.0` and includes `prophet==1.1.7`,
  `neuralprophet>=0.7.0`, `pmdarima`, `pytorch-lightning==2.5.5`, and
  `xgboost<3.0.0`. Current `neuralprophet` metadata excludes Python 3.14, and
  `prophet` does not publish Python 3.14 wheels. Treat `forecast` as blocked
  until the service conda dependency set identifies compatible replacements or
  defers support.
- `anomaly` pins `salesforce-merlion[all]==2.0.4`, `rrcf==0.4.4`, and
  `scikit-learn<1.6.0`. This extra is blocked by the scikit-learn cap and needs
  a Merlion dependency audit before Python 3.14 support can be claimed.
- `recommender` depends on `scikit-surprise`. Current scikit-surprise publishes
  Python 3.14 wheels, but this extra is commented out in
  `test-requirements-operators.txt`; keep it out of required support unless the
  service conda comparison Bead makes it mandatory.
- `pii` is blocked by the spaCy 3.6 pin and old spaCy transformer stack. It
  needs a text-stack upgrade plan or explicit deferred support ticket.
- `regression` inherits the `forecast` blockers through `oracle_ads[opctl,forecast]`.

### Test and Development Dependencies

- `test-requirements.txt` uses unpinned pytest, coverage, xdist, ruff, and
  setuptools packages, which should resolve on Python 3.14 but need an actual
  environment creation test.
- `dev-requirements.txt` installs all major extras, so it will fail on Python
  3.14 until the blocked extras above are guarded by markers, upgraded, or
  moved to deferred support.
- `testsuite` includes compiled packages with current Python 3.14 support
  (`fastparquet`, `pyarrow`, `tables`) and pure Python packages, but the pinned
  `notebook==6.4.12` should be replaced or marker-split for Python 3.14.

### Recommended Follow-up Actions

- Add Python 3.14-specific dependency markers in `pyproject.toml` only after
  install and targeted behavioral tests validate each upgraded dependency.
- Track deferred extras explicitly. At minimum, TensorFlow, forecast,
  anomaly, PII, geo, and Notebook 6-based AQUA support need owner decisions
  before ADS advertises Python 3.14 broadly.
- Re-run the dependency audit against service conda constraints in the next
  Bead, because service-provided builds may differ from PyPI wheel availability.
- Update `THIRD_PARTY_LICENSES.txt` if dependency upgrades introduce new
  packages or materially different versions that require license metadata
  refresh.

## Service Conda Constraint Comparison

Comparison date: 2026-07-06. No checked-in manifest for planned Python 3.14 OCI
Data Science service conda environments was available in this workspace. The
comparison below uses the repo-visible service pack index, documented service
conda slugs, operator manifests, and operator `environment.yaml` templates.
When planned Python 3.14 service manifests become available, this section must
be reconciled against their exact package pins.

### Repo-Visible Service Conda Baseline

- `tests/unitary/with_extras/model/index.json` lists historical service packs
  for Python 3.6, 3.7, 3.8, and 3.9. It does not include Python 3.14 service
  conda packs.
- Current docs and tests reference service slugs such as `dataexpl_p37_cpu_v3`,
  `generalml_p38_cpu_v1`, `tensorflow28_p38_cpu_v1`,
  `pytorch110_p38_cpu_v1`, `pytorch20_p39_gpu_v2`,
  `pytorch24_p310_gpu_x86_64_v1`, and `pyspark32_p38_cpu_v2`. These are not
  Python 3.14 targets and should not be used as evidence that ADS is ready on
  Python 3.14.
- Low-code operator service manifests reference Python 3.11 packs, for example
  `forecast_p311_cpu_x86_64_v10` in the checked-in `MLoperator` and
  `forecast_p311_cpu_x86_64_v13` in the forecast operator docs. These are the
  closest current service-facing operator constraints visible in the repo.
- Operator `environment.yaml` templates still create Python 3.8 or Python 3.9
  local environments for forecast, anomaly, recommender, regression, and PII.
  They must be refreshed before they can represent Python 3.14 support.

### PyPI vs Service Conda Mismatches

| Surface | Repo-visible service constraint | PyPI / ADS constraint | Decision |
| --- | --- | --- | --- |
| Core ADS service packs | No Python 3.14 service manifest is checked in. Historical packs bundle older ADS and Python 3.6-3.9. | Core PyPI dependencies mostly have Python 3.14-ready current releases, but ADS has broad lower bounds. | Service environment must provide an explicit Python 3.14 manifest; ADS should add Python 3.14 markers only after install validation against that manifest. |
| General ML / model serialization | Historical `generalml` and `dataexpl` packs include older pandas, scikit-learn, xgboost, lightgbm, TensorFlow, and ADS versions. | ADS caps scikit-learn below the current Python 3.14-ready line in several extras and caps pandas below 3.x. | Both sides likely need changes: service packs need modern Python 3.14-compatible ML pins, while ADS needs marker-split constraints for scikit-learn and other compiled packages. |
| ONNX runtime | Historical `onnx110_p37/p38/p39` and `onnx17_p37` packs are not Python 3.14 packs. | ADS selects `onnx~=1.17.0` and `onnxruntime~=1.22.0` for Python 3.12+, neither of which has Python 3.14 wheels. | ADS should update ONNX markers for Python 3.14 after runtime/model serialization tests; service pack should align with those validated ONNX and onnxruntime versions. |
| TensorFlow runtime | Docs reference TensorFlow service packs such as `tensorflow28_p38_cpu_v1` and `tensorflow26_p37_cpu_v2`. | Current TensorFlow PyPI metadata does not publish Python 3.14 wheels. | Service environment must either provide a supported TensorFlow build or TensorFlow must be explicitly deferred; ADS should not advertise TensorFlow model support on Python 3.14 from PyPI alone. |
| PyTorch runtime | Docs reference p38-p310 PyTorch service packs. | Current `torch` and `torchvision` publish Python 3.14 artifacts, though `torchvision` excludes Python 3.14.1 specifically. | Service pack should pin a Python 3.14 patch version and matching torch/torchvision pair; ADS likely does not need a PyPI constraint change unless validation finds API differences. |
| Spark / Data Flow / BDS | Docs reference p37/p38 PySpark service packs such as `pyspark30_p37_cpu_v5` and `pyspark32_p38_cpu_v2`. | ADS declares only `pyspark>=3.0.0`; Spark runtime compatibility depends heavily on Java and service runtime constraints. | Service environment should own Spark/PySpark version selection; ADS should validate builders and BDS workflows against the selected p314 Spark pack. |
| Forecast operator | Service manifests/docs currently show Python 3.11 forecast packs; local template still says Python 3.8. | ADS `forecast` includes packages that block or risk Python 3.14, including `neuralprophet`, `prophet`, `numpy<2.0.0`, and `xgboost<3.0.0`. | Treat Python 3.14 forecast support as deferred unless the service team supplies a compatible forecast pack. ADS should not force PyPI-only support ahead of service pack readiness. |
| Anomaly operator | Local template says Python 3.8; no p314 service manifest is checked in. | ADS `anomaly` pins `salesforce-merlion[all]==2.0.4` and `scikit-learn<1.6.0`. | ADS needs a Merlion/scikit-learn upgrade plan or a deferred support ticket; service pack cannot rely on current ADS constraints for p314. |
| PII operator | Local template says Python 3.9; no p314 service manifest is checked in. | ADS `pii` pins `spacy==3.6.1` and `spacy-transformers==1.2.5`, while Python 3.14-ready spaCy is in the 3.8 line. | ADS should own a marker-split or upgrade plan for the text stack; otherwise service p314 PII support should be deferred. |
| Recommender / regression operators | Recommender local template says Python 3.9; regression local template says Python 3.8. | `scikit-surprise` has current Python 3.14 wheels, but regression inherits forecast blockers. | Recommender may be service-pack feasible after validation; regression should follow the forecast decision. |
| AQUA / notebook tooling | No p314 AQUA service pack manifest is checked in. | ADS `aqua` and `testsuite` use Notebook 6.x, while current Python 3.14-ready notebook metadata is Notebook 7.x. | ADS should validate or defer Notebook 7 migration before service p314 AQUA support is claimed. |

### Required Service Conda Inputs

Before package metadata can advertise Python 3.14 support, obtain or generate
the planned service conda manifests for these buckets:

- Core ADS / data exploration.
- General ML and model serialization.
- ONNX model runtime.
- TensorFlow model runtime, or an explicit TensorFlow deferral.
- PyTorch model runtime.
- Spark/Data Flow and BDS.
- Forecast and anomaly operators.
- PII, recommender, and regression operators.
- AQUA / notebook / LLM tooling.

Each manifest should include the Python patch version, architecture, conda
channels, pip packages, exact pinned package versions, and whether ADS is
preinstalled or installed from the wheel under test.

### Handoff Decisions

- If service conda pins are older than the Python 3.14-compatible PyPI releases
  identified above, the service environment should change first; ADS should not
  encode constraints that are impossible to satisfy in the service pack.
- If ADS caps or pins prevent installation of otherwise service-approved Python
  3.14 packages, ADS should add Python-version markers in `pyproject.toml` after
  targeted tests pass.
- If neither PyPI nor service conda provides a compatible dependency stack, the
  affected extra or workflow should be excluded from the advertised Python 3.14
  support scope and tracked with a follow-up ticket.

## Package Metadata Update Status

Metadata review date: 2026-07-06. Python 3.14 package metadata is intentionally
not updated yet because dependency resolution, Python 3.14 service conda
manifests, CI coverage, unit tests, model artifact/runtime validation, and
selected optional-extra validation are not complete.

### Current Metadata Decision

- Do not add `Programming Language :: Python :: 3.14` to `pyproject.toml` in the
  current validation state.
- Keep `requires-python = ">=3.8"` unchanged. The Python 3.14 support work does
  not require dropping older supported Python versions unless later dependency
  upgrades make that unavoidable.
- Do not update dependency markers or optional-extra constraints until a Python
  3.14 install/build Bead proves the chosen resolver set and targeted runtime
  tests pass.
- Do not update `THIRD_PARTY_LICENSES.txt` yet because this Bead does not change
  dependency versions. Refresh license metadata when dependency upgrades are
  actually made.

### Metadata Changes Gated on Validation

After validation is complete, make the following `pyproject.toml` updates in a
single dependency-aware change:

- Add `Programming Language :: Python :: 3.14` only after core install, build,
  CI, unit-test, service-conda, and runtime validation pass for the advertised
  support scope.
- Add Python 3.14-specific markers for compiled dependencies whose currently
  selected ranges do not provide Python 3.14 wheels, including scikit-learn,
  ONNX, onnxruntime, xgboost, spaCy, SQLAlchemy, Notebook, and geo stack
  dependencies where those extras remain in scope.
- Add exclusions or markers for extras that remain unsupported on Python 3.14,
  rather than allowing unresolved installs. Current candidates for deferral are
  TensorFlow, forecast, anomaly, PII, geo, and Notebook 6-based AQUA support.
- Keep optional extras aligned with service conda manifests. If service conda
  owns a runtime stack such as Spark, TensorFlow, or forecast, ADS metadata
  should match the service-approved dependency set or clearly exclude that extra
  from Python 3.14 support.
- Refresh `THIRD_PARTY_LICENSES.txt` for any dependency version changes or new
  packages introduced while adding Python 3.14 markers.

### Pre-Merge Metadata Checklist

- `python3.14 -m pip install .` succeeds for core ADS.
- Source distribution and wheel build, then wheel install, succeed on Python
  3.14.
- Required service conda manifests have been compared against the selected ADS
  dependency ranges.
- Default unit tests pass on Python 3.14.
- Full or scoped unit tests pass for each advertised extra.
- Model artifact/runtime validation accepts Python 3.14 for the supported model
  framework set.
- Deferred extras have follow-up tickets and are excluded or documented before
  publishing the classifier.

## CI Coverage Plan

CI review date: 2026-07-06. Python 3.14 should not be added to the existing
GitHub Actions matrices until dependency markers, install/build validation, and
service-conda decisions are complete. Adding it now would create expected
failures because the full-test setup installs `dev-requirements.txt`, which
pulls extras already identified as blocked or unresolved on Python 3.14.

### Current Workflow Coverage

- `.github/workflows/run-unittests-default_setup.yml` runs default setup tests
  on Python 3.9, 3.10, 3.11, and 3.12.
- `.github/workflows/run-unittests-py310-py311.yml` runs broad unit tests on
  Python 3.10, 3.11, and 3.12 using `.github/workflows/test-env-setup/action.yml`.
- `.github/workflows/run-unittests-py39-cov-report.yml` runs coverage-oriented
  broad unit tests on Python 3.9.
- `.github/workflows/run-operators-unit-tests.yml` runs non-forecast operator
  tests on Python 3.10 and 3.11.
- `.github/workflows/run-forecast-unit-tests.yml` and
  `.github/workflows/run-forecast-explainer-tests.yml` run forecast operator
  tests on Python 3.10 and 3.11.

### Staged Python 3.14 Enablement

1. Enable core/default validation first by adding Python 3.14 to
   `.github/workflows/run-unittests-default_setup.yml` after Beads for core
   install/build and default tests pass on Python 3.14.
2. Add a temporary CI gate that runs only core install, wheel install, and
   `tests/unitary/default_setup` on Python 3.14. Keep it separate from
   `dev-requirements.txt` until optional extras are marker-split or deferred.
3. Add Python 3.14 to `.github/workflows/run-unittests-py310-py311.yml` only
   after `dev-requirements.txt` is Python 3.14-safe or the workflow has a
   Python 3.14-specific dependency setup that excludes deferred extras.
4. Add Python 3.14 to model artifact/runtime validation jobs after ONNX,
   PyTorch, TensorFlow, Spark, and framework-specific support decisions are
   resolved. TensorFlow should remain excluded unless a Python 3.14-compatible
   PyPI or service-conda build is available.
5. Add Python 3.14 to operator workflows only after service-conda manifests and
   ADS extras are ready:
   - `.github/workflows/run-operators-unit-tests.yml` for non-forecast operators.
   - `.github/workflows/run-forecast-unit-tests.yml` for forecast.
   - `.github/workflows/run-forecast-explainer-tests.yml` for forecast explainers.

### Temporary Gates and Exit Criteria

- Do not mark Python 3.14 as required CI until the default setup workflow passes
  reliably and the package metadata decision has been updated.
- Keep Python 3.14 optional or manually triggered while service conda manifests
  are missing.
- Each deferred extra must either have a follow-up ticket or a Python-version
  marker that prevents unresolved installs on Python 3.14.
- The final CI matrix should include Python 3.14 for every workflow that covers
  an advertised support surface, and should explicitly exclude workflows tied to
  deferred extras.

## Core Install and Build Validation

Validation date: 2026-07-06. Local validation used CPython 3.14.5 from
`/opt/homebrew/bin/python3.14` and temporary virtual environments under `/tmp`.
The validation covered core ADS only, not optional extras or runtime tests.

### Results

- `uv pip install --python /tmp/ads-py314-core-venv/bin/python .` succeeded for
  core ADS from the local checkout.
- `python -m build` succeeded and produced:
  - `dist/oracle_ads-2.15.2.tar.gz`
  - `dist/oracle_ads-2.15.2-py3-none-any.whl`
- Installing the built wheel into a fresh Python 3.14 virtual environment
  succeeded.
- A minimal import smoke check succeeded:
  - Python: `3.14.5`
  - ADS: `2.15.2`

### Resolved Core Dependency Set

The successful core install selected these relevant package versions:

- `numpy==2.5.1`
- `pandas==2.3.3`
- `scikit-learn==1.9.0`
- `scipy==1.18.0`
- `matplotlib==3.11.0`
- `oci==2.181.0`
- `ocifs==1.3.4`
- `pydantic==2.13.4`
- `pydantic-core==2.46.4`

### Packaging Notes

- The local Homebrew CPython 3.14 `venv`/`ensurepip` bootstrap produced a broken
  `pip` entrypoint, so validation used `uv` to create and install into clean
  Python 3.14 virtual environments. This is an environment tooling issue, not
  an ADS packaging failure.
- The import smoke check emitted Python 3.14 `SyntaxWarning`s for invalid escape
  sequences in existing string literals:
  - `ads/telemetry/telemetry.py`
  - `ads/text_dataset/dataset.py`
- These warnings did not block install, build, wheel install, or import, but
  should be cleaned up before treating Python 3.14 runtime validation as fully
  complete.

### Scope Not Covered

- No optional extras were installed in this Bead.
- No unit tests were run in this Bead.
- No service conda environment was validated in this Bead.
- No model artifact/runtime workflows were validated in this Bead.

## Default Setup Unit Test Validation

Validation date: 2026-07-06. Local validation used the Python 3.14.5 core ADS
environment from `/tmp/ads-py314-core-venv`, with `test-requirements.txt`
installed into that environment. The suite was run with `NoDependency=True` and
a temporary CI-style dummy OCI config under `/tmp/ads-py314-home/.oci`.

### Setup Triage

- Running `tests/unitary/default_setup` without a dummy OCI config failed during
  collection with `oci.exceptions.InvalidConfig: {'user': 'missing'}` in model
  deployment and pipeline tests. This was an environment setup issue, not a
  Python 3.14 compatibility issue.
- Setting only `OCI_CONFIG_LOCATION` was not sufficient for this suite; many
  tests still read the default `~/.oci/config` location. Re-running with
  `HOME=/tmp/ads-py314-home` and a dummy `~/.oci/config` matched the existing
  CI setup pattern.
- A stale local `/tmp/model` fixture directory caused ONNX-related
  `FileExistsError` failures during one exploratory run. Removing that temp
  directory before the final run resolved the local fixture contamination.

### Python 3.14 Code Issue

- With the CI-style dummy OCI config in place, the first full suite run produced
  `17 failed, 1219 passed, 13 skipped, 2 xfailed`. All failures were isolated to
  `tests/unitary/default_setup/common/test_common_base_properties.py`.
- The failing code used `self.__annotations__` inside
  `ads/model/base_properties.py`. On Python 3.14, class annotations are not
  available through the instance for this case, producing:
  `AttributeError: 'MockTestProperties' object has no attribute '__annotations__'`.
- `BaseProperties` now reads annotations from `type(self).__annotations__`
  through a small helper property, preserving the existing type-validation
  behavior while avoiding the Python 3.14 instance lookup change.

### Results

- Targeted regression:
  `/usr/bin/env HOME=/tmp/ads-py314-home NoDependency=True /tmp/ads-py314-core-venv/bin/python -m pytest -v -p no:warnings --durations=10 tests/unitary/default_setup/common/test_common_base_properties.py`
  passed with `27 passed in 2.41s`.
- Full default setup suite:
  `/usr/bin/env HOME=/tmp/ads-py314-home NoDependency=True /tmp/ads-py314-core-venv/bin/python -m pytest -v -p no:warnings --durations=10 tests/unitary/default_setup`
  passed with `1236 passed, 13 skipped, 2 xfailed in 405.41s`.

### Follow-up Notes

- The default setup suite is now passing on Python 3.14 for the core ADS
  environment after the `BaseProperties` compatibility fix.
- Test setup for future Python 3.14 CI should create a dummy `~/.oci/config`,
  not only set `OCI_CONFIG_LOCATION`, to match this suite's current assumptions.

## Full and Targeted Unit Test Validation

Validation date: 2026-07-06. Local validation reused the Python 3.14.5 core ADS
environment from `/tmp/ads-py314-core-venv` and the temporary CI-style dummy OCI
home at `/tmp/ads-py314-home`.

### Full Suite Dependency Resolution

- `uv pip install --python /tmp/ads-py314-core-venv/bin/python -r dev-requirements.txt`
  did not parse the quoted editable extras line in `dev-requirements.txt`.
- `/tmp/ads-py314-core-venv/bin/python -m pip install -r dev-requirements.txt`
  failed while resolving the all-extras development environment. The first hard
  blocker was `fiona<=1.9.6` from the `geo` extra, which attempted a source
  build and failed because no Python 3.14 wheel was available in the resolver
  result and no local `gdal-config`/`GDAL_VERSION` was configured.
- Because `dev-requirements.txt` is not currently installable on Python 3.14,
  the full with-extras unit suite cannot be considered a valid Python 3.14
  acceptance gate yet.

### Full `tests/unitary` Probe

The full unitary tree was still run in the available core/test environment with
collection errors enabled:

`/usr/bin/env HOME=/tmp/ads-py314-home NoDependency=True /tmp/ads-py314-core-venv/bin/python -m pytest -q -p no:warnings --continue-on-collection-errors --durations=10 tests/unitary`

Result: `62 failed, 1763 passed, 19 skipped, 2 xfailed, 54 errors in 544.56s`.

The passing count includes the full default setup suite plus runnable portions
of with-extras coverage, including ADS string regex, AQUA client/entity/config
tests that did not require missing notebook/cache dependencies, BDS auth,
database connection tests that did not require missing drivers, generic model
tests, model environment/runtime metadata, metadata provenance, opctl command
and backend tests, operator loader/config/runtime tests, and vault tests.

### Failure Triage

- Collection errors were primarily missing optional dependencies rather than
  Python 3.14 syntax or runtime failures. Representative missing modules
  included `nltk`, `tornado`, `cachetools`, `notebook`, `huggingface_hub`,
  `sqlalchemy`, `datefinder`, `optuna`, `onnxruntime`, `nbformat`, framework
  packages, and operator-specific dependencies.
- BDS Hive failures sampled from
  `tests/unitary/with_extras/bds/test_bds_hive_connection.py` were caused by
  missing `impala`, so they should be treated as `bds` dependency setup
  blockers until that extra is validated on Python 3.14.
- Dataset failures sampled from
  `tests/unitary/with_extras/dataset/test_dataset_dataset.py` were caused by
  missing `datefinder`, so they should be treated as `data` extra setup
  blockers.
- HPO failures sampled from
  `tests/unitary/with_extras/hpo/test_hpo_distributions.py` were caused by
  missing `optuna`, so they should be treated as `optuna` extra setup blockers.
- Embedding ONNX model failures sampled from
  `tests/unitary/with_extras/model/test_model_framework_embedding_onnx_model.py`
  were caused by missing `onnxruntime`, so they should be treated as `onnx`
  extra setup blockers.
- The operator build-image failure in
  `tests/unitary/with_extras/operator/test_common_utils.py` was environment
  sensitive: local proxy environment variables were injected as Docker build
  arguments, while the test expected only `RND=1`. This is not a Python 3.14
  dependency issue and should be isolated in future CI by clearing proxy
  variables or relaxing the assertion to account for configured proxy build
  args.

### Targeted Passing Coverage

The following targeted dependency-sensitive checks passed in the Python 3.14
environment:

`/usr/bin/env HOME=/tmp/ads-py314-home NoDependency=True /tmp/ads-py314-core-venv/bin/python -m pytest -q -p no:warnings tests/unitary/default_setup/feature_types/test_statistical_metrics.py tests/unitary/default_setup/model/test_onnx_transformer.py tests/unitary/with_extras/model/test_env_info.py tests/unitary/with_extras/model/test_runtime_info.py tests/unitary/with_extras/model/test_metadata_provenance.py`

Result: `45 passed, 1 skipped, 1 xfailed in 7.64s`.

This gives a positive signal for the currently installed pandas, NumPy,
scikit-learn, SciPy, model runtime metadata, ONNX transformer helper, and model
metadata/provenance paths. It does not validate framework extras whose packages
are not installed on Python 3.14.

### Installed Optional Dependency Snapshot

In the Python 3.14 validation environment, core scientific packages were
available (`numpy`, `pandas`, `sklearn`, `scipy`, `matplotlib`), but these
dependency-sensitive optional modules were not installed: `sqlalchemy`, `fiona`,
`notebook`, `cachetools`, `tornado`, `nltk`, `optuna`, `onnx`, `onnxruntime`,
`tensorflow`, `torch`, `spacy`, and `nbformat`.

### Follow-up Notes

- Keep the Python 3.14 full with-extras unit suite blocked until
  `dev-requirements.txt` or a Python 3.14-specific test dependency set resolves
  cleanly.
- Resolve or marker-split `geo`/`fiona` before using the all-extras development
  requirements as a Python 3.14 CI gate.
- Validate with-extras suites by support bucket after their dependency groups
  are installable: `data`, `bds`, `notebook`/`aqua`, `onnx`, `optuna`, model
  framework extras, text/PII, operators, and opctl.

## Model Artifact and Runtime Version Validation

Validation date: 2026-07-06. Local validation reused the Python 3.14.5 core ADS
environment from `/tmp/ads-py314-core-venv`.

### Runtime Python Version Gate

- The model artifact introspection validator previously accepted runtime
  `INFERENCE_PYTHON_VERSION` values from Python 3.6 through Python 3.12 and
  rejected Python 3.14.
- The validator now accepts Python 3.14 runtime artifacts explicitly while still
  rejecting Python 3.13 and unvalidated future versions such as Python 3.15.
  This keeps Python 3.14 enablement separate from Python 3.13 and prevents an
  overly broad "3.6 or higher" acceptance rule.
- The introspection test description and boilerplate README were updated from
  "3.6 or higher" to "supported Python version" so the runtime validation
  boundary matches the implementation.

### Results

- Focused runtime version gate:
  `/usr/bin/env HOME=/tmp/ads-py314-home NoDependency=True /tmp/ads-py314-core-venv/bin/python -m pytest -q -p no:warnings tests/unitary/default_setup/model/test_model_introspect.py::TestPythonVersionCheck::test_python_version_check`
  passed with `1 passed in 2.29s`.
- Targeted model artifact/runtime suite:
  `/usr/bin/env HOME=/tmp/ads-py314-home NoDependency=True /tmp/ads-py314-core-venv/bin/python -m pytest -q -p no:warnings tests/unitary/default_setup/model/test_model_introspect.py tests/unitary/default_setup/model/test_model_artifact.py tests/unitary/with_extras/model/test_runtime_info.py tests/unitary/with_extras/model/test_env_info.py tests/unitary/with_extras/model/test_metadata_provenance.py`
  passed with `54 passed, 2 skipped in 6.94s`.

### Scope Boundary

- This Bead validates model artifact introspection, generated `runtime.yaml`
  handling, runtime metadata parsing, environment info, and model metadata
  provenance paths that are runnable in the core Python 3.14 environment.
- Framework-specific model artifact validation remains gated by optional
  dependency readiness. ONNX, TensorFlow, PyTorch, Hugging Face, LightGBM,
  XGBoost, Spark, and embedding model suites still need their Python 3.14
  dependency groups installed and validated before those surfaces are considered
  supported.

## Service-Required Extras Validation

Validation date: 2026-07-06. Local validation used resolver dry-runs from the
Python 3.14.5 core ADS environment:
`/tmp/ads-py314-core-venv/bin/python -m pip install --dry-run '.[<extra>]'`.
These checks validate Python 3.14 package availability and resolver behavior
without mutating the shared validation environment. They do not replace real
service conda installation or behavioral test runs.

### Resolver Results

| Extra | Dry-run result | Python 3.14 readiness note |
| --- | --- | --- |
| `data` | Pass | Resolves, but `SQLAlchemy==1.4.46` is selected from source because the `data` extra caps SQLAlchemy at `<=1.4.46`. Needs install/build and data-access behavioral validation in service conda. |
| `bds` | Pass | Resolves with current `ibis-framework[impala]`, SQLAlchemy 2.x, PyArrow, Kerberos/GSSAPI packages, and HDFS dependencies. Needs service conda runtime validation for native and Kerberos pieces. |
| `spark` | Pass | Resolves with `pyspark-4.1.2` from source and `py4j`. Needs Java/Spark runtime validation in the service image. |
| `notebook` | Pass with caveat | Resolves by selecting `scikit-learn-1.5.2` from source under the `<1.6.0` cap. Notebook/runtime support should not be considered complete until the cap is updated or a service-provided build is validated. |
| `opctl` | Pass with caveat | Resolves after heavy backtracking through `oci-cli`, `huggingface_hub`, `PyYAML`, `click`, and related runtime tooling. Needs pinned service conda constraints and operator behavioral tests to avoid resolver drift. |
| `aqua` | Pass with caveat | Resolves with Notebook 6.x-era constraints and `openai==1.109.1`. Needs Notebook/Jupyter runtime validation on Python 3.14 before support is advertised. |
| `torch` | Pass | Resolves with Python 3.14 wheels for `torch` and `torchvision` on the validation platform. Needs model artifact and framework integration tests in service conda. |
| `llm` | Pass | Resolves with current LangChain/OpenAI/evaluate stack and Python 3.14 wheels for key dependencies. Needs behavioral validation for service runtime and provider integrations. |
| `anomaly` | Pass with caveat | Resolves, but downgrades to `numpy-1.26.4`, selects `scikit-learn-1.5.2` from source, and pulls a large Merlion `[all]` stack including Prophet, PySpark, Torch, Dash, LightGBM, and opctl dependencies. Treat as deferred until a pinned service conda environment installs and operator tests pass. |
| `recommender` | Pass with caveat | Resolves with a Python 3.14 `scikit-surprise-1.1.5` wheel plus report/opctl dependencies. Needs recommender operator tests before support is advertised. |
| `forecast` | Fail | Fails while resolving `mlforecast==1.0.2` dependencies after falling back to a `coreforecast` source candidate: `ERROR: Use build.verbose instead of cmake.verbose for scikit-build-core >= 0.10`. The stack also forces `numpy<2.0.0`. |
| `regression` | Fail | Fails on the same `coreforecast` source build path because `regression` includes `forecast`. |
| `onnx` | Fail | Fails because `onnxruntime~=1.22.0` has no matching Python 3.14 distribution; available Python 3.14 candidates start at newer releases. The extra also retains old `onnx`, `xgboost`, and scikit-learn caps. |
| `tensorflow` | Fail | Fails because no matching TensorFlow distribution is available for the Python 3.14 validation environment. |
| `text` | Fail | Fails building the capped spaCy stack (`spacy<3.8` through `thinc`/`blis`) with Cython/NumPy build errors under Python 3.14. |
| `pii` | Fail | Fails on pinned `spacy==3.6.1` / `spacy-transformers==1.2.5`; the dependency chain reaches old `thinc`/`blis` source builds that are not Python 3.14-ready. |
| `geo` | Fail | Fails building `fiona<=1.9.6` without a usable Python 3.14 wheel or local GDAL build configuration. |

### Required Follow-ups

- Forecast and regression: update or marker-split the forecast stack for Python
  3.14, including `numpy<2.0.0`, `mlforecast==1.0.2`, `coreforecast`,
  `neuralprophet`, `prophet`, `pmdarima`, `xgboost`, and related operator
  tests.
- Anomaly: create a pinned Python 3.14 service conda validation item for the
  Merlion `[all]` stack. The dry-run resolves, but it depends on source-built
  `scikit-learn-1.5.2`, `numpy-1.26.4`, PySpark, Torch, Prophet, Dash, and
  opctl dependencies.
- ONNX: move Python 3.14 markers to compatible `onnxruntime` and `onnx`
  versions, then rerun ONNX model artifact and transformer tests.
- TensorFlow: defer TensorFlow support until Python 3.14 distributions are
  available or service conda provides a validated build.
- Text and PII: upgrade spaCy, thinc/blis, spaCy Transformers, and scrubadub
  integration constraints to Python 3.14-compatible releases before enabling
  these extras.
- Geo: update Fiona/GeoPandas/GDAL constraints or validate service-provided
  native builds before including `geo` in Python 3.14 all-extras gates.
- Notebook, AQUA, data, opctl, BDS, Spark, Torch, LLM, and recommender: dry-run
  resolution is a positive signal only. Each still needs a real Python 3.14
  service conda install and targeted behavioral tests before ADS advertises
  support for that surface.

## Python 3.14 Follow-up Backlog

Backlog date: 2026-07-06. These items consolidate the known Python 3.14
limitations found during the readiness pass. They are written so each item can
be copied into a follow-up ticket without relying on this whole document for
context.

### Package Metadata and Release Signaling

Problem:
ADS core install/build and default unit tests pass on Python 3.14, but optional
extras, service conda inputs, and CI gates are not complete enough to advertise
full Python 3.14 support.

Evidence:
- `pyproject.toml` classifiers and dependency markers were intentionally left
  unchanged.
- Planned OCI Data Science Python 3.14 service conda manifests were not present
  in this workspace.
- Several service-facing extras fail or resolve only with source-build caveats.

Follow-up action:
- Add Python 3.14 metadata only after the agreed support bucket is explicit:
  either core/default-only support, or broader service conda support after the
  dependency groups below are validated.
- If dependency pins change, update license metadata in the same change set.

Exit criteria:
- Python 3.14 classifier and dependency markers match the validated support
  surface.
- Package metadata, dependency pins, and license metadata are reviewed together.

### CI Enablement

Problem:
Current workflow matrices should not include Python 3.14 yet because all-extras
and service-facing dependency groups still produce expected failures.

Evidence:
- Core build/install passed in a local Python 3.14.5 environment.
- `tests/unitary/default_setup` passed with `1236 passed, 13 skipped, 2 xfailed`.
- Full `tests/unitary` probing still had `62 failed` and `54 errors`, mostly
  tied to missing optional extras or unresolved dependency groups.
- `dev-requirements.txt` cannot currently install on Python 3.14 because the
  all-extras path reaches `geo` / `fiona<=1.9.6`.

Follow-up action:
- Add a staged Python 3.14 workflow gate for core install, wheel install, and
  default setup tests first.
- Add optional-extra and operator workflow coverage by support bucket after each
  dependency group is installable.
- Ensure CI creates a dummy `~/.oci/config`; setting only `OCI_CONFIG_LOCATION`
  was not enough for the default setup suite.
- Clear or account for proxy build arguments in operator Docker build tests.

Exit criteria:
- Core/default Python 3.14 CI passes reliably.
- Optional-extra CI gates are added only for dependency groups that resolve and
  pass targeted behavioral tests.

### Service Conda Manifests

Problem:
ADS Python 3.14 support must match OCI Data Science service conda environments,
but the planned Python 3.14 service conda manifests were not available in this
workspace.

Evidence:
- Repo-visible service conda evidence is historical or template-based.
- PyPI dry-runs are useful readiness signals but do not prove service conda
  compatibility, especially for native packages and ML frameworks.

Follow-up action:
- Obtain planned Python 3.14 service conda manifests for General ML, ONNX,
  TensorFlow, PyTorch, Spark/Data Flow/BDS, forecast, anomaly, PII, recommender,
  regression, AQUA, notebook, and operator environments.
- Compare service pins against `pyproject.toml` extras and test requirements.
- Decide per mismatch whether ADS pins should move, Python 3.14 markers should
  be split, or the service environment should provide a pinned build.

Exit criteria:
- Every service-required ADS extra either installs in the Python 3.14 service
  conda image or has an explicit deferred-support decision.

### Forecast and Regression Operators

Problem:
`forecast` and `regression` are not Python 3.14-ready under current pins.
`regression` inherits the `forecast` blocker.

Evidence:
- `pip install --dry-run '.[forecast]'` failed while resolving
  `mlforecast==1.0.2` dependencies after falling back to a `coreforecast`
  source candidate.
- The concrete failure was
  `ERROR: Use build.verbose instead of cmake.verbose for scikit-build-core >= 0.10`.
- The stack also forces `numpy<2.0.0`, which pulls older source-build paths on
  Python 3.14.

Follow-up action:
- Revisit `numpy<2.0.0`, `mlforecast==1.0.2`, `coreforecast`,
  `neuralprophet`, `prophet`, `pmdarima`, `xgboost`, `sktime`, and related
  operator test constraints for Python 3.14.
- Decide whether to upgrade pins, marker-split Python 3.14, or defer these
  operators from the first Python 3.14 service conda release.

Exit criteria:
- Forecast and regression extras install in the Python 3.14 validation
  environment.
- Forecast/regression operator test suites pass under the service conda image.

### Anomaly Operators

Problem:
`anomaly` resolves in dry-run but is not ready to advertise because the resolved
stack is large, downgrade-heavy, and behaviorally untested on Python 3.14.

Evidence:
- Dry-run resolution selected `numpy-1.26.4` and source-built
  `scikit-learn-1.5.2`.
- `salesforce-merlion[all]==2.0.4` pulled Prophet, PySpark, Torch, Dash,
  LightGBM, and opctl dependencies.

Follow-up action:
- Build a pinned Python 3.14 service conda environment for anomaly and run the
  anomaly operator suite.
- Decide whether Merlion and scikit-learn constraints need upgrades or Python
  3.14-specific markers.

Exit criteria:
- Anomaly installs without unexpected source-build drift in service conda.
- Anomaly operator tests pass or the extra is explicitly deferred.

### ONNX

Problem:
The ONNX extra is blocked by Python 3.14 package availability under current
pins.

Evidence:
- `pip install --dry-run '.[onnx]'` failed because `onnxruntime~=1.22.0` had no
  matching Python 3.14 distribution in the validation environment.
- The extra also keeps older ONNX, XGBoost, and scikit-learn caps that need
  review for Python 3.14.
- Full-suite probing showed ONNX embedding model tests blocked by missing
  `onnxruntime`.

Follow-up action:
- Move Python 3.14 markers to compatible ONNX Runtime and ONNX releases.
- Revalidate XGBoost and scikit-learn caps for the ONNX support bucket.
- Run ONNX model artifact and transformer tests after installation succeeds.

Exit criteria:
- ONNX extra installs on Python 3.14.
- ONNX model artifact and transformer tests pass.

### TensorFlow

Problem:
TensorFlow is unavailable for the Python 3.14 validation environment.

Evidence:
- `pip install --dry-run '.[tensorflow]'` failed because no matching TensorFlow
  distribution was available.

Follow-up action:
- Track TensorFlow Python 3.14 wheel availability or validate a service-provided
  TensorFlow build.
- Keep TensorFlow outside the advertised Python 3.14 ADS support surface until
  an installable build and behavioral tests exist.

Exit criteria:
- TensorFlow extra installs on Python 3.14.
- TensorFlow model integration tests pass in the service environment.

### Text and PII

Problem:
Text and PII extras are blocked by old spaCy-family constraints.

Evidence:
- `text` failed while building `spacy<3.8` through `thinc` / `blis`.
- `pii` failed on pinned `spacy==3.6.1` and
  `spacy-transformers==1.2.5`, reaching old `thinc` / `blis` source builds.

Follow-up action:
- Upgrade or marker-split spaCy, thinc/blis, spaCy Transformers, scrubadub, and
  scrubadub spaCy integration constraints for Python 3.14.
- Re-run text and PII dependency installation plus relevant NLP/PII tests.

Exit criteria:
- Text and PII extras install on Python 3.14 without unsupported source builds.
- Text and PII tests pass or these extras are explicitly deferred.

### Geo

Problem:
The geo extra blocks all-extras test dependency installation on Python 3.14.

Evidence:
- `dev-requirements.txt` failed because all-extras reaches `geo` /
  `fiona<=1.9.6`.
- `pip install --dry-run '.[geo]'` failed building Fiona without a usable Python
  3.14 wheel or local GDAL build configuration.

Follow-up action:
- Update Fiona/GeoPandas/GDAL constraints for Python 3.14 or validate
  service-provided native builds.
- Split `geo` out of Python 3.14 all-extras CI until it is installable.

Exit criteria:
- `geo` installs in the Python 3.14 service/test environment.
- All-extras development requirements no longer fail on Fiona/GDAL.

### Notebook, AQUA, Data Access, BDS, Spark, Opctl, Torch, LLM, Recommender

Problem:
These extras produced positive resolver signals but still need real service
conda installation and targeted behavioral tests.

Evidence:
- `notebook` resolves by selecting `scikit-learn-1.5.2` from source under the
  `<1.6.0` cap.
- `aqua` resolves with Notebook 6.x-era constraints.
- `data` resolves but selects `SQLAlchemy==1.4.46` from source due the
  `<=1.4.46` cap.
- `bds`, `spark`, `opctl`, `torch`, `llm`, and `recommender` resolve, with
  native/runtime dependencies that PyPI dry-runs do not fully validate.

Follow-up action:
- Validate each extra in its planned service conda image.
- Run targeted data access, notebook/runtime, Spark/Data Flow/BDS, operator,
  model framework, LLM, and recommender tests.
- Review whether scikit-learn, SQLAlchemy, Notebook, and opctl-related pins
  should be updated or marker-split for Python 3.14.

Exit criteria:
- Each extra has a passing install and targeted test result, or a documented
  deferred-support decision.

### Python 3.14 Code and Test Hygiene

Problem:
The default test suite is passing after the `BaseProperties` fix, but the
readiness run exposed hygiene items that should be cleaned before long-term
Python 3.14 CI enforcement.

Evidence:
- `BaseProperties` needed class-level annotation lookup on Python 3.14.
- Core import/build surfaced existing `SyntaxWarning` messages for invalid
  escape sequences in `ads/telemetry/telemetry.py` and
  `ads/text_dataset/dataset.py`.
- Test setup depends on a dummy home-level OCI config.

Follow-up action:
- Keep the `BaseProperties` regression covered in default setup tests.
- Fix invalid escape sequence warnings.
- Document or automate Python 3.14 test bootstrap requirements for OCI config.

Exit criteria:
- Python 3.14 default CI runs without setup-only failures or avoidable warning
  noise.

## Final Acceptance Criteria Review

Review date: 2026-07-06. This section maps the completed readiness work back to
the parent acceptance criteria for ODSC-88493.

| Acceptance criterion | Status | Review result |
| --- | --- | --- |
| ADS package metadata advertises Python 3.14 support only after validation is complete. | Met | `pyproject.toml` was intentionally left unchanged. Python 3.14 classifiers and broad dependency markers are still gated on service conda and optional-extra validation. |
| Core ADS installation succeeds on Python 3.14 with resolved dependencies. | Met | Core ADS installed on CPython 3.14.5, source and wheel distributions built, the built wheel installed into a fresh Python 3.14 venv, and an import/version smoke check passed. |
| Required service-conda dependency groups are installable on Python 3.14 or have explicit follow-up tickets for deferred support. | Partially met with documented follow-ups | PyPI dry-runs identified installable groups, caveated groups, and blocked groups. The follow-up backlog captures service conda manifest needs and ticket-ready deferred items for blocked or unvalidated groups. |
| Relevant GitHub Actions workflows include Python 3.14 coverage or a documented staged validation plan if CI images need separate enablement. | Met as staged plan | Workflow matrices were left unchanged because current dependency groups would create expected failures. A staged CI plan documents core/default gates first, then optional-extra/operator gates by support bucket. |
| Unit tests required for the agreed support scope pass on Python 3.14. | Met for core/default scope | `tests/unitary/default_setup` passed with `1236 passed, 13 skipped, 2 xfailed`. Targeted dependency-sensitive tests passed where dependencies were available. Full with-extras remains blocked by dependency readiness and is documented as out of the current validated scope. |
| Model artifact/runtime validation accepts Python 3.14 where ADS support is intended. | Met | Runtime artifact introspection now accepts Python 3.14 explicitly while rejecting Python 3.13 and future unvalidated Python 3.15. Focused and targeted model artifact/runtime tests passed. |
| Dependency changes are reflected in `pyproject.toml` and license metadata is updated if required. | Met by no-op decision | No dependency pins were changed in this readiness pass, so no license metadata update was required. Future dependency changes are called out in the metadata checklist and backlog. |
| Known Python 3.14 incompatibilities are captured as follow-up work. | Met | The follow-up backlog covers package metadata, CI, service conda manifests, forecast/regression, anomaly, ONNX, TensorFlow, text/PII, geo, caveated resolver-pass extras, and Python 3.14 test hygiene. |

### Final Support Decision

ADS has a validated Python 3.14 core/default-test readiness path, plus a small
runtime validation code change for model artifacts. ADS should not yet advertise
broad Python 3.14 package support because service conda manifests are missing
and multiple service-facing optional extras are either blocked or only resolved
with caveats. The PR-ready state for this pass is therefore:

- Keep Python 3.14 package metadata and CI matrices gated.
- Merge the Python 3.14 code fixes and runtime validation updates.
- Use this document as the audit trail and follow-up backlog for completing
  broader Python 3.14 support.
