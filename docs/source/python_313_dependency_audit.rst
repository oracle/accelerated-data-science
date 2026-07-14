Python 3.13 Dependency Audit
============================

This audit records the Python 3.13 dependency surface reviewed for
ODSC-88492. It documents the initial supported surface, the deferred optional
extras, the validation completed before advertising Python 3.13 support, and
the remaining follow-up work for older ML, NLP, and operator dependency stacks.

Environment Checked
-------------------

* Local interpreter: Python 3.13.13.
* Resolver checks used ``python -m pip install --dry-run --ignore-installed``.
* Editable metadata generation for the core package succeeded.

Core Dependencies
-----------------

The base package resolves on Python 3.13.13. Pip selected Python 3.13-compatible
wheels for the core native dependency path, including:

* ``numpy==2.5.1``
* ``pandas==2.3.3``
* ``scikit-learn==1.9.0``
* ``scipy==1.18.0``
* ``matplotlib==3.11.0``
* ``pydantic==2.13.4``
* ``oci==2.181.1``
* ``ocifs==1.3.4``

The current lower bounds are broad enough to let Python 3.13 resolve, but they
also permit materially newer NumPy and scikit-learn versions than older service
environments may have exercised. Runtime and unit validation should pay
particular attention to pandas, NumPy, scikit-learn, serialization, model
artifact generation, and schema validation behavior.

Optional Extras
---------------

``onnx``
  Deferred on Python 3.13. Resolver checks selected ``onnx==1.17.0``,
  ``onnxruntime==1.22.1``, ``skl2onnx==1.18.0``, ``tf2onnx==1.17.0``,
  ``xgboost==1.6.2``, and ``scikit-learn==1.5.2`` because the extra caps
  scikit-learn below 1.6, but Python 3.13 CI attempted to build ONNX from
  source and failed before tests started. This extra needs a Python 3.13 wheel
  path or an updated ONNX stack before support can be claimed.

``tensorflow``
  Resolves on Python 3.13.13. The unbounded ``python_version >= '3.12'`` branch
  selected ``tensorflow==2.21.0`` and ``keras==3.15.0``. This is a large version
  jump from the pre-3.12 cap of ``tensorflow<=2.15.1`` and needs TensorFlow
  model serialization and artifact verification before Python 3.13 is claimed.

``text``
  Does not resolve cleanly. The current ``spacy>=3.4.2,<3.8`` cap selected
  ``spacy==3.7.5`` and attempted to build ``thinc``/``blis`` from source.
  Build dependency preparation failed because the selected stack requires
  Cython and NumPy build behavior that is not compatible with Python 3.13 in
  this resolver path. This extra needs a spaCy/thinc/blis version update or a
  Python 3.13 exclusion/deferral.

``opctl``
  Resolves on Python 3.13.13, but pip performed extensive backtracking across
  ``oci-cli`` and ``oci`` versions. The selected dry-run set included
  ``oci-cli==3.89.1`` and ``oci==2.181.1``. Consider tightening the compatible
  ``oci-cli`` range to reduce CI and service-environment resolver time.

``forecast``
  Did not reach a clean fast resolver path. The extra forces ``numpy<2.0.0``,
  which selected ``numpy==1.26.4`` on Python 3.13 and entered source metadata
  preparation. This is a high-risk signal because Python 3.13 support generally
  depends on NumPy 2.x wheels for the ML stack. Packages needing targeted review
  include ``mlforecast==1.0.2``, ``neuralprophet>=0.7.0``,
  ``pytorch-lightning==2.5.5``, ``pmdarima``, ``prophet==1.1.7``,
  ``cmdstanpy==1.2.5``, ``xgboost<3.0.0``, ``sktime``, and ``statsmodels``.

``anomaly``
  Did not reach a clean fast resolver path. The extra caps scikit-learn below
  1.6 and depends on ``salesforce-merlion[all]==2.0.4``; the resolver selected
  ``scikit-learn==1.5.2`` and entered ``numpy==1.26.4`` source metadata
  preparation through the older anomaly stack. This extra should be deferred or
  updated for Python 3.13 after a focused Merlion, NumPy, and scikit-learn
  compatibility review.

``pii`` and ``recommender``
  Not fully dry-run checked in this pass. They should be treated as risk areas:
  ``pii`` pins ``spacy==3.6.1``, ``spacy-transformers==1.2.5``, and
  ``scrubadub==2.0.1``; ``recommender`` depends on ``scikit-surprise``. Both
  include native or older ML/NLP dependencies likely to need Python 3.13-specific
  validation or deferral.

``geo``
  Deferred on Python 3.13. The current stack pins ``fiona<=1.9.6`` through
  ``geopandas<1.0.0``. In CI, Python 3.13 dependency setup attempted to install
  Fiona without a compatible wheel and failed because ``gdal-config`` was not
  available. This extra needs either a GeoPandas/Fiona/GDAL stack update or a CI
  image with GDAL headers and configuration before Python 3.13 support can be
  claimed.

Concrete Dependency Actions
---------------------------

Use this list as the dependency-change input for the Python 3.13 constraint
update work:

* Core package: no immediate dependency exclusion is required for Python 3.13.
  Keep the base resolver path open for NumPy 2.x, pandas 2.3.x, scikit-learn
  1.9.x, scipy 1.18.x, matplotlib 3.11.x, pydantic 2.x, OCI SDK, and OCIFS,
  then validate behavior in the targeted runtime and unit-test beads.
* ``opctl``: retain the ``oci-cli>=3.89.1`` floor. If CI still spends too much
  time backtracking across OCI CLI and OCI SDK versions, add a narrower upper or
  compatible-release bound after checking the service conda target version.
* ``tensorflow``: keep Python 3.13 installability in scope, but review whether
  the ``python_version >= '3.12'`` branch should be bounded to the validated
  TensorFlow/Keras range. Do not claim TensorFlow model parity until serializer
  and artifact tests run against the Python 3.13-selected Keras 3 stack.
* ``onnx``: keep Python 3.13 excluded until a wheel-backed stack is selected.
  The required future change is a coordinated update for ``onnx``,
  ``onnxruntime``, ``onnxmltools``, ``skl2onnx``, ``tf2onnx``, ``xgboost``,
  and the scikit-learn cap.
* ``text`` and ``pii``: keep Python 3.13 excluded until the spaCy/thinc/blis
  stack is raised. ``pii`` also requires review of ``spacy-transformers``,
  ``scrubadub``, and ``scrubadub_spacy`` pins.
* ``forecast``: keep Python 3.13 excluded until the ``numpy<2.0.0`` constraint
  is removed or justified by a compatible wheel path. Review ``mlforecast``,
  ``neuralprophet``, ``pmdarima``, ``prophet``, ``cmdstanpy``, ``sktime``,
  ``statsmodels``, ``xgboost``, and ``pytorch-lightning`` together because the
  stack resolves as a coupled time-series environment.
* ``anomaly``: keep Python 3.13 excluded until ``salesforce-merlion[all]``,
  ``rrcf``, and the ``scikit-learn<1.6.0`` cap are validated or updated.
* ``regression``: keep Python 3.13 excluded because it composes the deferred
  forecast extra.
* ``recommender``: keep Python 3.13 excluded until ``scikit-surprise`` native
  wheel/build behavior is validated.
* ``geo``: keep Python 3.13 excluded until the GeoPandas/Fiona/GDAL stack is
  upgraded or CI reliably provides the GDAL headers and ``gdal-config`` needed
  for source builds.
* ``boosted`` and ``notebook``: keep their ``scikit-learn<1.6.0`` compatibility
  caps for Python versions below 3.13 only. On Python 3.13, let these extras use
  the base package's newer scikit-learn resolver path instead of forcing the
  older cap.
* ``viz``, ``data``, ``bds``, ``spark``, ``llm``, ``aqua``, ``torch``,
  ``huggingface``, and ``optuna``: no Python 3.13-specific exclusion is
  identified in this audit, but each still needs an explicit resolver check
  before being listed as supported for service conda environments.

Test and Development Dependencies
---------------------------------

``test-requirements.txt`` installs the base package plus pytest, coverage, ruff,
setuptools, and related test harness packages. It is the dependency input for
the Python 3.13 default-setup workflow and has no Python 3.13-specific package
exclusion from this audit.

``dev-requirements.txt`` installs
``.[aqua,bds,data,geo,huggingface,llm,notebook,onnx,opctl,optuna,spark,tensorflow,text,torch,viz]``
and ``.[testsuite]``. On Python 3.13, the marker-gated ``geo``, ``onnx``, and
``text`` dependencies are intentionally skipped, so a successful install of
``dev-requirements.txt`` does not prove those deferred extras are supported.

``test-requirements-operators.txt`` composes the deferred ``forecast`` and
``anomaly`` extras and pins ``protobuf==5.29.6``. It should remain outside
Python 3.13 initial support until those operator stacks are upgraded and
resolver checked.

The ``testsuite`` extra still needs a Python 3.13 resolver pass before broad
test/development support is claimed. Pay attention to native or behavior-heavy
packages including ``faiss-cpu``, ``fastparquet==2024.2.0``,
``imbalanced-learn``, ``notebook==6.4.12``, ``pyarrow>=15.0.0``,
``tables>3.9.0``, ``sktime``, and ``report-creator==1.0.37``.

Existing Python-Version Gates
-----------------------------

Current Python-version-specific dependency markers include the ONNX/TensorFlow
support paths and explicit Python 3.13 exclusions for deferred extras:

* ``onnx>=1.12.0,<=1.15.0; python_version < '3.12'``
* ``onnx~=1.17.0; python_version >= '3.12' and python_version < '3.13'``
* ``onnxmltools~=1.13.0; python_version < '3.13'``
* ``onnxruntime~=1.17.0,!=1.16.0; python_version < '3.12'``
* ``onnxruntime~=1.22.0; python_version >= '3.12' and python_version < '3.13'``
* ``skl2onnx>=1.10.4; python_version < '3.12'``
* ``skl2onnx~=1.18.0; python_version >= '3.12' and python_version < '3.13'``
* ``tensorflow<=2.15.1; python_version < '3.12'``
* ``tensorflow; python_version >= '3.12'``
* ``boosted`` and ``notebook`` retain their ``scikit-learn<1.6.0`` caps only
  with ``python_version < '3.13'``.
* ``text`` dependencies are excluded with ``python_version < '3.13'``.
* ``forecast``, ``anomaly``, ``regression``, ``recommender``, ``pii``, ``geo``,
  and ``onnx`` dependencies are excluded with ``python_version < '3.13'``.

The ``opctl`` extra now requires ``oci-cli>=3.89.1`` to keep Python 3.13
resolution on the current OCI CLI line.

Package metadata now includes the ``Programming Language :: Python :: 3.13``
classifier after the initial support scope completed install, build, runtime,
CI, and scoped test validation.

CI and Runtime Support Gates
----------------------------

The repository has these Python-version gates that define where Python 3.13 is
validated and where it remains intentionally deferred:

* ``.github/workflows/run-unittests-default_setup.yml`` runs
  ``tests/unitary/default_setup`` on Python 3.9, 3.10, 3.11, 3.12, and 3.13.
  This is the core minimum-dependency validation gate for initial support.
* ``.github/workflows/run-unittests-py310-py311.yml`` now covers Python 3.10,
  3.11, 3.12, and 3.13 for the broader unitary and model test suites. The
  Python 3.13 leg excludes deferred extra-backed paths that are marker-gated in
  ``pyproject.toml``.
* ``.github/workflows/run-operators-unit-tests.yml``,
  ``.github/workflows/run-forecast-unit-tests.yml``, and
  ``.github/workflows/run-forecast-explainer-tests.yml`` remain on Python 3.10
  and 3.11. They should stay outside the initial Python 3.13 claim until the
  operator, forecast, and forecast explainer stacks are upgraded and resolver
  checked.
* Model artifact introspection accepts inference Python versions from 3.6
  through 3.13 via ``PYTHON_VER_PATTERN`` in
  ``ads/model/model_artifact_boilerplate/artifact_introspection_test/model_artifact_validate.py``.
  ``tests/unitary/default_setup/model/test_model_introspect.py`` verifies that
  3.13 is accepted and 3.14 is rejected.
* Operator metadata defaults in ``ads/opctl/operator/*/MLoperator`` and
  ``ads/opctl/operator/common/operator_loader.py`` remain pinned to Python 3.11
  examples/defaults. These are not blockers for core package support, but they
  are explicit follow-up points before operator Python 3.13 support is claimed.

Initial Support Scope
---------------------

The recommended initial Python 3.13 support scope is the ADS core package plus
the optional extras that already resolve or have a bounded validation path. This
scope keeps service conda readiness moving while separating older ML, NLP, and
operator dependency stacks that need their own upgrade work.

In scope for initial support:

* Core ADS install and import behavior.
* Default setup unit tests under ``tests/unitary/default_setup``.
* Core data science dependencies resolved by the base package: NumPy, pandas,
  scikit-learn, matplotlib, pydantic, OCI SDK, and OCIFS.
* Model artifact and runtime metadata paths that do not require deferred extras,
  including generic model artifact generation, runtime YAML handling, model
  metadata, provenance, and serializer utilities.
* ``opctl`` installability, with resolver-time follow-up for ``oci-cli`` if CI
  backtracking remains high.
* ``tensorflow`` installability and TensorFlow model artifact validation after
  the dependency constraints are reviewed in the next step.
* ``viz``, ``data``, ``notebook``, ``llm``, ``aqua``, ``torch``,
  ``huggingface``, ``spark``, ``optuna``, ``boosted``, and ``bds`` only after
  each extra has a Python 3.13 resolver check. They were not proven by this
  audit and should not be assumed supported from the base resolver result.

Out of scope for initial support unless separately upgraded and validated:

* ``text`` because the current spaCy cap fails the Python 3.13 resolver path.
* ``pii`` because it pins older spaCy/spaCy-transformers and scrubadub packages.
* ``forecast`` because it currently forces NumPy 1.x and includes multiple
  older time-series dependencies.
* ``anomaly`` because it depends on ``salesforce-merlion[all]==2.0.4`` and caps
  scikit-learn below 1.6.
* ``regression`` because it composes ``forecast``.
* ``recommender`` because ``scikit-surprise`` needs native dependency validation
  on Python 3.13.
* ``geo`` because ``fiona<=1.9.6`` failed Python 3.13 CI dependency setup when
  GDAL config was not available.
* ``onnx`` because Python 3.13 CI attempted to build ONNX from source and failed
  before tests started.
* Forecast explainer tests because they depend on the forecast stack and should
  follow forecast support rather than block core support.
* Operator workflows backed by deferred extras. Operator framework utilities and
  YAML generation can stay in scope if they do not import or install deferred
  runtime stacks.

Validation Completed Before Advertising Support
-----------------------------------------------

The Python 3.13 package classifier was added only after the following
initial-scope validation completed:

* Built editable metadata, source distribution, and wheel artifacts on
  Python 3.13.13.
* Installed the base package from both source distribution and wheel artifacts
  in fresh Python 3.13 virtual environments.
* Verified artifact metadata contains the Python 3.13 deferred-extra markers
  and the ``oci-cli>=3.89.1`` ``opctl`` floor.
* Ran the default setup unit suite locally with the same dummy OCI config shape
  used by CI. The Python 3.13 run passed with ``1237 passed, 13 skipped,
  2 xfailed``.
* Ran targeted model artifact, runtime validation, artifact upload/download,
  model metadata, serializer, ONNX transformer helper, job runtime, service
  conda translation, and runtime dependency tests on Python 3.13. These targeted
  slices passed with ``47 passed, 2 skipped``, ``141 passed``, and
  ``55 passed`` across the grouped commands.
* Ran selected installed optional-extra tests that are relevant to service
  conda environments and do not require deferred dependency stacks:
  Aqua client/global/recommendation tests passed with ``85 passed``,
  opctl/operator utility tests passed with ``7 passed``, and vault helper tests
  passed with ``3 passed``.
* Ran a broader local unitary command with the Python 3.13 deferred-extra ignore
  list used by CI. The run reached execution and passed most collected tests,
  but the local core/dev environment lacks maximum-extra packages such as
  ``notebook``, ``jupyter_server``, ``autogen``, ``mysql``, ``oracledb`` or
  ``cx_Oracle``, ``bokeh``, ``datefinder``, ``lightgbm``, ``torch``,
  ``langchain``, and ``langchain_core``. The resulting local limitation was
  ``1831 passed, 9 skipped, 2 xfailed, 29 failed, 24 errors`` and should not be
  interpreted as a resolved maximum-extra support claim.
* Updated GitHub Actions unit-test matrices to include Python 3.13.
* Scoped Python 3.13 CI exclusions to deferred optional surfaces:
  ``ads_string``/``text``, geo-backed feature engineering/type tests,
  ONNX-backed model tests, ``operator/pii``, ``operator/forecast``, and
  ``operator/regression``.

Service Conda Support Matrix
----------------------------

Use this matrix when deciding which ADS surfaces can be included in Python 3.13
service conda environments and which require separate follow-up work:

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Surface
     - Python 3.13 status
     - Notes for service conda owners
   * - Core ADS
     - Supported
     - Base install, import, sdist, wheel, default setup tests, and targeted
       runtime paths passed on Python 3.13.13.
   * - Model artifact/runtime
     - Supported
     - Generic model artifacts, runtime validation, artifact upload/download,
       metadata population, model metadata, and serializer helper paths passed.
   * - Jobs and service conda helpers
     - Supported
     - Python/container job runtime tests, job serialization, runtime dependency
       handling, and service conda runtime translation passed.
   * - ``opctl``
     - Partially supported
     - Installability and utility/config tests passed. Operator workflows that
       depend on forecast, PII, or regression stacks remain deferred.
   * - ``aqua``
     - Partially supported
     - Client, global config, and recommendation tests passed. Notebook server
       extension and UI handler tests were not functionally validated in the
       local Python 3.13 environment because ``notebook`` and
       ``jupyter_server`` were not installed there.
   * - ``data``
     - Partially supported
     - The extra remains active on Python 3.13, but broad data/evaluator paths
       that require ``oracledb``, ``cx_Oracle``, ``datefinder``, ``bokeh``, or
       database client packages still need maximum-extra validation.
   * - ``notebook`` and ``boosted``
     - Partially supported
     - Their scikit-learn compatibility caps now apply only below Python 3.13.
       Python 3.13 uses the base newer scikit-learn resolver path, but notebook
       and boosted functionality still needs explicit max-extra validation.
   * - ``tensorflow``
     - Installable, not parity-validated
     - Python 3.13 resolves to a much newer TensorFlow/Keras stack. Keep model
       serializer and artifact parity as follow-up work before claiming full
       TensorFlow behavior.
   * - ``text``/``ads_string`` and ``pii``
     - Deferred
     - Current spaCy/thinc/blis and PII NLP stacks are marker-gated below
       Python 3.13.
   * - ``forecast``, forecast explainers, ``anomaly``, and ``regression``
     - Deferred
     - Time-series and anomaly stacks remain marker-gated below Python 3.13.
   * - ``geo``
     - Deferred
     - GeoPandas/Fiona/GDAL dependency stack remains marker-gated below
       Python 3.13.
   * - ``onnx``
     - Deferred
     - Current ONNX stack remains marker-gated below Python 3.13 until a
       reliable wheel-backed resolver and runtime test path is selected.
   * - ``recommender``
     - Deferred
     - ``scikit-surprise`` native wheel/build behavior still needs Python 3.13
       validation.

Known Incompatibilities and Deferred Extras
-------------------------------------------

The following extras are explicitly deferred from initial Python 3.13 support
with ``python_version < '3.13'`` dependency markers:

* ``text``/``ads_string`` because the current spaCy/thinc/blis stack does not
  resolve cleanly on Python 3.13.
* ``pii`` because it pins older spaCy, spaCy transformers, scrubadub, and
  scrubadub-spacy packages.
* ``forecast`` because it currently forces NumPy 1.x and includes multiple
  older time-series dependencies.
* ``anomaly`` because ``salesforce-merlion[all]==2.0.4``, ``rrcf==0.4.4``, and
  the scikit-learn cap require a focused Python 3.13 compatibility update.
* ``regression`` because it composes the deferred ``forecast`` extra.
* ``recommender`` because ``scikit-surprise`` native build/wheel behavior still
  needs Python 3.13 validation.
* ``geo`` because ``fiona<=1.9.6`` needs GDAL headers/config when a compatible
  Python 3.13 wheel is not available in CI.
* ``onnx`` because the current ONNX stack still tries to build ONNX from source
  on Python 3.13 in CI and fails before tests run.
* Forecast explainers because they depend on the deferred forecast stack.

``tensorflow`` remains installable in the initial support scope, but it needs
heavier Linux runtime coverage before its full serializer and model artifact
behavior should be considered equivalent to earlier Python versions. The Python
3.13 resolver selects a newer TensorFlow/Keras stack.

Follow-Up Ticket Candidates
---------------------------

Track these as separate compatibility follow-ups after the initial Python 3.13
metadata update. Replace the ``ODSC-TBD`` placeholders with Jira tickets before
claiming support for the corresponding surface:

* ``ODSC-TBD``: Lift the ``text`` deferral by raising the spaCy/thinc/blis
  stack.
* ``ODSC-TBD``: Lift the ``pii`` deferral by reviewing ``spacy==3.6.1``,
  ``spacy-transformers==1.2.5``, ``scrubadub==2.0.1``, and
  ``scrubadub_spacy``.
* ``ODSC-TBD``: Lift the ``forecast`` deferral by removing the NumPy 1.x
  requirement and validating ``mlforecast``, ``neuralprophet``, ``pmdarima``,
  ``prophet``, ``cmdstanpy``, ``sktime``, ``statsmodels``, and forecast
  explainers.
* ``ODSC-TBD``: Lift the ``anomaly`` deferral by validating
  ``salesforce-merlion[all]==2.0.4``, ``rrcf==0.4.4``, and the scikit-learn cap.
* ``ODSC-TBD``: Lift the ``regression`` deferral after forecast is compatible.
* ``ODSC-TBD``: Lift the ``recommender`` deferral by validating
  ``scikit-surprise`` wheels/build behavior.
* ``ODSC-TBD``: Lift the ``geo`` deferral by updating the GeoPandas/Fiona/GDAL
  stack or provisioning GDAL reliably in Python 3.13 CI environments.
* ``ODSC-TBD``: Lift the ``onnx`` deferral by updating ONNX, ONNX Runtime,
  skl2onnx, tf2onnx, xgboost, and scikit-learn constraints to a stack with
  reliable Python 3.13 wheels and runtime behavior.
* ``ODSC-TBD``: Validate Aqua notebook server extension and UI handler tests on
  Python 3.13 with ``notebook`` and ``jupyter_server`` installed.
* ``ODSC-TBD``: Validate data, database, evaluator, and BDS maximum-extra paths
  on Python 3.13 with ``oracledb`` or ``cx_Oracle``, ``mysql``, ``datefinder``,
  ``bokeh``, and related native/database dependencies installed.
* ``ODSC-TBD``: Validate LangChain, torch, autogen, and boosted/model framework
  maximum-extra paths on Python 3.13 with their full dependency stacks
  installed.
* ``ODSC-TBD``: Continue reviewing ``opctl`` dependency resolution and constrain
  ``oci-cli`` further if resolver backtracking slows Python 3.13 CI or service
  conda builds.
* ``ODSC-TBD``: Validate TensorFlow/Keras 3 behavior for ADS model artifact
  generation because Python 3.13 resolves to a much newer TensorFlow stack.

Recommended Follow-Up
---------------------

* Keep core Python 3.13 support in scope because the base package resolves.
* Validate base unit tests against the resolved NumPy 2.x, pandas 2.3.x, and
  scikit-learn 1.9.x stack in CI after the classifier update.
* Treat ``text``, ``pii``, ``forecast``, ``anomaly``, ``regression``,
  ``recommender``, ``geo``, and ``onnx`` as deferred until their dependency
  stacks are updated and validated.
* Runtime-test ``tensorflow`` because the Python 3.13 resolver selects a much
  newer TensorFlow/Keras stack.
* Continue monitoring ``opctl`` resolver behavior in CI after the
  ``oci-cli>=3.89.1`` floor is applied.
