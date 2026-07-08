Python 3.13 Dependency Audit
============================

This audit records the Python 3.13 dependency surface reviewed for
ODSC-88492. It is intentionally limited to dependency readiness and follow-up
targets; package metadata should not advertise Python 3.13 until install,
runtime, CI, and scoped test validation are complete.

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
  Resolves on Python 3.13.13. The current marker branch
  ``python_version >= '3.12'`` selected ``onnx==1.17.0``,
  ``onnxruntime==1.22.1``, ``skl2onnx==1.18.0``, ``tf2onnx==1.17.0``,
  ``xgboost==1.6.2``, and ``scikit-learn==1.5.2`` because the extra caps
  scikit-learn below 1.6. The ONNX path should still be runtime-tested because
  ONNX itself built from source in this environment rather than using a wheel.

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

Existing Python-Version Gates
-----------------------------

Current Python-version-specific dependency markers are concentrated in the
ONNX and TensorFlow extras:

* ``onnx>=1.12.0,<=1.15.0; python_version < '3.12'``
* ``onnx~=1.17.0; python_version >= '3.12'``
* ``onnxruntime~=1.17.0,!=1.16.0; python_version < '3.12'``
* ``onnxruntime~=1.22.0; python_version >= '3.12'``
* ``skl2onnx>=1.10.4; python_version < '3.12'``
* ``skl2onnx~=1.18.0; python_version >= '3.12'``
* ``tensorflow<=2.15.1; python_version < '3.12'``
* ``tensorflow; python_version >= '3.12'``

No Python 3.13 classifier is currently present, which is correct until the
remaining validation criteria are complete.

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
* ``onnx`` installability and ONNX serializer/runtime validation after the
  dependency constraints are reviewed in the next step.
* ``tensorflow`` installability and TensorFlow model artifact validation after
  the dependency constraints are reviewed in the next step.
* ``viz``, ``data``, ``notebook``, ``llm``, ``aqua``, ``torch``,
  ``huggingface``, ``spark``, ``optuna``, ``boosted``, ``bds``, and ``geo`` only
  after each extra has a Python 3.13 resolver check. They were not proven by
  this audit and should not be assumed supported from the base resolver result.

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
* Forecast explainer tests because they depend on the forecast stack and should
  follow forecast support rather than block core support.
* Operator workflows backed by deferred extras. Operator framework utilities and
  YAML generation can stay in scope if they do not import or install deferred
  runtime stacks.

Validation Required Before Advertising Support
----------------------------------------------

Do not add the Python 3.13 package classifier until all initial-scope validation
is complete. At minimum, validation should include:

* Build editable metadata and source/wheel artifacts on Python 3.13.
* Install the base package in a clean Python 3.13 environment.
* Run default setup unit tests on Python 3.13.
* Run targeted model artifact/runtime tests, especially
  ``tests/unitary/default_setup/model`` and ``tests/unitary/with_extras/model``.
* Run ONNX and TensorFlow model serialization tests if those extras remain in
  the initial support scope.
* Run an ``opctl`` install check and a focused command/import smoke test.
* Confirm deferred extras are documented with follow-up tickets and, where
  needed, Python 3.13 environment markers.

Follow-Up Ticket Candidates
---------------------------

Create separate follow-up tickets for these compatibility gaps if they cannot
be resolved in the initial Python 3.13 support change:

* Update or defer ``text`` for Python 3.13 by raising the spaCy/thinc/blis stack
  or adding explicit Python 3.13 exclusion markers.
* Update or defer ``pii`` for Python 3.13 by reviewing ``spacy==3.6.1``,
  ``spacy-transformers==1.2.5``, ``scrubadub==2.0.1``, and
  ``scrubadub_spacy``.
* Update or defer ``forecast`` for Python 3.13 by removing the NumPy 1.x
  requirement and validating ``mlforecast``, ``neuralprophet``, ``pmdarima``,
  ``prophet``, ``cmdstanpy``, ``sktime``, ``statsmodels``, and forecast
  explainers.
* Update or defer ``anomaly`` for Python 3.13 by validating
  ``salesforce-merlion[all]==2.0.4``, ``rrcf==0.4.4``, and the scikit-learn cap.
* Update or defer ``recommender`` for Python 3.13 by validating
  ``scikit-surprise`` wheels/build behavior.
* Review ``opctl`` dependency resolution and constrain ``oci-cli`` if resolver
  backtracking slows Python 3.13 CI or service conda builds.
* Validate ``onnx`` wheel/build behavior on Linux Python 3.13 and adjust ONNX,
  ONNX Runtime, skl2onnx, xgboost, and scikit-learn constraints as needed.
* Validate TensorFlow/Keras 3 behavior for ADS model artifact generation because
  Python 3.13 resolves to a much newer TensorFlow stack.

Recommended Follow-Up
---------------------

* Keep core Python 3.13 support in scope because the base package resolves.
* Validate base unit tests against the resolved NumPy 2.x, pandas 2.3.x, and
  scikit-learn 1.9.x stack before changing package classifiers.
* Update or defer the ``text`` extra because the current spaCy cap fails on
  Python 3.13.
* Treat ``forecast`` and ``anomaly`` as deferred unless their NumPy 1.x and
  older ML-package constraints are updated.
* Runtime-test ``onnx`` despite resolver success because ONNX may build from
  source on Python 3.13 in some environments.
* Runtime-test ``tensorflow`` before support is advertised because the
  Python 3.13 resolver selects a much newer TensorFlow/Keras stack.
* Review ``opctl`` resolver backtracking and consider constraining ``oci-cli``
  compatibility for faster CI and service conda builds.
