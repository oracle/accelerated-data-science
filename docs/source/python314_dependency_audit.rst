Python 3.14 Dependency Readiness Audit
======================================

This audit supports ODSC-88493. It records the current dependency-resolution
state for ADS on Python 3.14 before package metadata advertises Python 3.14
support.

Audit environment
-----------------

* Date: 2026-07-15
* Platform: macOS arm64
* Python: 3.14.5
* Resolver: pip 26.1.1
* Source: local checkout on branch ``csa/odsc-88493``
* Binary policy: ``--only-binary=:all:`` for optional-extra checks to model
  wheel availability. Core was checked without that flag and resolved to
  Python 3.14 wheels for compiled dependencies.

Commands used
-------------

.. code-block:: shell

   python3.14 -m venv .venv-py314-audit
   .venv-py314-audit/bin/python -m pip install --dry-run --ignore-installed .
   .venv-py314-audit/bin/python -m pip install --dry-run --ignore-installed --only-binary=:all: ".[<extra>]"

Core dependency result
----------------------

Core ADS dependencies resolve on Python 3.14. The resolver selected Python
3.14-compatible wheels for the main compiled dependencies, including
``numpy``, ``pandas``, ``matplotlib``, ``scikit-learn``, ``scipy``, ``PyYAML``,
``pydantic-core``, ``cryptography``, and ``oracledb`` transitive dependencies.

This is only a resolver result. It does not validate runtime behavior, import
coverage, model artifact behavior, or service-conda compatibility.

Agreed Python 3.14 support scope
--------------------------------

Python 3.14 support should be staged. The initial support claim should cover
only surfaces that resolve on Python 3.14 and have targeted runtime or unit-test
validation. Package classifiers should remain unchanged until that validation is
complete.

Initial support candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following surfaces are candidates for the first Python 3.14 support claim:

* Core ADS install from source and wheel, including default dependencies.
* Default setup unit tests that do not require optional extras or service
  resources.
* Model artifact/runtime metadata paths that only parse, preserve, and emit
  explicit Python version strings, including ``INFERENCE_PYTHON_VERSION`` and
  training Python version fields.
* Jobs and model APIs that accept service-conda slugs or full OCI conda paths
  without importing unresolved optional stacks.
* Optional groups that resolved in the Python 3.14 audit and still require
  targeted import/runtime tests before support is advertised: ``aqua``,
  ``huggingface``, ``llm``, ``optuna``, ``torch``, and ``viz``.

Validation-gated support
~~~~~~~~~~~~~~~~~~~~~~~~

These areas remain in scope for Python 3.14 only after dependency metadata and
runtime tests are updated:

* Data access extras that need a SQLAlchemy 2.x compatibility decision.
* Notebook and boosted model tooling that currently inherits the
  ``scikit-learn<1.6.0`` cap.
* ONNX model serialization and conversion after a Python 3.14-specific ONNX,
  ONNX Runtime, ``skl2onnx``, and ``tf2onnx`` combination is selected.
* Text and PII features after the spaCy dependency stack is moved to Python
  3.14-compatible releases.
* Geo features after the ``geopandas``/``fiona`` pin strategy is updated.
* HPO, LLM, AQUA, Hugging Face, Torch, and visualization paths after targeted
  import and behavioral tests pass.

Deferred or unsupported until follow-up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following areas should not be included in an initial Python 3.14 support
claim unless service-conda owners confirm compatible internal builds and ADS
captures that decision in a follow-up ticket:

* BDS, because ``hdfs[kerberos]``/``docopt`` did not resolve with Python 3.14
  binary availability in the audit.
* Spark, because ``pyspark>=3.0.0`` did not resolve under the Python 3.14 binary
  audit policy.
* TensorFlow, because no Python 3.14 TensorFlow binary match was available for
  the current unconstrained ``python_version >= "3.12"`` dependency.
* Low-code forecast, anomaly, recommender, and regression operators, because
  they inherit blockers from ``opctl``, ``forecast``, ``rrcf``, and NumPy pins.
* Full ``testsuite`` coverage until ``arff`` and any subsequent Python 3.14 test
  dependency blockers are replaced, scoped, or deferred.

Service-conda coordination requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any service-conda-required group that remains blocked must have one of the
following before Python 3.14 support is advertised:

* a metadata update in ``pyproject.toml`` plus passing Python 3.14 install and
  runtime validation,
* a documented service-conda-only support decision naming the compatible
  service environment, or
* a follow-up ticket that records the deferred group, the blocking dependency,
  and the user-visible limitation.

For model artifacts and runtime metadata, Python 3.14 should be accepted only
where the selected inference or training conda environment is also validated for
Python 3.14. ADS should preserve full OCI conda paths directly and should
resolve service-conda slugs through the service index before using the
environment Python version.

Optional-extra resolver results
-------------------------------

.. list-table::
   :header-rows: 1

   * - Extra
     - Result
     - Finding
     - Follow-up
   * - ``aqua``
     - Resolves
     - Current pins resolve, including ``openai==1.109.1`` and
       ``huggingface_hub>=0.36.2,<0.37``.
     - Run AQUA import/unit tests before declaring support.
   * - ``anomaly``
     - Blocked
     - ``rrcf==0.4.4`` has no Python 3.14 binary match; this group also
       inherits the shared ``opctl`` resolver blocker through
       ``oracle_ads[opctl]``.
     - Defer or update the anomaly operator stack after validating replacement
       dependencies.
   * - ``bds``
     - Blocked
     - ``hdfs[kerberos]`` requires ``docopt``; no matching Python 3.14 binary
       distribution was available under the audit policy.
     - Decide whether BDS is in the Python 3.14 support scope or defer with a
       follow-up ticket.
   * - ``boosted``
     - Blocked
     - ``scikit-learn>=1.0,<1.6.0`` excludes Python 3.14-capable
       ``scikit-learn`` wheels selected by core resolution.
     - Relax or split the ``scikit-learn`` cap for Python 3.14 after behavioral
       validation.
   * - ``data``
     - Blocked
     - ``sqlalchemy>=1.4.1,<=1.4.46`` has no Python 3.14 binary match; the
       resolver only found SQLAlchemy 2.x candidates.
     - Validate ADS database/data code with SQLAlchemy 2.x or add a Python
       3.14 exclusion.
   * - ``forecast``
     - Blocked
     - ``numpy<2.0.0`` conflicts with the Python 3.14 NumPy wheel set selected
       by the resolver.
     - Treat forecast as service-conda-gated until NumPy 2.x compatibility is
       validated or an internal compatible environment is confirmed.
   * - ``geo``
     - Blocked
     - ``fiona<=1.9.6`` has no Python 3.14 binary match.
     - Revisit the ``geopandas``/``fiona`` caps for Python 3.14, or defer geo
       support.
   * - ``huggingface``
     - Resolves
     - Resolver selected current ``transformers`` and ``tf-keras`` packages.
     - Run import/runtime tests because this stack is sensitive to transitive
       version movement.
   * - ``llm``
     - Resolves
     - Resolver selected current LangChain/OpenAI-compatible packages.
     - Run targeted LLM serialization/client tests.
   * - ``notebook``
     - Blocked
     - Shares the ``scikit-learn>=1.0,<1.6.0`` blocker with ``boosted``.
     - Align with the ``scikit-learn`` Python 3.14 decision.
   * - ``onnx``
     - Blocked
     - ``onnx~=1.17.0`` for ``python_version >= "3.12"`` has no Python 3.14
       binary match; the resolver found newer ONNX candidates only.
     - Add a Python 3.14-specific marker after validating newer ONNX,
       ONNX Runtime, ``skl2onnx``, and ``tf2onnx`` combinations.
   * - ``opctl``
     - Blocked
     - ``oci-cli`` resolution conflicts with current ``oci``/``PyYAML``
       availability for Python 3.14.
     - Validate whether service-conda environments provide a compatible
       ``oci-cli`` stack or split/exclude ``oci-cli`` on Python 3.14.
   * - ``optuna``
     - Resolves
     - ``optuna==2.9.0`` resolves with ``oracle_ads[viz]``.
     - Run HPO tests before declaring support.
   * - ``pii``
     - Blocked
     - ``spacy-transformers==1.2.5`` has no Python 3.14 binary match.
     - Upgrade the spaCy transformer stack or defer PII support.
   * - ``recommender``
     - Blocked
     - Inherits the shared ``opctl`` resolver blocker.
     - Resolve or defer through the ``opctl`` service-conda decision.
   * - ``regression``
     - Blocked
     - Inherits the ``forecast`` ``numpy<2.0.0`` blocker.
     - Resolve or defer through the forecast/operator decision.
   * - ``spark``
     - Blocked
     - ``pyspark>=3.0.0`` had no Python 3.14 binary match under the audit
       policy.
     - Confirm whether Spark is service-conda-only for Python 3.14 or requires
       a PyPI marker.
   * - ``tensorflow``
     - Blocked
     - The unconstrained ``tensorflow`` dependency for
       ``python_version >= "3.12"`` had no Python 3.14 binary match.
     - Defer TensorFlow support or add a Python 3.14 marker once compatible
       wheels exist.
   * - ``testsuite``
     - Blocked
     - ``arff`` has no Python 3.14 binary match under the audit policy.
     - Replace or scope the test dependency before full Python 3.14 unit-test
       coverage.
   * - ``text``
     - Blocked
     - ``spacy>=3.4.2,<3.8`` excludes Python 3.14-compatible spaCy candidates
       found by the resolver.
     - Validate spaCy 3.8+ behavior or defer text support.
   * - ``torch``
     - Resolves
     - Resolver selected Python 3.14-compatible ``torch`` and ``torchvision``
       wheels on macOS arm64.
     - Run model framework import/serialization tests.
   * - ``viz``
     - Resolves
     - Current visualization dependencies resolve.
     - Run plotting/import smoke tests.

Service-conda-relevant groups
-----------------------------

The service-facing groups with immediate Python 3.14 resolver blockers are
``opctl``, ``forecast``, ``anomaly``, ``recommender``, ``regression``, ``spark``,
``tensorflow``, and ``onnx``. These should not be treated as Python 3.14-ready
until either:

* the package metadata is updated and validated against Python 3.14, or
* the service-conda environment owners confirm compatible internal builds and
  ADS scopes those groups as service-conda-only for Python 3.14.

Recommended next steps
----------------------

* Keep package classifiers unchanged until dependency and runtime validation
  complete.
* Add Python 3.14-specific dependency markers for blockers that only need newer
  releases, especially ``scikit-learn``, ``onnx``, ``spacy``, ``fiona``, and
  ``sqlalchemy``.
* Create explicit follow-up tickets for groups that are not in the initial
  Python 3.14 support scope, especially TensorFlow, Spark, forecast/anomaly
  operators, and BDS.
* Run targeted import/runtime tests for resolver-passing groups:
  ``aqua``, ``huggingface``, ``llm``, ``optuna``, ``torch``, and ``viz``.
* Re-run the resolver audit on the target Linux service-conda platform before
  enabling CI or advertising Python 3.14 package metadata.
