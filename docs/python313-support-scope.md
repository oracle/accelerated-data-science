# ADS Python 3.13 Support Scope

This audit supports ODSC-88492 and records the Python 3.13 readiness surface
used to update package metadata and CI coverage.

## Current Support Markers

- `pyproject.toml` sets `requires-python = ">=3.8"` and now advertises
  classifiers through Python 3.13 after the validation below passed.
- Default unit CI in `.github/workflows/run-unittests-default_setup.yml` now
  runs Python 3.9, 3.10, 3.11, 3.12, and 3.13.
- Full unit CI in `.github/workflows/run-unittests-py310-py311.yml` now runs
  Python 3.10, 3.11, 3.12, and 3.13.
- Operator, forecast, and forecast explainer CI workflows run Python 3.10 and
  3.11 only.
- `docs/source/user_guide/cli/quickstart.rst` now documents CLI support as
  Python >=3.8, <=3.13.
- Model-artifact validation tests now classify Python 3.13 as supported in
  `tests/unitary/default_setup/model/test_model_introspect.py`.
- Several model framework tests are skipped on Python 3.12+ for dependency or
  framework compatibility, including TensorFlow, sklearn export, xgboost export,
  and default model artifact tests.
- Low-code operator metadata pins runtime Python versions below 3.13:
  regression, anomaly, and forecast `MLoperator` files use Python 3.11; PII and
  recommender conda environment files use Python 3.9.

## Dependency Surfaces To Validate

Validate installation and tests in this order so support is not advertised ahead
of dependency resolution:

1. Core package install from `pyproject.toml` `[project.dependencies]`.
2. Default unit test environment from `test-requirements.txt`.
3. Full development environment from `dev-requirements.txt`, which installs
   `.[aqua,bds,data,geo,huggingface,llm,notebook,onnx,opctl,optuna,spark,tensorflow,text,torch,viz]`
   plus `.[testsuite]`.
4. Operator service environment from `test-requirements-operators.txt`, which
   installs `.[forecast]` and `.[anomaly]` plus operator test dependencies.

The optional extras with explicit Python-version-sensitive or high-risk runtime
constraints are:

- `onnx`: `onnx`, `onnxruntime`, and `skl2onnx` switch pins at Python 3.12 and
  should be re-resolved specifically on Python 3.13.
- `tensorflow`: TensorFlow switches at Python 3.12 and must remain compatible
  with the Keras/tf2onnx behavior noted in the existing comments.
- `huggingface`: `tf-keras` is present for Keras 3 compatibility and should be
  validated with the TensorFlow surface.
- `text` and `pii`: spaCy is constrained below 3.8 in `text`, while `pii` pins
  `spacy==3.6.1`, `spacy-transformers==1.2.5`, `scrubadub==2.0.1`, and
  `scrubadub_spacy`.
- `forecast`, `anomaly`, and `regression`: these carry the largest operator
  dependency surface, including `numpy<2.0.0`, `pmdarima`, `prophet`,
  `cmdstanpy`, `pytorch-lightning`, `salesforce-merlion[all]`, and shared
  `opctl` dependencies. On Python 3.13, `neuralprophet` and
  `salesforce-merlion[all]` are deferred because their current releases require
  NumPy 1.x, which has no wheel-backed Python 3.13 install path in this
  validation environment.
- `boosted`, `notebook`, and `onnx`: these keep `scikit-learn<1.6.0`; validate
  whether that bound still resolves on Python 3.13.
- `geo`: `geopandas<1.0.0` and the Python 3.13 Fiona dependency line need
  wheel and dependency resolution checks on Python 3.13.
- `testsuite`: validate all pinned test dependencies, especially
  `category_encoders==2.6.3`, `cohere==4.53`, `fastparquet==2024.2.0`,
  `notebook==6.4.12`, `pyarrow>=15.0.0`, and `tables>3.9.0`.
- `aqua`: validate `openai==1.109.1`, `notebook>=6.4,<=6.6`, and shared
  `huggingface_hub` constraints.

## Initial Support Scope

Treat the following as in scope for ODSC-88492:

- Core ADS installation on Python 3.13 from source and wheel.
- Default setup unit tests in `tests/unitary/default_setup`.
- Full unitary coverage from `.github/workflows/run-unittests-py310-py311.yml`,
  including `tests/unitary` and `tests/unitary/with_extras/model` after the
  dependency set resolves.
- Model artifact and runtime metadata validation for Python 3.13, including
  `inference_python_version`, `training_python_version`, service conda metadata
  parsing, and model introspection validation.
- Required service conda optional extras represented by `dev-requirements.txt`
  and `test-requirements-operators.txt`.
- CI matrix updates for default setup and full unit validation after the above
  install and test surfaces pass.
- Package classifier update and any license metadata updates after dependency
  changes are finalized.

## Deferred Or Separately Tracked Scope

Do not block initial Python 3.13 metadata on these areas unless service conda
requirements explicitly make them mandatory:

- Recommender operator tests, because `test-requirements-operators.txt` leaves
  `.[recommender]` commented out today.
- Forecast explainer coverage beyond the existing dedicated workflow, if its
  third-party explainer stack does not resolve on Python 3.13 with the required
  service environment.
- NeuralProphet-backed forecast functionality and Salesforce Merlion-backed
  anomaly functionality on Python 3.13, until upstream releases support a
  Python 3.13-compatible NumPy dependency line.
- PII operator functionality on Python 3.13, because the current extra pins
  `spacy==3.6.1`, `spacy-transformers==1.2.5`, `scrubadub==2.0.1`, and
  `scrubadub_spacy`, which remain tied to the older spaCy stack.
- Low-code operator runtime version bumps from Python 3.9 or 3.11 to 3.13.
  Those YAML/MLoperator defaults affect generated operator environments and
  should be handled only after service conda runtime images are available.
- Historical model examples and fixture values that intentionally reference
  older Python versions.

## Validation Handoff

Later beads should record exact commands and results for:

- `python -m pip install -e .`
- `python -m pip install -r test-requirements.txt`
- `python -m pip install -r dev-requirements.txt`
- `python -m pip install -r test-requirements-operators.txt`
- `python -m pytest -v -p no:warnings --durations=5 tests/unitary/default_setup`
- Full unitary workflow-equivalent coverage from
  `.github/workflows/run-unittests-py310-py311.yml`.
- Targeted model artifact/runtime tests covering Python 3.13 acceptance.
- Source distribution and wheel build/install validation.
