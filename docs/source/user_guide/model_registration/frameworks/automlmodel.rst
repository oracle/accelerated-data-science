.. AutoMLModel:

AutoMLModel
***********

See `API Documentation <../../../ads.model_framework.html#ads.model.framework.automl_model.AutoMLModel>`__

Overview
========

The ``ads.model.framework.automl_model.AutoMLModel`` class in ADS is designed to rapidly get your AutoML model into production. The ``.prepare()`` method creates the model artifacts needed to deploy the model without you having to configure it or write code. The ``.prepare()`` method serializes the model and generates a ``runtime.yaml`` and a ``score.py`` file that you can later customize.

.. include:: ../_template/overview.rst

The following steps take your trained ``AutoML`` model and deploy it into production with a few lines of code.


**Creating an Oracle Labs AutoML Model**

Create an ``OracleAutoMLProvider`` object and use it to define how an Oracle Labs ``AutoML`` model is trained.

.. code-block:: python3

    import logging
    from ads.automl.driver import AutoML
    from ads.automl.provider import OracleAutoMLProvider
    from ads.dataset.dataset_browser import DatasetBrowser

    ds = DatasetBrowser.sklearn().open("wine").set_target("target")
    train, test = ds.train_test_split(test_size=0.1, random_state = 42)

    ml_engine = OracleAutoMLProvider(n_jobs=-1, loglevel=logging.ERROR)
    oracle_automl = AutoML(train, provider=ml_engine)
    model, baseline = oracle_automl.train(
                          model_list=['LogisticRegression', 'DecisionTreeClassifier'],
                          random_state = 42, time_budget = 500)


Initialize
==========

Instantiate an ``AutoMLModel()`` object with an ``AutoML`` model. Each instance accepts the following parameters:

* ``artifact_dir: str``: Artifact directory to store the files needed for deployment.
* ``auth: (Dict, optional)``: Defaults to ``None``. The default authentication is set using the ``ads.set_auth`` API. To override the default, use ``ads.common.auth.api_keys()`` or ``ads.common.auth.resource_principal()`` and create the appropriate authentication signer and the ``**kwargs`` required to instantiate the ``IdentityClient`` object.
* ``estimator: (Callable)``: Trained AutoML model.
* ``properties: (ModelProperties, optional)``: Defaults to ``None``. The ``ModelProperties`` object required to save and deploy a  model.

.. include:: ../_template/initialize.rst

Summary Status
==============

.. include:: ../_template/summary_status.rst


Example
=======

.. code-block:: python3

  import logging
  import tempfile

  from ads.automl.driver import AutoML
  from ads.automl.provider import OracleAutoMLProvider
  from ads.common.model_metadata import UseCaseType
  from ads.dataset.dataset_browser import DatasetBrowser
  from ads.model.framework.automl_model import AutoMLModel

  ds = DatasetBrowser.sklearn().open("wine").set_target("target")
  train, test = ds.train_test_split(test_size=0.1, random_state = 42)

  ml_engine = OracleAutoMLProvider(n_jobs=-1, loglevel=logging.ERROR)
  oracle_automl = AutoML(train, provider=ml_engine)
  model, baseline = oracle_automl.train(
              model_list=['LogisticRegression', 'DecisionTreeClassifier'],
              random_state = 42,
              time_budget = 500
      )

  artifact_dir = tempfile.mkdtemp()
  automl_model = AutoMLModel(estimator=model, artifact_dir=artifact_dir)
  automl_model.prepare(
          inference_conda_env="generalml_p38_cpu_v1",
          training_conda_env="generalml_p38_cpu_v1",
          use_case_type=UseCaseType.BINARY_CLASSIFICATION,
          X_sample=test.X,
          force_overwrite=True,
          training_id=None
      )
  automl_model.verify(test.X.iloc[:10])
  model_id = automl_model.save(display_name='Demo AutoMLModel model')
  deploy = automl_model.deploy(display_name='Demo AutoMLModel deployment')
  automl_model.predict(test.X.iloc[:10])
  automl_model.delete_deployment(wait_for_completion=True)

