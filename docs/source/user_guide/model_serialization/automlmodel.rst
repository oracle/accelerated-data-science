.. AutoMLModel:

AutoMLModel
***********

Overview
========

The ``AutoMLModel`` class in ADS is designed to rapidly get your AutoML model into production. The ``.prepare()`` method creates the model artifacts needed to deploy the model without you having to configure it or write code. The ``.prepare()`` method serializes the model and generates a ``runtime.yaml`` and a ``score.py`` file that you can later customize.

.. include:: _template/overview.rst

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

.. include:: _template/initialize.rst

Summary Status
==============

.. include:: _template/summary_status.rst

Model Deployment
================

Prepare
-------

The prepare step is performed by the ``.prepare()`` method. It creates several customized files that are used to run the model once it is deployed. These include:

* ``input_schema.json``: A JSON file that defines the nature of the feature data. It includes information about the features. This includes metadata such as the data type, name, constraints, summary statistics, and feature type.
* ``model.pkl``: The default file name of the serialized model.  You can change the file name with the ``model_file_name`` attribute. By default, the model is stored in a pickle file. To save your file in an ONNX format, use the ``as_onnx`` parameter.
* ``output_schema.json``: A JSON file that defines the dependent variable. This file includes metadata for the dependent variable, such as the data type, name, constraints, summary statistics, and feature type.
* ``runtime.yaml``: This file contains information needed to set up the runtime environment on the deployment server. It includes information about the conda environment used to train the model, the environment for deploying the model, and the Python version to use.
* ``score.py``: This script contains the ``load_model()`` and ``predict()`` functions. The ``load_model()`` function understands the format of the saved model and loads it into memory. The ``predict()`` function makes inferences for the deployed model. You can add hooks to perform operations before and after the inference. You can also modify this script with your specifics.

To create the model artifacts, use the ``.prepare()`` method. The ``.prepare()`` method includes parameters for storing model provenance information.

.. include:: _template/prepare.rst

Verify
------

.. include:: _template/verify.rst

* ``data (Union[dict, str])``: The data is used to test if deployment works in the local environment.

Save
----

.. include:: _template/save.rst

Deploy
------

.. include:: _template/deploy.rst

Predict
-------

.. include:: _template/predict.rst

* ``data: Any``: JSON serializable data to used for making inferences.

The ``.predict()`` and ``.verify()`` methods take the same data formats. You must ensure that the data passed into and returned by the ``predict()`` function in the ``score.py`` file is JSON serializable.

Load
====

You can restore serialization models from model artifacts, from model deployments or from models in the model catalog. This section provides details on how to restore serialization models.

.. include:: _template/loading_model_artifact.rst

.. code-block:: python3

    from ads.model.framework.automl_model import AutoMLModel

    model = AutoMLModel.from_model_artifact(
                    uri="/folder_to_your/artifact.zip",
                    model_file_name="model.pkl",
                    artifact_dir="/folder_store_artifact"
                )

.. include:: _template/loading_model_catalog.rst

.. code-block:: python3

    from ads.model.framework.automl_model import AutoMLModel

    model = AutoMLModel.from_model_catalog(model_id="<model_id>",
                        model_file_name="model.pkl",
                        artifact_dir="/folder_store_artifact")

.. include:: _template/loading_model_deployment.rst

.. code-block:: python3

    from ads.model.generic_model import AutoMLModel

    model = AutoMLModel.from_model_deployment(
        model_deployment_id="<model_deployment_id>",
        model_file_name="model.pkl",
        artifact_dir=tempfile.mkdtemp())

Delete a Deployment
===================

.. include:: _template/delete_deployment.rst

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
  from ads.catalog.model import ModelCatalog

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
          inference_conda_env="generalml_p37_cpu_v1",
          training_conda_env="generalml_p37_cpu_v1",
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
  ModelCatalog(compartment_id=os.environ['NB_SESSION_COMPARTMENT_OCID']).delete_model(model_id)

