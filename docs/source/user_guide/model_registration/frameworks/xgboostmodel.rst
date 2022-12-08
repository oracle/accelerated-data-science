.. XGBoostModel:

XGBoostModel
************

See `API Documentation <../../../ads.model_framework.html#ads.model.framework.xgboost_model.XGBoostModel>`__

Overview
========

The ``ads.model.framework.xgboost_model.XGBoostModel`` class in ADS is designed to allow you to rapidly get a XGBoost model into production. The ``.prepare()`` method creates the model artifacts that are needed to deploy a functioning model without you having to configure it or write code. However, you can customize the required ``score.py`` file.

.. include:: ../_template/overview.rst

The following steps take your trained ``XGBoost`` model and deploy it into production with a few lines of code.

The ``XGBoostModel`` module in ADS supports serialization for models generated from both the  `Learning API <https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training>`_ using ``xgboost.train()`` and the `Scikit-Learn API <https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn>`_ using ``xgboost.XGBClassifier()``. Both of these interfaces are defined by `XGBoost <https://xgboost.readthedocs.io/en/stable/index.html>`_.

**Create XGBoost Model**

.. code-block:: python3

    from ads.model.framework.xgboost_model import XGBoostModel
    from ads.common.model_metadata import UseCaseType

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    import tempfile

    import xgboost

    seed = 42

    X, y = make_classification(n_samples=10000, n_features=15, n_classes=2, flip_y=0.05)
    trainx, testx, trainy, testy = train_test_split(X, y, test_size=30, random_state=seed)
    model = xgboost.XGBClassifier(
            n_estimators=100, learning_rate=0.01, random_state=42, use_label_encoder=False
        )
    model.fit(
            trainx,
            trainy,
        )


Prepare Model Artifact
======================

.. code-block:: python3

    from ads.model.framework.xgboost_model import XGBoostModel
    from ads.common.model_metadata import UseCaseType

    artifact_dir = tempfile.mkdtemp()
    xgb_model = XGBoostModel(estimator=model, artifact_dir=artifact_dir)
    xgb_model.prepare(
        inference_conda_env="generalml_p38_cpu_v1",
        training_conda_env="generalml_p38_cpu_v1",
        X_sample=trainx,
        y_sample=trainy,
        use_case_type=UseCaseType.BINARY_CLASSIFICATION,
    )

Instantiate a ``ads.model.framework.xgboost_model.XGBoostModel`` object with an XGBoost model. Each instance accepts the following parameters:

* ``artifact_dir: str``: Artifact directory to store the files needed for deployment.
* ``auth: (Dict, optional)``: Defaults to ``None``. The default authentication is set using the ``ads.set_auth`` API. To override the default, use ``ads.common.auth.api_keys()`` or ``ads.common.auth.resource_principal()`` and create the appropriate authentication signer and the ``**kwargs`` required to instantiate the ``IdentityClient`` object.
* ``estimator: (Callable)``: Trained XGBoost model either using the Learning API or the Scikit-Learn Wrapper interface.
* ``properties: (ModelProperties, optional)``: Defaults to ``None``. The ``ModelProperties`` object required to save and deploy a  model.

.. include:: ../_template/initialize.rst

Summary Status
==============

.. include:: ../_template/summary_status.rst

.. figure:: ../figures/summary_status.png
   :align: center

Register Model
==============

.. code-block:: python3

    >>> # Register the model
    >>> model_id = xgb_model.save()

    Start loading model.joblib from model directory /tmp/tmphl0uhtbb ...
    Model is successfully loaded.
    ['runtime.yaml', 'model.joblib', 'score.py', 'input_schema.json']

    'ocid1.datasciencemodel.oc1.xxx.xxxxx'

Deploy and Generate Endpoint
============================

.. code-block:: python3

    >>> # Deploy and create an endpoint for the XGBoost model
    >>> xgb_model.deploy(
            display_name="XGBoost Model For Classification",
            deployment_log_group_id="ocid1.loggroup.oc1.xxx.xxxxx",
            deployment_access_log_id="ocid1.log.oc1.xxx.xxxxx",
            deployment_predict_log_id="ocid1.log.oc1.xxx.xxxxx",
        )


    >>> print(f"Endpoint: {xgb_model.model_deployment.url}")

    https://modeldeployment.{region}.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx


Run Prediction against Endpoint
===============================

.. code-block:: python3

    # Generate prediction by invoking the deployed endpoint
    >>> xgb_model.predict(testx)['prediction']
    [0.22879330813884735, 0.2054443359375, 0.20657016336917877,...,0.8005291223526001]

Example
=======

.. code-block:: python3

    from ads.model.framework.xgboost_model import XGBoostModel
    from ads.common.model_metadata import UseCaseType

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    import tempfile

    import xgboost

    seed = 42

    # Create a classification dataset
    X, y = make_classification(n_samples=10000, n_features=15, n_classes=2, flip_y=0.05)
    trainx, testx, trainy, testy = train_test_split(X, y, test_size=30, random_state=seed)

    # Train XGBoost model
    model = xgboost.XGBClassifier(n_estimators=100, learning_rate=0.01, random_state=42)
    model.fit(
        trainx,
        trainy,
    )

    artifact_dir = tempfile.mkdtemp()
    xgb_model = XGBoostModel(estimator=model, artifact_dir=artifact_dir)
    xgb_model.prepare(
        inference_conda_env="generalml_p38_cpu_v1",
        training_conda_env="generalml_p38_cpu_v1",
        X_sample=trainx,
        y_sample=trainy,
        use_case_type=UseCaseType.BINARY_CLASSIFICATION,
    )

    # Check if the artifacts are generated correctly.
    # The verify method invokes the ``predict`` function defined inside ``score.py`` in the artifact_dir
    xgb_model.verify(testx)

    # Register the model
    model_id = xgb_model.save(display_name="XGBoost Model")

    # Deploy and create an endpoint for the XGBoost model
    xgb_model.deploy(
        display_name="XGBoost Model For Classification",
        deployment_log_group_id="ocid1.loggroup.oc1.xxx.xxxxx",
        deployment_access_log_id="ocid1.log.oc1.xxx.xxxxx",
        deployment_predict_log_id="ocid1.log.oc1.xxx.xxxxx",
    )

    print(f"Endpoint: {xgb_model.model_deployment.url}")

    # Generate prediction by invoking the deployed endpoint
    xgb_model.predict(testx)["prediction"]

    # To delete the deployed endpoint uncomment the line below
    # xgb_model.delete_deployment(wait_for_completion=True)

