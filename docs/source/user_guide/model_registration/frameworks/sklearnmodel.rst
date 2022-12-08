.. SklearnModel:

SklearnModel
************

See `API Documentation <../../../ads.model_framework.html#ads.model.framework.sklearn_model.SklearnModel>`__


Overview
========

The ``SklearnModel`` class in ADS is designed to allow you to rapidly get a Scikit-learn model into production. The ``.prepare()`` method creates the model artifacts that are needed to deploy a functioning model without you having to configure it or write code. However, you can customize the required ``score.py`` file.

.. include:: ../_template/overview.rst

The following steps take your trained ``scikit-learn`` model and deploy it into production with a few lines of code.

**Create a Scikit-learn Model**

.. code-block:: python3

    from sklearn.ensemble import RandomForestClassifier 
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    seed = 42

    X, y = make_classification(n_samples=10000, n_features=15, n_classes=2, flip_y=0.05)
    trainx, testx, trainy, testy = train_test_split(X, y, test_size=30, random_state=seed)
    model = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
    model.fit(
            trainx,
            trainy,
        )


Prepare Model Artifact
======================

.. code-block:: python3

    from ads.model.framework.sklearn_model import SklearnModel
    from ads.common.model_metadata import UseCaseType
    
    sklearn_model = SklearnModel(estimator=model, artifact_dir="~/sklearn_artifact_dir")
    sklearn_model.prepare(
        inference_conda_env="generalml_p38_cpu_v1",
        training_conda_env="generalml_p38_cpu_v1",
        X_sample=trainx,
        y_sample=trainy,
        use_case_type=UseCaseType.BINARY_CLASSIFICATION,
    )

Instantiate a ``ads.model.framework.sklearn_model.SklearnModel()`` object with an Scikit-learn model. Each instance accepts the following parameters:

* ``artifact_dir: str``: Artifact directory to store the files needed for deployment.
* ``auth: (Dict, optional)``: Defaults to ``None``. The default authentication is set using the ``ads.set_auth`` API. To override the default, use ``ads.common.auth.api_keys()`` or ``ads.common.auth.resource_principal()`` and create the appropriate authentication signer and the ``**kwargs`` required to instantiate the ``IdentityClient`` object.
* ``estimator: (Callable)``: Trained Scikit-learn model or Scikit-learn pipeline.
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
    >>> model_id = sklearn_model.save()

    Start loading model.joblib from model directory /tmp/tmphl0uhtbb ...
    Model is successfully loaded.
    ['output_schema.json', 'runtime.yaml', 'model.joblib', 'score.py', 'input_schema.json']

    'ocid1.datasciencemodel.oc1.xxx.xxxxx'

Deploy and Generate Endpoint
============================

.. code-block:: python3

    >>> # Deploy and create an endpoint for the Random Forest model
    >>> sklearn_model.deploy(
            display_name="Random Forest Model For Classification",
            deployment_log_group_id="ocid1.loggroup.oc1.xxx.xxxxx",
            deployment_access_log_id="ocid1.log.oc1.xxx.xxxxx",
            deployment_predict_log_id="ocid1.log.oc1.xxx.xxxxx",
        )
    >>> print(f"Endpoint: {sklearn_model.model_deployment.url}")
    https://modeldeployment.{region}.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx


Run Prediction against Endpoint
===============================

.. code-block:: python3

    >>> # Generate prediction by invoking the deployed endpoint
    >>> sklearn_model.predict(testx)['prediction']
    [1,0,...,1]


Examples
========

.. code-block:: python3

    from ads.model.framework.sklearn_model import SklearnModel
    from ads.common.model_metadata import UseCaseType

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    import tempfile

    seed = 42

    # Create a classification dataset
    X, y = make_classification(n_samples=10000, n_features=15, n_classes=2, flip_y=0.05)
    trainx, testx, trainy, testy = train_test_split(X, y, test_size=30, random_state=seed)

    # Train LGBM model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(
        trainx,
        trainy,
    )


    # Deploy the model, test it and clean up.
    # Prepare Model Artifact for RandomForest Classifier model
    artifact_dir = tempfile.mkdtemp()
    sklearn_model = SklearnModel(estimator=model, artifact_dir=artifact_dir)
    sklearn_model.prepare(
        inference_conda_env="generalml_p38_cpu_v1",
        training_conda_env="generalml_p38_cpu_v1",
        use_case_type=UseCaseType.BINARY_CLASSIFICATION,
        X_sample=trainx,
        y_sample=trainy,
        force_overwrite=True,
    )

    # Check if the artifacts are generated correctly.
    # The verify method invokes the ``predict`` function defined inside ``score.py`` in the artifact_dir

    sklearn_model.verify(testx[:10])["prediction"]
    sklearn_model.save(display_name="SKLearn Model")

    # Deploy and create an endpoint for the RandomForest model
    sklearn_model.deploy(
        display_name="Random Forest Model For Classification",
        deployment_log_group_id="ocid1.loggroup.oc1.xxx.xxxx",
        deployment_access_log_id="ocid1.log.oc1.xxx.xxxx",
        deployment_predict_log_id="ocid1.log.oc1.xxx.xxxx",
    )


    print(f"Endpoint: {sklearn_model.model_deployment.url}")

    sklearn_model.predict(testx)["prediction"]

    # To delete the deployed endpoint uncomment the following line
    # sklearn_model.delete_deployment(wait_for_completion=True)


