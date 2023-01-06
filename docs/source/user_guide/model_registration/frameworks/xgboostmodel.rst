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
    from ads.model.model_metadata import UseCaseType

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
    from ads.model.model_metadata import UseCaseType

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

Run Prediction with oci raw-request command
===========================================

Model deployment endpoints can be invoked with the OCI-CLI. The below examples invoke a model deployment with the CLI with different types of payload: ``json``,  ``numpy.ndarray``, ``pandas.core.frame.DataFrame`` or ``dict``.

`json` payload example
----------------------

.. code-block:: python3

    >>> # Prepare data sample for prediction
    >>> data = testx[[12]]
    >>> data
    array([[ 0.66098176, -1.06487896, -0.88581208,  0.05667259,  0.42884393,
        -0.52552184,  0.75322749, -0.58112776, -0.81102029, -1.35854886,
        -1.16440502, -0.67791303, -0.04810906,  0.72970972, -0.24120756]])

Use printed output of the data and endpoint to invoke prediction with raw-request command in terminal:

.. code-block:: bash

    export uri=https://modeldeployment.{region}.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx/predict
    export data='{"data": [[ 0.66098176, -1.06487896, ... , -0.24120756]]}'
    oci raw-request \
        --http-method POST \
        --target-uri $uri \
        --request-body "$data"

`numpy.ndarray` payload example
-------------------------------

.. code-block:: python3

    >>> # Prepare data sample for prediction
    >>> from io import BytesIO
    >>> import base64
    >>> import numpy as np

    >>> data = testx[[12]]
    >>> np_bytes = BytesIO()
    >>> np.save(np_bytes, data, allow_pickle=True)
    >>> data = base64.b64encode(np_bytes.getvalue()).decode("utf-8")
    >>> print(data)
    k05VTVBZAQB2AHsnZGVzY......pePfzr8=

Use printed output of ``base64`` data and endpoint to invoke prediction with raw-request command in terminal:

.. code-block:: bash

    export uri=https://modeldeployment.{region}.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx/predict
    export data='{"data":"k05VTVBZAQB2AHsnZGVzY......pePfzr8=", "data_type": "numpy.ndarray"}'
    oci raw-request \
        --http-method POST \
        --target-uri $uri \
        --request-body "$data"

`pandas.core.frame.DataFrame` payload example
---------------------------------------------

.. code-block:: python3

    >>> # Prepare data sample for prediction
    >>> import pandas as pd

    >>> df = pd.DataFrame(testx[[12]])
    >>> print(json.dumps(df.to_json())
    "{\"0\":{\"0\":0.6609817554},\"1\":{\"0\":-1.0648789569},...,\"14\":{\"0\":-0.2412075575}}"

Use printed output of `DataFrame` data and endpoint to invoke prediction with raw-request command in terminal:

.. code-block:: bash

    export uri=https://modeldeployment.{region}.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx/predict
    export data='{"data":"{\"0\":{\"0\":0.6609817554},...,\"14\":{\"0\":-0.2412075575}}","data_type":"pandas.core.frame.DataFrame"}'
    oci raw-request \
        --http-method POST \
        --target-uri $uri \
        --request-body "$data"

`dict` payload example
----------------------

    >>> # Prepare data sample for prediction
    >>> import pandas as pd

    >>> df = pd.DataFrame(testx[[12]])
    >>> print(json.dumps(df.to_dict()))
    {"0": {"0": -0.6712208871908425}, "1": {"0": 0.5266565978285116}, ...,"14": {"0": 0.9062102978188604}}

Use printed output of ``dict`` data and endpoint to invoke prediction with raw-request command in terminal:

.. code-block:: bash

    export uri=https://modeldeployment.{region}.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx/predict
    export data='{"data": {"0": {"0": -0.6712208871908425}, ...,"14": {"0": 0.9062102978188604}}}'
    oci raw-request \
        --http-method POST \
        --target-uri $uri \
        --request-body "$data"

Expected output of raw-request command
--------------------------------------

.. code-block:: bash

    {
      "data": {
        "prediction": [
          0.5611757040023804
        ]
      },
      "headers": {
        "Connection": "keep-alive",
        "Content-Length": "35",
        "Content-Type": "application/json",
        "Date": "Wed, 07 Dec 2022 17:27:17 GMT",
        "X-Content-Type-Options": "nosniff",
        "opc-request-id": "19E90A7F2AFE401BB437DBC6168D2F1C/4A1CC9969F40B9F0656DD0497A28B51A/FEB42D1B690E8A665244046C7A151AB5",
        "server": "uvicorn"
      },
      "status": "200 OK"
    }

Example
=======

.. code-block:: python3

    from ads.model.framework.xgboost_model import XGBoostModel
    from ads.model.model_metadata import UseCaseType

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

