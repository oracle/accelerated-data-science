.. LightGBMModel:

LightGBMModel
*************

See `API Documentation <../../../ads.model_framework.html#ads.model.framework.lightgbm_model.LightGBMModel>`__


Overview
========

The ``ads.model.framework.lightgbm_model.LightGBMModel`` class in ADS is designed to allow you to rapidly get a LightGBM model into production. The ``.prepare()`` method creates the model artifacts that are needed to deploy a functioning model without you having to configure it or write code. However, you can customize the required ``score.py`` file.

.. include:: ../_template/overview.rst

The following steps take your trained ``LightGBM`` model and deploy it into production with a few lines of code.

The ``LightGBMModel`` module in ADS supports serialization for models generated from both the  `Training API <https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api>`_ using ``lightgbm.train()`` and the `Scikit-Learn API <https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api>`_ using ``lightgbm.LGBMClassifier()``. Both of these interfaces are defined by `LightGBM <https://lightgbm.readthedocs.io/en/latest/>`_.

The Training API in ``LightGBM`` contains training and cross-validation routines. The ``Dataset`` class is an internal data structure that is used by LightGBM when using the ``lightgbm.train()`` method. You can also create LightGBM models using the Scikit-Learn Wrapper interface. The `LightGBMModel` class handles the differences between the LightGBM Training and SciKit-Learn APIs seamlessly.

**Create LightGBM Model**

.. code-block:: python3

    import lightgbm as lgb
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    seed = 42

    X, y = make_classification(n_samples=10000, n_features=15, n_classes=2, flip_y=0.05)
    trainx, testx, trainy, testy = train_test_split(X, y, test_size=30, random_state=seed)
    model = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.01, random_state=42
        )
    model.fit(
            trainx,
            trainy,
        )
    

Prepare Model Artifact
======================

.. code-block:: python3

    from ads.common.model_metadata import UseCaseType
    from ads.model.framework.lightgbm_model import LightGBMModel

    artifact_dir = tempfile.mkdtemp()
    lightgbm_model = LightGBMModel(estimator=model, artifact_dir=artifact_dir)
    lightgbm_model.prepare(
        inference_conda_env="generalml_p38_cpu_v1",
        training_conda_env="generalml_p38_cpu_v1",
        X_sample=trainx,
        y_sample=trainy,
        use_case_type=UseCaseType.BINARY_CLASSIFICATION,
    )


Instantiate a ``ads.model.framework.lightgbm_model.LightGBMModel()`` object with a LightGBM model. Each instance accepts the following parameters:

* ``artifact_dir: str``: Artifact directory to store the files needed for deployment.
* ``auth: (Dict, optional)``: Defaults to ``None``. The default authentication is set using the ``ads.set_auth`` API. To override the default, use ``ads.common.auth.api_keys()`` or ``ads.common.auth.resource_principal()`` and create the appropriate authentication signer and the ``**kwargs`` required to instantiate the ``IdentityClient`` object.
* ``estimator: (Callable)``: Trained LightGBM model using the Training API or the Scikit-Learn Wrapper interface.
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
    >>> model_id = lightgbm_model.save()

    Start loading model.joblib from model directory /tmp/tmphl0uhtbb ...
    Model is successfully loaded.
    ['runtime.yaml', 'model.joblib', 'score.py', 'input_schema.json']

    'ocid1.datasciencemodel.oc1.xxx.xxxxx'

Deploy and Generate Endpoint
============================

.. code-block:: python3

    >>> # Deploy and create an endpoint for the LightGBM model
    >>> lightgbm_model.deploy(
            display_name="LightGBM Model For Classification",
            deployment_log_group_id="ocid1.loggroup.oc1.xxx.xxxxx",
            deployment_access_log_id="ocid1.log.oc1.xxx.xxxxx",
            deployment_predict_log_id="ocid1.log.oc1.xxx.xxxxx",
        )


    >>> print(f"Endpoint: {lightgbm_model.model_deployment.url}")

    https://modeldeployment.{region}.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx


Run Prediction against Endpoint
===============================

.. code-block:: python3

    # Generate prediction by invoking the deployed endpoint
    lightgbm_model.predict(testx)['prediction']

.. parsed-literal:: 

    [1,0,...,1]

Run Prediction with oci raw-request command
===========================================

Model deployment endpoints can be invoked with the OCI-CLI. The below examples invoke a model deployment with the CLI with different types of payload: ``json``,  ``numpy.ndarray``, ``pandas.core.frame.DataFrame`` or ``dict``.

`json` payload example
----------------------

.. code-block:: python3

    >>> # Prepare data sample for prediction
    >>> data = testx[[11]]
    >>> data
    array([[ 0.59990614,  0.95516275, -1.22366985, -0.89270887, -1.14868768,
        -0.3506047 ,  0.28529227,  2.00085413,  0.31212668,  0.39411511,
         0.87301082, -0.01743923, -1.15089633,  1.03461823,  1.50228029]])

Use printed output of the data and endpoint to invoke prediction with raw-request command in terminal:

.. code-block:: bash

    export uri=https://modeldeployment.{region}.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx/predict
    export data='{"data": [[ 0.59990614,  0.95516275, ... , 1.50228029]]}'
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

    >>> data = testx[[10]]
    >>> np_bytes = BytesIO()
    >>> np.save(np_bytes, data, allow_pickle=True)
    >>> data = base64.b64encode(np_bytes.getvalue()).decode("utf-8")
    >>> print(data)
    k05VTVBZAQB2AHsnZGVzY......D1cJ+D8=

Use printed output of `base64` data and endpoint to invoke prediction with raw-request command in terminal:

.. code-block:: bash

    export uri=https://modeldeployment.{region}.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx/predict
    export data='{"data":"k05VTVBZAQB2AHsnZGVzY......4UdN0L8=", "data_type": "numpy.ndarray"}'
    oci raw-request \
        --http-method POST \
        --target-uri $uri \
        --request-body "$data"

`pandas.core.frame.DataFrame` payload example
---------------------------------------------

.. code-block:: python3

    >>> # Prepare data sample for prediction
    >>> import pandas as pd

    >>> df = pd.DataFrame(testx[[10]])
    >>> print(json.dumps(df.to_json())
    "{\"0\":{\"0\":0.4133005141},\"1\":{\"0\":0.676589266},...,\"14\":{\"0\":-0.2547168443}}"

Use printed output of ``DataFrame`` data and endpoint to invoke prediction with raw-request command in terminal:

.. code-block:: bash

    export uri=https://modeldeployment.{region}.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx/predict
    export data='{"data":"{\"0\":{\"0\":0.4133005141},...,\"14\":{\"0\":-0.2547168443}}","data_type":"pandas.core.frame.DataFrame"}'
    oci raw-request \
        --http-method POST \
        --target-uri $uri \
        --request-body "$data"

`dict` payload example
----------------------

    >>> # Prepare data sample for prediction
    >>> import pandas as pd

    >>> df = pd.DataFrame(testx[[11]])
    >>> print(json.dumps(df.to_dict()))
    {"0": {"0": 0.5999061426438217}, "1": {"0": 0.9551627492226553}, ...,"14": {"0": 1.5022802918908846}}

Use printed output of ``dict`` data and endpoint to invoke prediction with raw-request command in terminal:

.. code-block:: bash

    export uri=https://modeldeployment.{region}.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx/predict
    export data='{"data": {"0": {"0": 0.5999061426438217}, ...,"14": {"0": 1.5022802918908846}}}'
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
          1
        ]
      },
      "headers": {
        "Connection": "keep-alive",
        "Content-Length": "18",
        "Content-Type": "application/json",
        "Date": "Wed, 07 Dec 2022 18:31:05 GMT",
        "X-Content-Type-Options": "nosniff",
        "opc-request-id": "B67EED723ADD43D7BBA1AD5AFCCBD0C6/03F218FF833BC8A2D6A5BDE4AB8B7C12/6ACBC4E5C5127AC80DA590568D628B60",
        "server": "uvicorn"
      },
      "status": "200 OK"
    }

Example
=======

.. code-block:: python3

    from ads.model.framework.lightgbm_model import LightGBMModel
    from ads.common.model_metadata import UseCaseType

    import lightgbm as lgb

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    import tempfile

    seed = 42

    # Create a classification dataset
    X, y = make_classification(n_samples=10000, n_features=15, n_classes=2, flip_y=0.05)

    trainx, testx, trainy, testy = train_test_split(X, y, test_size=30, random_state=seed)

    # Train LGBM model
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.01, random_state=42)
    model.fit(
        trainx,
        trainy,
    )

    # Prepare Model Artifact for LightGBM model
    artifact_dir = tempfile.mkdtemp()
    lightgbm_model = LightGBMModel(estimator=model, artifact_dir=artifact_dir)
    lightgbm_model.prepare(
        inference_conda_env="generalml_p38_cpu_v1",
        training_conda_env="generalml_p38_cpu_v1",
        X_sample=trainx,
        y_sample=trainy,
        force_overwrite=True,
        use_case_type=UseCaseType.BINARY_CLASSIFICATION,
    )

    # Check if the artifacts are generated correctly.
    # The verify method invokes the ``predict`` function defined inside ``score.py`` in the artifact_dir
    lightgbm_model.verify(testx[:10])["prediction"]

    # Register the model
    model_id = lightgbm_model.save(display_name="LightGBM Model")

    # Deploy and create an endpoint for the LightGBM model
    lightgbm_model.deploy(
        display_name="LightGBM Model For Classification",
        deployment_log_group_id="ocid1.loggroup.oc1.xxx.xxxxx",
        deployment_access_log_id="ocid1.log.oc1.xxx.xxxxx",
        deployment_predict_log_id="ocid1.log.oc1.xxx.xxxxx",
    )


    print(f"Endpoint: {lightgbm_model.model_deployment.url}")

    # Generate prediction by invoking the deployed endpoint
    lightgbm_model.predict(testx)["prediction"]

    # To delete the deployed endpoint uncomment the line below
    # lightgbm_model.delete_deployment(wait_for_completion=True)



