.. SklearnModel:

SklearnModel
************

See `API Documentation <../../../ads.model.framework.html#ads.model.framework.sklearn_model.SklearnModel>`__


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

    # Deploy and create an endpoint for the Random Forest model
    sklearn_model.deploy(
        display_name="Random Forest Model For Classification",
        deployment_log_group_id="ocid1.loggroup.oc1.xxx.xxxxx",
        deployment_access_log_id="ocid1.log.oc1.xxx.xxxxx",
        deployment_predict_log_id="ocid1.log.oc1.xxx.xxxxx",
        # Shape config details mandatory for flexible shapes:
        # deployment_instance_shape="VM.Standard.E4.Flex",
        # deployment_ocpus=<number>,
        # deployment_memory_in_gbs=<number>,
    )
    print(f"Endpoint: {sklearn_model.model_deployment.url}")
    # Output: "Endpoint: https://modeldeployment.{region}.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx"

Run Prediction against Endpoint
===============================

.. code-block:: python3

    >>> # Generate prediction by invoking the deployed endpoint
    >>> sklearn_model.predict(testx)['prediction']
    [1,0,...,1]

Run Prediction with oci raw-request command
===========================================

Model deployment endpoints can be invoked with the OCI-CLI. The below examples invoke a model deployment with the CLI with different types of payload: ``json``,  ``numpy.ndarray``, ``pandas.core.frame.DataFrame`` or ``dict``.

`json` payload example
----------------------

.. code-block:: python3

    >>> # Prepare data sample for prediction
    >>> data = testx[[10]]
    >>> data
    array([[ 0.41330051,  0.67658927, -0.39189561,  0.21879805, -0.79208514,
         0.0906022 , -1.60595137,  1.65853693, -1.61337437,  0.82302124,
        -0.87032051,  0.70721209, -1.81956653, -0.26537296, -0.25471684]])

Use output of the data and endpoint to invoke prediction with raw-request command in terminal:

.. code-block:: bash

    export uri=https://modeldeployment.{region}.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx/predict
    export data='{"data": [[ 0.41330051,  0.67658927, ... , -0.25471684]]}'
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
    k05VTVBZAQB2AHsnZGVzY......4UdN0L8=

Use printed output of ``base64`` data and endpoint to invoke prediction with raw-request command in terminal:

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

    >>> df = pd.DataFrame(testx[[10]])
    >>> print(json.dumps(df.to_dict()))
    {"0": {"0": 0.413300514080485}, "1": {"0": 0.6765892660311731}, ...,"14": {"0": -0.2547168443271222}}

Use printed output of `dict` data and endpoint to invoke prediction with raw-request command in terminal:

.. code-block:: bash

    export uri=https://modeldeployment.{region}.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.xxx.xxxxx/predict
    export data='{"data": {"0": {"0": 0.413300514080485}, ...,"14": {"0": -0.2547168443271222}}}'
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
          0
        ]
      },
      "headers": {
        "Connection": "keep-alive",
        "Content-Length": "18",
        "Content-Type": "application/json",
        "Date": "Wed, 07 Dec 2022 18:31:39 GMT",
        "X-Content-Type-Options": "nosniff",
        "opc-request-id": "E1125E17AE084DFAB6BCCFA045C16966/0BBB65235292EE0817B67BD9141A620A/5956FE428CBAB2878EBA605CEECAD39D",
        "server": "uvicorn"
      },
      "status": "200 OK"
    }

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


