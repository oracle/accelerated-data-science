Introduction to LightGBMModel
=============================

Overview:
---------

The ``LightGBMModel`` module in ADS provides different ways of serializing a
trained LightGBM`` model. This example demonstrates how to utilize the 
``LightGBMModel`` module to prepare model artifacts, save models into the
model catalog, and then deploy any unsupported model framework.

A `model artifact <https://docs.oracle.com/en-us/iaas/data-science/using/models-prepare-artifact.htm>`_ includes the model, metadata about the model, input and
output schema, and a script to load the model and make predictions.
These model artifacts can be shared among data scientists, tracked for
provenance, reproduced, and deployed.


Initiate
--------
``LightGBMModel()`` initiates a LightGBM model instance and accepts the following variables:

- ``estimator: (Callable)``. Trained LightGBM model using the learning API or the sklearn API.
- ``artifact_dir: str``. Artifact directory to store the files needed for deployment.
- ``properties: (ModelProperties, optional)``. Defaults to None. The ``ModelProperties`` object required to save and deploy model.
- ``auth :(Dict, optional). Defaults to None``. The default authentication is set using the ``ads.set_auth`` API. If you want to override the default, use ``ads.common.auth.api_keys`` or ``ads.common.auth.resource_principal`` to create appropriate authentication signer and kwargs required to instantiate ``IdentityClient`` object.

The ``score.py``file  is automatically generated and you don't need to modify it. You can change its contents. For example, add your preferred steps to ``pre_inference`` and ``post_inference``.
The ``properties`` instance of ``ModelProperties`` and it has the following predefined fields:

- ``inference_conda_env: str``
- ``inference_python_version: str``
- ``training_conda_env: str``
- ``training_python_version: str``
- ``training_resource_id: str``
- ``training_script_path: str``
- ``training_id: str``
- ``compartment_id: str``
- ``project_id: str``
- ``deployment_instance_shape: str``
- ``deployment_instance_count: int``
- ``deployment_bandwidth_mbps: int``
- ``deployment_log_group_id: str``
- ``deployment_access_log_id: str``
- ``deployment_predict_log_id: str``

 By default, ``properties`` is populated from environment variables if it's not specified. For example, in the notebook session the environment variables for such as project id, compartment id are preset and stored in ``PROJECT_OCID`` and ``NB_SESSION_COMPARTMENT_OCID`` by default. And ``properties`` populates those variables from the environment variables ,and use the values in functions such as ``.save()`` and ``.deployment()`` by default. However, you can always explicitly pass the variables into those functions to overwrite the values. For the fields that ``properties`` has, it records the values that you pass into the functions. For example, when you pass ``inference_conda_env`` into ``.prepare()``, then ``properties`` records this value and you can export it using ``.to_yaml()``. You can reload it using ``.from_yaml()`` from any machine, which allows you to reuse the properties in different places.


Summary_status
--------------
You can call the ``.summary_status()`` function any time after the ``LightGBMModel`` instance is created. 

An example of summary status table is similar to the following after you initiate the model instance. The step column shows all the functions. It shows that the initiate step is completed, and the ``Details`` column explains what initiate step did,and that ``prepare()`` is now available. The next step is to call ``prepare()``. 

.. figure:: figure/summary_status.png
   :align: center


Prepare
-------
``.prepare()`` takes the following parameters:

- ``inference_conda_env: (str, optional)``. Defaults to None. Can be either slug or object storage path of the conda pack. You can only pass in slugs if the conda pack is a service pack.
- ``inference_python_version: (str, optional)``. Defaults to None. Python version which will be used in deployment.
- ``training_conda_env: (str, optional)``. Defaults to None. Can be either slug or object storage path of the conda pack. You can only pass in slugs if the conda pack is a service pack.
- ``training_python_version: (str, optional)``. Defaults to None. Python version used during training.
- ``model_file_name: (str)``. Name of the serialized model.
- ``as_onnx: (bool, optional)``. Defaults to False. Whether to serialize as onnx model.
- ``initial_types: (List[Tuple], optional)``. Defaults to None. Each element is a tuple of a variable name and a type. Check this `link <http://onnx.ai/sklearn-onnx/api_summary.html#id2>`__ for more explanation and examples for ``initial_types``.
- ``force_overwrite: (bool, optional)``. Defaults to False. Whether to overwrite existing files.
- ``namespace: (str, optional)``. Namespace of region. This is used for identifying which region the service pack is from when you pass a slug to inference_conda_env and training_conda_env.
- ``use_case_type: str``. The use case type of the model. Use it through ``UserCaseType`` class or string provided in ``UseCaseType``. For example, ``use_case_type=UseCaseType.BINARY_CLASSIFICATION`` or ``use_case_type="binary_classification"``. Check with ``UseCaseType`` class to see all supported types.
- ``X_sample: Union[Dict, str, List, np.ndarray, pd.core.series.Series, pd.core.frame.DataFrame,]``. Defaults to None.
- ``y_sample: Union[Dict, str, List, pd.Series, np.ndarray]``. Defaults to None. A sample of output data that will be used to generate output schema.
- ``training_script_path: str``. Defaults to None. Training script path.
- ``training_id: (str, optional)``. Defaults to value from environment variables. The training OCID for model. Can be notebook session or job OCID.
- ``ignore_pending_changes: bool``. Defaults to False. Whether to ignore the pending changes in the git.
- ``max_col_num: (int, optional)``. Defaults to ``utils.DATA_SCHEMA_MAX_COL_NUM``. The maximum column size of the data that allows to auto generate schema.

``kwargs``:

- ``impute_values: (dict, optional)``. The dictionary where the key is the column index(or names is accepted for pandas dataframe) and the value is the impute value for the corresponding column.

**Notes:**

1. We provide two ways of serializing the models: local method which is supported by lightgbm and onnx method. By default, local method is used and also it is recommended way of serialize the model.
2. ``prepare()`` also takes any variables that skl2onnx.convert_sklearn takes when the estimstor is using sklearn API.   If the estimator is using learning API, then kwargs takes any variable that onnxmltools.convert_lightgbm takes.

It will automatically generate the following files.

- ``runtime.yaml``
- ``score.py``
- ``model.txt`` for learning api, ``model.joblib`` for sklearn api by default. If ``as_onnx=True`` the default file name should be ``model.onnx``. However, you can set model file name yourself.
- ``input_schema.json`` when ``X_sample`` is passed in and the schema is more than 32kb.
- ``output_schema.json`` when ``y_sample`` is passed in and the schema is more than 32kb.
- ``hyperparameters.json`` if extracted hyperparameters is more than 32kb.


Verify
------
``.verify()`` function takes one parameter:

- ``data (Union[Dict, str, List, np.ndarray, pd.core.series.Series, pd.core.frame.DataFrame])``. Data used to test if deployment works in local environment.

It is used to test if deployment would work in the local environment. Before saving and deploying the model, it is recommended to call this function to check if ``load_model`` and ``predict`` function in ``score.py`` works. It takes and returns the same data as model deployment predict takes and returns.

In ``LightGBMModel``, data serialization is supported for dictionary, string, list, np.ndarray, pd.core.series.Series, pd.core.frame.DataFrame, which means that user can pass in Pandas DataFrame or Numpy array even though they are not JSON serializable. The reason is that we internally serialize and deserialize the data automatically. 

Save
----
``.save()`` function takes the following parameters:

- ``display_name: (str, optional)``. Defaults to None. The name of the model.
- ``description: (str, optional)``. Defaults to None. The description of the model.
- ``freeform_tags : Dict(str, str)``. Defaults to None. Freeform tags for the model.
- ``defined_tags : (Dict(str, dict(str, object)), optional)``. Defaults to None. Defined tags for the model.
- ``ignore_introspection: (bool, optional)``. Defaults to None. Determines whether to ignore the result of model introspection or not. If set to True, the save will ignore all model introspection errors.

``kwargs``:
- ``project_id: (str, optional)``. Project OCID. If not specified, the value will be taken either from the environment variables or model properties.
- ``compartment_id : (str, optional)``. Compartment OCID. If not specified, the value will be taken either from the environment variables or model properties.
- ``timeout: (int, optional)``. Defaults to 10 seconds. The connection timeout in seconds for the client.

It will first reload the ``score.py`` and ``runtime.yaml`` files from the disk so that any changes made to those files can be picked up. And then, it conducts an instropection test by default. However, you can set ``ignore_introspection=False`` to avoid it. Introspection test checks if ``.deployment()`` later could have some issues and suggests neccessary actions needed to get them fixed. Lastly, it will upload the artifacts to the model catalog and return a ``model_id`` for the saved model.
You can also call ``.instrospect()`` to conduct the test any time after ``.prepare()`` is called.

Deploy
------
``.deploy()`` takes the following parameters:

- ``wait_for_completion : (bool, optional)``. Defaults to True. Flag set for whether to wait for deployment to complete before proceeding.
- ``display_name: (str, optional)``. Defaults to None. The name of the model.
- ``description: (str, optional)``. Defaults to None. The description of the model.
- ``deployment_instance_shape: (str, optional)``. Default to ``VM.Standard2.1``. The shape of the instance used for deployment.
- ``deployment_instance_count: (int, optional)``. Defaults to 1. The number of instance used for deployment.
- ``deployment_bandwidth_mbps: (int, optional)``. Defaults to 10. The bandwidth limit on the load balancer in Mbps.
- ``deployment_log_group_id: (str, optional)``. Defaults to None. The oci logging group id. The access log and predict log share the same log group.
- ``deployment_access_log_id: (str, optional)``. Defaults to None. The access log OCID for the access logs. Link: `<https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm>`__
- ``deployment_predict_log_id: (str, optional)``. Defaults to None. The predict log OCID for the predict logs. Link: `<https://docs.oracle.com/en-us/iaas/data-science/using/model_dep_using_logging.htm>`__

``kwargs``:

- ``project_id: (str, optional)``. Project OCID. If not specified, the value will be taken from the environment variables.
- ``compartment_id : (str, optional)``. Compartment OCID. If not specified, the value will be taken from the environment variables.
- ``max_wait_time : (int, optional)``. Defaults to 1200 seconds. Maximum amount of time to wait in seconds. Negative implies infinite wait time.
- ``poll_interval : (int, optional)``. Defaults to 60 seconds. Poll interval in seconds.

It will deploy the model. In order to make deployment more smooth, we suggest using exactly the same conda environments for both local development and deployment. Discrepancy between the two could cause problems.

You can pass in ``deployment_log_group_id``, ``deployment_access_log_id`` and ``deployment_predict_log_id`` to enable the logging. Please refer to this :ref:`logging example <logging_example>` for an example on logging.  To create a log group, you can reference :ref:`Logging <logging>` session. 

Predict
-------
``.predict()`` will take one parameter ``Data`` expected by the predict API in ``score.py``.
- ``data (Union[Dict, str, List, np.ndarray, pd.core.series.Series, pd.core.frame.DataFrame])``. 

``.predict()``takes the same data  that ``.verify()`` takes, user has to make sure the data passed and returned by ``predict`` in the ``score.py`` is json serializable. It passes the data to the model deployment endpoint and calls the ``predict`` function in the ``score.py``.


Delete_deployment
-----------------
``.delete_deployment()`` takes one parameter:

- ``wait_for_completion: (bool, optional)``. Defaults to False. Whether to wait till completion.

Once you dont need the deployment any more. You can call ``delete_deployment`` to delete the current deployment that is attached to this model. Note that each time you call deploy, it will create a new deployment and only the new deployment is attached to this model. 

from_model_artifact
-------------------

``.from_model_artifact()`` allows to load a model from a folder, zip or tar achive files, where the folder/zip/tar files should contain the files such as runtime.yaml, score.py, the serialized model file needed for deployments. It takes the following parameters:

- ``uri: str``: The folder path, ZIP file path, or TAR file path. It could contain a seriliazed model(required) as well as any files needed for deployment including: serialized model, runtime.yaml, score.py and etc. The content of the folder will be copied to the ``artifact_dir`` folder.
- ``model_file_name: str``: The serialized model file name.
- ``artifact_dir: str``: The artifact directory to store the files needed for deployment.
- ``auth: (Dict, optional)``: Defaults to None. The default authetication is set using ``ads.set_auth`` API. If you need to override the default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate authentication signer and kwargs required to instantiate IdentityClient object.
- ``force_overwrite: (bool, optional)``: Defaults to False. Whether to overwrite existing files or not.
- ``properties: (ModelProperties, optional)``: Defaults to None. ModelProperties object required to save and deploy model.


After this is called, you can call ``.verify()``, ``.save()`` and etc.


from_model_catalog
------------------

``from_model_catalog`` allows to load a remote model from model catalog using a model id , which should contain the files such as runtime.yaml, score.py, the serialized model file needed for deployments. It takes the following parameters:

- ``model_id: str``. The model OCID.
- ``model_file_name: (str)``. The name of the serialized model.
- ``artifact_dir: str``. The artifact directory to store the files needed for deployment. Will be created if not exists.
- ``auth: (Dict, optional)``. Defaults to None. The default authetication is set using ``ads.set_auth`` API. If you need to override the default, use the ``ads.common.auth.api_keys`` or ``ads.common.auth.resource_principal`` to create appropriate authentication signer and kwargs required to instantiate IdentityClient object.
- ``force_overwrite: (bool, optional)``. Defaults to False. Whether to overwrite existing files or not.
- ``properties: (ModelProperties, optional)``. Defaults to None. ModelProperties object required to save and deploy model.

``kwargs``:

- ``compartment_id : (str, optional)``. Compartment OCID. If not specified, the value will be taken from the environment variables.
- ``timeout : (int, optional)``. Defaults to 10 seconds. The connection timeout in seconds for the client.


Examples
--------
Create a Lightgbm Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code:: python3
    
    import ads
    import lightgbm as lgb
    import logging
    import numpy as np
    import pandas as pd
    import os
    import tempfile
    import warnings

    from ads.catalog.model import ModelCatalog
    from ads.model.framework.lightgbm_model import LightGBMModel
    from shutil import rmtree
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    # Load data
    df_path = os.path.join("/", "opt", "notebooks", "ads-examples", "oracle_data", "orcl_attrition.csv")
    df = pd.read_csv(df_path)
    y = df["Attrition"]
    X = df.drop(columns=["Attrition"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Label encode the y values
    le = LabelEncoder()
    y_train_transformed = le.fit_transform(y_train)
    y_test_transformed = le.transform(y_test)

    # Extract numerical columns and categorical columns
    categorical_cols = []
    numerical_cols = []
    for i, col in X.iteritems():
        if col.dtypes == "object":
            categorical_cols.append(col.name)
        else:
            numerical_cols.append(col.name)

    categorical_transformer = Pipeline(
        steps=[
            ('encoder', OrdinalEncoder())
        ]
    )

    # Build a pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    preprocessor_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    preprocessor_pipeline.fit(X_train)
    X_train_transformed = preprocessor_pipeline.transform(X_train)
    X_test_transformed = preprocessor_pipeline.transform(X_test)

    # LightGBM Training API
    dtrain = lgb.Dataset(X_train_transformed, label=y_train_transformed)
    dtest = lgb.Dataset(X_test_transformed, label=y_test_transformed)

    estimator_train = lgb.train(
        params={'num_leaves': 31, 'objective': 'binary', 'metric': 'auc'}, 
        train_set=dtrain, num_boost_round=10)
    
    # LightGBM Scikit-Learn API
    estimator = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.01, random_state=42
    )
    estimator.fit(
        X_train_transformed,
        y_train_transformed,
    )

Lightgbm Framework Serialization - Training API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python3

    learning_api_model = LightGBMModel(estimator=estimator_train, artifact_dir=tempfile.mkdtemp())
    learning_api_model.prepare(
        inference_conda_env="generalml_p37_cpu_v1",
        force_overwrite=True,
    )
    learning_api_model.verify(X_test_transformed)['prediction'][:10]
    learning_api_model.save()
    learning_api_model.deploy()
    learning_api_model.delete_deployment()


Lightgbm Framework Serialization - Sklearn API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python3

    model = LightGBMModel(estimator=estimator, artifact_dir=artifact_dir)
    model.prepare(
        inference_conda_env="generalml_p37_cpu_v1",
        training_conda_env="generalml_p37_cpu_v1",
        X_sample=X_train_transformed[:10],
        as_onnx=True,
        force_overwrite=True,
    )
    model.verify(X_test_transformed[:10])['prediction']
    model.save()
    model.deploy()
    model.predict(X_test_transformed[:10])['prediction']
    model.delete_deployment()


Onnx Serialization - Training API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python3

    learning_api_model_onnx = LightGBMModel(estimator=estimator_train, artifact_dir=tempfile.mkdtemp())
    initial_types = [('input', FloatTensorType(shape=[None, 8]))]
    learning_api_model_onnx.prepare(
        inference_conda_env="oci://bucket@namespace/path/to/custom_conda_pack",
        inference_python_version="3.7",
        as_onnx=True,
        force_overwrite=True,
        initial_types=initial_types,
    )
    learning_api_model_onnx.verify(X_test_transformed[:10].astype("float32"))['prediction']
    learning_api_model_onnx.save()
    learning_api_model_onnx.deploy()
    learning_api_model_onnx.predict(X_test_transformed[:10].astype("float32"))['prediction']
    learning_api_model_onnx.delete_deployment()


Onnx Serialization - Sklearn API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python3

    sklearn_api_model_onnx = LightGBMModel(estimator=estimator, artifact_dir=tempfile.mkdtemp())
    initial_types = [('input', FloatTensorType(shape=[None, 8]))]
    sklearn_api_model_onnx.prepare(
        inference_conda_env="oci://license_checker@ociodscdev/published_conda_environments/cpu/ads_env/1.0/ads_envv1_0",
        inference_python_version="3.7",
        as_onnx=True,
        force_overwrite=True,
        initial_types=initial_types,
    )
    sklearn_api_model_onnx.verify(pd.DataFrame(X_test_transformed[:10]))['prediction']
    sklearn_api_model_onnx.save()
    sklearn_api_model_onnx.deploy(wait_for_completion=False)
    sklearn_api_model_onnx.predict(X_test_transformed[:10])['prediction']
    sklearn_api_model_onnx.delete_deployment()

Loading Model From a Zip Archive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python3

   model = LightGBMModel.from_model_catalog(model_id="ocid1.datasciencemodel.oc1.iad.amaaaa....",
                                         model_file_name="your_model_file_name",
                                         artifact_dir=tempfile.mkdtemp())
   model.verify(your_data)

Loading Model From Model Catalog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python3

   model = LightGBMModel.from_model_artifact("/folder_to_your/artifact.zip",
                                         model_file_name="your_model_file_name",
                                         artifact_dir=tempfile.mkdtemp())
   model.verify(your_data)
