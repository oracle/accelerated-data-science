.. XGBoostModel:

XGBoostModel
************

Overview
========

The ``XGBoostModel`` class in ADS is designed to allow you to rapidly get a XGBoost model into production. The ``.prepare()`` method creates the model artifacts that are needed to deploy a functioning model without you having to configure it or write code. However, you can customize the required ``score.py`` file.

.. include:: _template/overview.rst

The following steps take your trained ``XGBoost`` model and deploy it into production with a few lines of code.

The ``XGBoostModel`` module in ADS supports serialization for models generated from both the  `Learning API <https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training>`_ using ``xgboost.train()`` and the `Scikit-Learn API <https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn>`_ using ``xgboost.XGBClassifier()``. Both of these interfaces are defined by `XGBoost <https://xgboost.readthedocs.io/en/stable/index.html>`_.

**Create Learning API and Scikit-Learn Wrapper XGBoost Models**

In the following several code snippets you will prepare the data and train XGBoost models. In the first snippet, the data will be prepared. This will involved loading a dataset, splitting it into dependent and independent variables and into test and training sets. The data will be encoded and a preprocessing pipeline will be defined. In the second snippet, the XGBoost Learning API will be used to train the model. In the third and final code snippet, the Scikit-Learn Wrapper interface is used to create another XGBoost model.

.. code-block:: python3

    import pandas as pd
    import os
    import tempfile
    import xgboost as xgb

    from ads.model.framework.xgboost_model import XGBoostModel
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    df_path = os.path.join("/", "opt", "notebooks", "ads-examples", "oracle_data", "orcl_attrition.csv")
    df = pd.read_csv(df_path)
    y = df["Attrition"]
    X = df.drop(columns=["Attrition", "name"])

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
        steps=[('encoder', OrdinalEncoder())]
    )

    # Build a pipeline
    preprocessor = ColumnTransformer(
        transformers=[('cat', categorical_transformer, categorical_cols)]
    )

    preprocessor_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    preprocessor_pipeline.fit(X_train)

    X_train_transformed = preprocessor_pipeline.transform(X_train)
    X_test_transformed = preprocessor_pipeline.transform(X_test)


Create an XGBoost model using the Learning API.

.. code-block:: python3

    dtrain = xgb.DMatrix(X_train_transformed, y_train_transformed)
    dtest = xgb.DMatrix(X_test_transformed, y_test_transformed)

    model_learn = xgb.train(
        params = {"learning_rate": 0.01, "max_depth": 3},
        dtrain = dtrain,
    )


Create an XGBoost model using the Scikit-Learn Wrapper interface.

.. code-block:: python3

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42,
        use_label_encoder=False
    )
    model.fit(
        X_train_transformed,
        y_train_transformed,
    )

Initialize
==========

Instantiate a ``XGBoostModel()`` object with an XGBoost model. Each instance accepts the following parameters:

* ``artifact_dir: str``: Artifact directory to store the files needed for deployment.
* ``auth: (Dict, optional)``: Defaults to ``None``. The default authentication is set using the ``ads.set_auth`` API. To override the default, use ``ads.common.auth.api_keys()`` or ``ads.common.auth.resource_principal()`` and create the appropriate authentication signer and the ``**kwargs`` required to instantiate the ``IdentityClient`` object.
* ``estimator: (Callable)``: Trained XGBoost model either using the Learning API or the Scikit-Learn Wrapper interface.
* ``properties: (ModelProperties, optional)``: Defaults to ``None``. The ``ModelProperties`` object required to save and deploy a  model.

.. include:: _template/initialize.rst

Summary Status
==============

.. include:: _template/summary_status.rst

.. figure:: figures/summary_status.png
   :align: center

Model Deployment
================

Prepare
-------

The prepare step is performed by the ``.prepare()`` method. It creates several customized files used to run the model after it is deployed. These files include:

* ``input_schema.json``: A JSON file that defines the nature of the features of the ``X_sample`` data. It includes metadata such as the data type, name, constraints, summary statistics, feature type, and more.
* ``model.json``: This is the default filename of the serialized model. It can be changed with the ``model_file_name`` attribute. By default, the model is stored in a JSON file. You can use the ``as_onnx`` parameter to save in ONNX format, and the model name defaults to ``model.onnx``.
* ``output_schema.json``: A JSON file that defines the nature of the dependent variable in the ``y_sample`` data. It includes metadata such as the data type, name, constraints, summary statistics, feature type, and more.
* ``runtime.yaml``: This file contains information needed to set up the runtime environment on the deployment server. It has information about what conda environment was used to train the model and what environment to use to deploy the model. The file also specifies what version of Python should be used.
* ``score.py``: This script contains the ``load_model()`` and ``predict()`` functions. The ``load_model()`` function understands the format the model file was saved in and loads it into memory. The ``.predict()`` method is used to make inferences in a deployed model. There are also hooks that allow you to perform operations before and after inference. You can modify this script to fit your specific needs.

To create the model artifacts you use the ``.prepare()`` method. There are a number of parameters that allow you to store model provenance information.

To serialize the model to ONNX format, set the ``as_onnx`` parameter to ``True``. You can provide the ``initial_types`` parameter, which is a Python list describing the variable names and types. Alternatively, the service tries to infer this information from the data in the ``X_sample`` parameter. ``X_sample`` supports List, Numpy array or Pandas dataframe. ``DMatrix`` class is not supported because this format can't convert into a JSON serializable format, see the `ONNX docs <http://onnx.ai/sklearn-onnx/api_summary.html>`_.

.. include:: _template/prepare.rst

When using the Scikit-Learn Wrapper interface, the ``.prepare()`` method accepts any parameter that ``skl2onnx.convert_sklearn`` accepts. When using the Learning API, the ``.prepare()`` method accepts any parameter that ``onnxmltools.convert_xgboost`` accepts.

Verify
------

.. include:: _template/verify.rst

* ``data: Any``: Data used to test if deployment works in a local environment.

Save
----

.. include:: _template/save.rst

Deploy
------

.. include:: _template/deploy.rst

Predict
-------

.. include:: _template/predict.rst

* ``data: Any``: JSON serializable data used for making inferences.

The ``.predict()`` and ``.verify()`` methods take the same data formats. You must ensure that the data passed into and returned by the ``predict()`` function in the ``score.py`` file is JSON serializable.

Load
====

You can restore serialization models from model artifacts, from model deployments or from models in the model catalog. This section provides details on how to restore serialization models.

.. include:: _template/loading_model_artifact.rst

.. code-block:: python3

    from ads.model.framework.xgboost_model import XGBoostModel

    model = XGBoostModel.from_model_artifact(
                    uri="/folder_to_your/artifact.zip",
                    model_file_name="model.joblib",
                    artifact_dir="/folder_store_artifact"
                )

.. include:: _template/loading_model_catalog.rst


.. code-block:: python3

    from ads.model.framework.xgboost_model import XGBoostModel

    model = XGBoostModel.from_model_catalog(model_id="<model_id>",
                                            model_file_name="model.json",
                                            artifact_dir=tempfile.mkdtemp())

.. include:: _template/loading_model_deployment.rst

.. code-block:: python3

    from ads.model.generic_model import XGBoostModel

    model = XGBoostModel.from_model_deployment(
        model_deployment_id="<model_deployment_id>",
        model_file_name="model.pkl",
        artifact_dir=tempfile.mkdtemp())

Delete a Deployment
===================

.. include:: _template/delete_deployment.rst

Example
=======

.. code-block:: python3

    import pandas as pd
    import os
    import tempfile
    import xgboost as xgb

    from ads.catalog.model import ModelCatalog
    from ads.common.model_metadata import UseCaseType
    from ads.model.framework.xgboost_model import XGBoostModel
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    df_path = os.path.join("/", "opt", "notebooks", "ads-examples", "oracle_data", "orcl_attrition.csv")
    df = pd.read_csv(df_path)
    y = df["Attrition"]
    X = df.drop(columns=["Attrition", "name"])

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
        steps=[('encoder', OrdinalEncoder())]
    )

    # Build a pipeline
    preprocessor = ColumnTransformer(
        transformers=[('cat', categorical_transformer, categorical_cols)]
    )

    preprocessor_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    preprocessor_pipeline.fit(X_train)

    X_train_transformed = preprocessor_pipeline.transform(X_train)
    X_test_transformed = preprocessor_pipeline.transform(X_test)

    # XGBoost Scikit-Learn API
    model = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.01, random_state=42,
        use_label_encoder=False
    )
    model.fit(X_train_transformed, y_train_transformed)

    # Deploy the model, test it and clean up.
    artifact_dir = tempfile.mkdtemp()
    xgboost_model = XGBoostModel(estimator=model, artifact_dir=artifact_dir)
    xgboost_model.prepare(
        inference_conda_env="generalml_p37_cpu_v1",
        training_conda_evn="generalml_p37_cpu_v1",
        use_case_type=UseCaseType.BINARY_CLASSIFICATION,
        X_sample=X_test_transformed,
        y_sample=y_test_transformed,
    )
    xgboost_model.verify(X_test_transformed[:10])['prediction']
    model_id = xgboost_model.save()
    xgboost_model.deploy()
    xgboost_model.predict(X_test_transformed[:10])['prediction']
    xgboost_model.delete_deployment(wait_for_completion=True)
    ModelCatalog(compartment_id=os.environ['NB_SESSION_COMPARTMENT_OCID']).delete_model(model_id)

