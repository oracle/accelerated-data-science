.. SklearnModel:

SklearnModel
************

Overview
========

The ``SklearnModel`` class in ADS is designed to allow you to rapidly get a Scikit-learn model into production. The ``.prepare()`` method creates the model artifacts that are needed to deploy a functioning model without you having to configure it or write code. However, you can customize the required ``score.py`` file.

.. include:: _template/overview.rst

The following steps take your trained ``scikit-learn`` model and deploy it into production with a few lines of code.

**Create a Scikit-learn Model**

.. code-block:: python3

    import pandas as pd
    import os

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
    from sklearn.model_selection import train_test_split

    ds_path = os.path.join("/", "opt", "notebooks", "ads-examples", "oracle_data", "orcl_attrition.csv")
    df = pd.read_csv(ds_path)
    y = df["Attrition"]
    X = df.drop(columns=["Attrition", "name"])

    # Data Preprocessing
    for i, col in X.iteritems():
        col.replace("unknown", "", inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Label encode the y values
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Extract numerical columns and categorical columns
    categorical_cols = []
    numerical_cols = []
    for i, col in X.iteritems():
        if col.dtypes == "object":
            categorical_cols.append(col.name)
        else:
            numerical_cols.append(col.name)

    categorical_transformer = Pipeline(steps=[
        ('encoder', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-999))
    ])
    preprocessor = ColumnTransformer(
        transformers=[('cat', categorical_transformer, categorical_cols)]
    )

    ml_model = RandomForestClassifier(n_estimators=100, random_state=0)
    model = Pipeline(
        steps=[('preprocessor', preprocessor),
               ('model', ml_model)
        ])

    model.fit(X_train, y_train)




Initialize
==========

Instantiate a ``SklearnModel()`` object with an Scikit-learn model. Each instance accepts the following parameters:

* ``artifact_dir: str``: Artifact directory to store the files needed for deployment.
* ``auth: (Dict, optional)``: Defaults to ``None``. The default authentication is set using the ``ads.set_auth`` API. To override the default, use ``ads.common.auth.api_keys()`` or ``ads.common.auth.resource_principal()`` and create the appropriate authentication signer and the ``**kwargs`` required to instantiate the ``IdentityClient`` object.
* ``estimator: (Callable)``: Trained Scikit-learn model or Scikit-learn pipeline.
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
* ``model.joblib``: This is the default filename of the serialized model. It can be changed with the ``model_file_name`` attribute. By default, the model is stored in a joblib file. The parameter ``as_onnx`` can be used to save it in the ONNX format.
* ``output_schema.json``: A JSON file that defines the nature of the dependent variable in the ``y_sample`` data. It includes metadata such as the data type, name, constraints, summary statistics, feature type, and more.
* ``runtime.yaml``: This file contains information that is needed to set up the runtime environment on the deployment server. It has information about which conda environment was used to train the model, and what environment should be used to deploy the model. The file also specifies what version of Python should be used.
* ``score.py``: This script contains the ``load_model()`` and ``predict()`` functions. The ``load_model()`` function understands the format the model file was saved in and loads it into memory. The ``.predict()`` method is used to make inferences in a deployed model. There are also hooks that allow you to perform operations before and after inference. You can modify this script to fit your specific needs.

.. include:: _template/prepare.rst

Verify
------

.. include:: _template/verify.rst

* ``data: Any``: Data used to test if deployment works in local environment.

In ``SklearnModel``, data serialization is supported for JSON serializable objects. Plus, there is support for a dictionary, string, list, ``np.ndarray``, ``pd.core.series.Series``, and ``pd.core.frame.DataFrame``. Not all these objects are JSON serializable, however, support to automatically serializes and deserialized is provided.

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

In ``SklearnModel``, data serialization is supported for JSON serializable objects. Plus, there is support for a dictionary, string, list, ``np.ndarray``, ``pd.core.series.Series``, and ``pd.core.frame.DataFrame``. Not all these objects are JSON serializable, however, support to automatically serializes and deserialized is provided.

Load
====

You can restore serialization models from model artifacts, from model deployments or from models in the model catalog. This section provides details on how to restore serialization models.

.. include:: _template/loading_model_artifact.rst

.. code-block:: python3

    from ads.model.framework.sklearn_model import SklearnModel

    model = SklearnModel.from_model_artifact(
                    uri="/folder_to_your/artifact.zip",
                    model_file_name="model.joblib",
                    artifact_dir="/folder_store_artifact"
                )

.. include:: _template/loading_model_catalog.rst


.. code-block:: python3

    from ads.model.framework.sklearn_model import SklearnModel

    model = SklearnModel.from_model_catalog(model_id="<model_id>",
                                            model_file_name="model.pkl",
                                            artifact_dir=tempfile.mkdtemp())

.. include:: _template/loading_model_deployment.rst

.. code-block:: python3

    from ads.model.generic_model import SklearnModel

    model = SklearnModel.from_model_deployment(
        model_deployment_id="<model_deployment_id>",
        model_file_name="model.pkl",
        artifact_dir=tempfile.mkdtemp())

Delete a Deployment
===================

.. include:: _template/delete_deployment.rst

Examples
========

.. code-block:: python3

    import pandas as pd
    import os
    import tempfile

    from ads.catalog.model import ModelCatalog
    from ads.common.model_metadata import UseCaseType
    from ads.model.framework.sklearn_model import SklearnModel
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
    from sklearn.model_selection import train_test_split

    ds_path = os.path.join("/", "opt", "notebooks", "ads-examples", "oracle_data", "orcl_attrition.csv")
    df = pd.read_csv(ds_path)
    y = df["Attrition"]
    X = df.drop(columns=["Attrition", "name"])

    # Data Preprocessing
    for i, col in X.iteritems():
        col.replace("unknown", "", inplace=True)
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

    categorical_transformer = Pipeline(steps=[
        ('encoder', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-999))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols)
    ])

    ml_model = RandomForestClassifier(n_estimators=100, random_state=0)
    model = Pipeline(
        steps=[('preprocessor', preprocessor),
               ('model', ml_model)
    ])

    model.fit(X_train, y_train_transformed)

    # Deploy the model, test it and clean up.
    artifact_dir = tempfile.mkdtemp()
    sklearn_model = SklearnModel(estimator=model, artifact_dir= artifact_dir)
    sklearn_model.prepare(
        inference_conda_env="generalml_p37_cpu_v1",
        training_conda_env="generalml_p37_cpu_v1",
        use_case_type=UseCaseType.BINARY_CLASSIFICATION,
        as_onnx=False,
        X_sample=X_test,
        y_sample=y_test_transformed,
        force_overwrite=True,
    )
    sklearn_model.verify(X_test.head(2))
    model_id = sklearn_model.save()
    sklearn_model.deploy()
    sklearn_model.predict(X_test.head(2))
    sklearn_model.delete_deployment(wait_for_completion=True)
    ModelCatalog(compartment_id=os.environ['NB_SESSION_COMPARTMENT_OCID']).delete_model(model_id)

