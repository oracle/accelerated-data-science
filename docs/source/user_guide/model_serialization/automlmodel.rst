.. AutoMLModel:

AutoMLModel
***********

Overview
========

The ``AutoMLModel`` class in ADS is designed to rapidly get your AutoML model into production. The ``.prepare()`` method creates the model artifacts needed to deploy the model without you having to configure it or write code. The ``.prepare()`` method serializes the model and generates a ``runtime.yaml`` and a ``score.py`` file that you can later customize.

.. include:: _template/overview.rst

The following steps take your trained ``AutoML`` model and deploy it into production with a few lines of code.


**Creating an Oracle Labs AutoML Model**

Train a model using AutoMLx.

.. code-block:: python3

    import pandas as pd
    import numpy as np
    import tempfile
    from sklearn.metrics import roc_auc_score, confusion_matrix, make_scorer, f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.compose import make_column_selector as selector
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    import ads
    import automl
    from automl import init
    from ads.model import AutoMLModel
    from ads.common.model_metadata import UseCaseType
    from ads.model.framework.automl_model import AutoMLModel

    dataset = fetch_openml(name='adult', as_frame=True)
    df, y = dataset.data, dataset.target

    # Several of the columns are incorrectly labeled as category type in the original dataset
    numeric_columns = ['age', 'capitalgain', 'capitalloss', 'hoursperweek']
    for col in df.columns:
        if col in numeric_columns:
            df[col] = df[col].astype(int)
        

    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        y.map({'>50K': 1, '<=50K': 0}).astype(int),
                                                        train_size=0.7,
                                                        random_state=0)

    init(engine='local')
    est = automl.Pipeline(task='classification')
    est.fit(X_train, y_train)

    ads.set_auth("resource_principal")
    artifact_dir = tempfile.mkdtemp()
    automl_model = AutoMLModel(estimator=model, artifact_dir=artifact_dir)
    automl_model.prepare(inference_conda_env="automlx_p38_cpu_v1",
                         training_conda_env="automlx_p38_cpu_v1",
                         use_case_type=UseCaseType.BINARY_CLASSIFICATION,
                         X_sample=X_test,
                         force_overwrite=True)
    automl_model.verify(X_test.iloc[:2])
    model_id = automl_model.save(display_name='Demo AutoMLModel model')
    deploy = automl_model.deploy(display_name='Demo AutoMLModel deployment')
    automl_model.predict(X_test.iloc[:2])
    automl_model.delete_deployment(wait_for_completion=True)
    ModelCatalog(compartment_id=os.environ['NB_SESSION_COMPARTMENT_OCID']).delete_model(model_id)


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

    import pandas as pd
    import numpy as np
    import tempfile
    from sklearn.metrics import roc_auc_score, confusion_matrix, make_scorer, f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.compose import make_column_selector as selector
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    import ads
    import automl
    from automl import init
    from ads.model import AutoMLModel
    from ads.common.model_metadata import UseCaseType
    from ads.model.framework.automl_model import AutoMLModel

    dataset = fetch_openml(name='adult', as_frame=True)
    df, y = dataset.data, dataset.target

    # Several of the columns are incorrectly labeled as category type in the original dataset
    numeric_columns = ['age', 'capitalgain', 'capitalloss', 'hoursperweek']
    for col in df.columns:
        if col in numeric_columns:
            df[col] = df[col].astype(int)
        

    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        y.map({'>50K': 1, '<=50K': 0}).astype(int),
                                                        train_size=0.7,
                                                        random_state=0)

    init(engine='local')
    est = automl.Pipeline(task='classification')
    est.fit(X_train, y_train)

    ads.set_auth("resource_principal")
    artifact_dir = tempfile.mkdtemp()
    automl_model = AutoMLModel(estimator=model, artifact_dir=artifact_dir)
    automl_model.prepare(inference_conda_env="automlx_p38_cpu_v1",
                         training_conda_env="automlx_p38_cpu_v1",
                         use_case_type=UseCaseType.BINARY_CLASSIFICATION,
                         X_sample=X_test,
                         force_overwrite=True)
    automl_model.verify(X_test.iloc[:2])
    model_id = automl_model.save(display_name='Demo AutoMLModel model')
    deploy = automl_model.deploy(display_name='Demo AutoMLModel deployment')
    automl_model.predict(X_test.iloc[:2])
    automl_model.delete_deployment(wait_for_completion=True)
    ModelCatalog(compartment_id=os.environ['NB_SESSION_COMPARTMENT_OCID']).delete_model(model_id)

