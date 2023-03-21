.. GenericModel:

================
Other Frameworks
================

See `API Documentation <../../../ads.model.html#ads.model.generic_model.GenericModel>`__

Overview
========

The ``ads.model.generic_model.GenericModel`` class in ADS provides an efficient way to serialize almost any model class. This section demonstrates how to use the ``GenericModel`` class to prepare model artifacts, verify models, save models to the model catalog, deploy models, and perform predictions on model deployment endpoints.

The ``GenericModel`` class works with any unsupported model framework that has a ``.predict()`` method. For the most common model classes such as scikit-learn, XGBoost, LightGBM, TensorFlow, and PyTorch, we recommend that you use the ADS provided, framework-specific serializations models. For example, for a scikit-learn model, use SKLearnmodel. For other models, use the ``GenericModel`` class.

.. include:: ../_template/overview.rst

These simple steps take your trained model and will deploy it into production with just a few lines of code.

Prepare Model Artifact
======================

Instantiate a ``GenericModel()`` object by giving it any model object. It accepts the following parameters:

* ``artifact_dir: str``: Artifact directory to store the files needed for deployment.
* ``auth: (Dict, optional)``: Defaults to ``None``. The default authentication is set using the ``ads.set_auth`` API. To override the default, use ``ads.common.auth.api_keys()`` or ``ads.common.auth.resource_principal()`` and create the appropriate authentication signer and the ``**kwargs`` required to instantiate the ``IdentityClient`` object.
* ``estimator: (Callable)``: Trained model.
* ``properties: (ModelProperties, optional)``: Defaults to ``None``. ModelProperties object required to save and deploy the model.
* ``serialize: (bool, optional)``: Defaults to ``True``. If ``True`` the model will be serialized into a pickle file. If ``False``, you must set the ``model_file_name`` in the ``.prepare()`` method, serialize the model manually, and save it in the ``artifact_dir``. You will also need to update the ``score.py`` file to work with this model.

.. include:: ../_template/initialize.rst

Summary Status
==============

.. include:: ../_template/summary_status.rst

.. figure:: ../figures/summary_status.png
   :align: center



Example
=======

By default, the ``GenericModel`` serializes to a pickle file. The following example, the user creates a model. In the prepare step, the user saves the model as a pickle file with the name ``toy_model.pkl``. Then the user verifies the model, saves it to the model catalog, deploys the model and makes a prediction. Finally, the user deletes the model deployment and then deletes the model.

.. code-block:: python3

    import tempfile
    from ads.model.generic_model import GenericModel

    class Toy:
        def predict(self, x):
            return x ** 2
    model = Toy()

    generic_model = GenericModel(estimator=model, artifact_dir=tempfile.mkdtemp())
    generic_model.summary_status()

    generic_model.prepare(
            inference_conda_env="dbexp_p38_cpu_v1",
            model_file_name="toy_model.pkl",
            force_overwrite=True
         )

    # Check if the artifacts are generated correctly.
    # The verify method invokes the ``predict`` function defined inside ``score.py`` in the artifact_dir
    generic_model.verify(2)

    # Register the model
    model_id = generic_model.save(display_name="Custom Model")

    # Deploy and create an endpoint for the XGBoost model
    generic_model.deploy(
        display_name="My Custom Model",
        deployment_log_group_id="ocid1.loggroup.oc1.xxx.xxxxx",
        deployment_access_log_id="ocid1.log.oc1.xxx.xxxxx",
        deployment_predict_log_id="ocid1.log.oc1.xxx.xxxxx",
    )

    print(f"Endpoint: {generic_model.model_deployment.url}")

    # Generate prediction by invoking the deployed endpoint
    generic_model.predict(2)

    # To delete the deployed endpoint uncomment the line below
    # generic_model.delete_deployment(wait_for_completion=True)

You can also use the shortcut ``.prepare_save_deploy()`` instead of calling ``.prepare()``, ``.save()`` and ``.deploy()`` seperately.

.. code-block:: python3

    import tempfile
    from ads.model.generic_model import GenericModel

    class Toy:
        def predict(self, x):
            return x ** 2
    estimator = Toy()

    model = GenericModel(estimator=estimator)
    model.summary_status()

    # If you are running the code inside a notebook session and using a service pack, `inference_conda_env` can be omitted.
    model.prepare_save_deploy(inference_conda_env="dbexp_p38_cpu_v1")
    model.verify(2)

    # Generate prediction by invoking the deployed endpoint
    model.predict(2)

    # To delete the deployed endpoint uncomment the line below
    # model.delete_deployment(wait_for_completion=True)


Example -- CatBoost
===================

Here is a more realistic  example using CatBoost model.

.. code-block:: python3

    import tempfile
    import ads
    from ads.model.generic_model import GenericModel
    from catboost import CatBoostRegressor

    ads.set_auth(auth="resource_principal")

    # Initialize data

    X_train = [[1, 4, 5, 6],
                [4, 5, 6, 7],
                [30, 40, 50, 60]]

    X_test = [[2, 4, 6, 8],
                [1, 4, 50, 60]]

    y_train = [10, 20, 30]

    # Initialize CatBoostRegressor
    catboost_estimator = CatBoostRegressor(iterations=2,
                            learning_rate=1,
                            depth=2)
    # Train a CatBoostRegressor model
    catboost_estimator.fit(X_train, y_train)

    # Get predictions
    preds = catboost_estimator.predict(X_test)

    # Instantiate ads.model.generic_model.GenericModel using the trained Custom Model using the trained CatBoost Classifier  model
    catboost_model = GenericModel(estimator=catboost_estimator,
                                artifact_dir=tempfile.mkdtemp(),
                                model_save_serializer="cloudpickle",
                                model_input_serializer="json")

    # Autogenerate score.py, pickled model, runtime.yaml, input_schema.json and output_schema.json
    catboost_model.prepare(
        inference_conda_env="oci://bucket@namespace/path/to/your/conda/pack",
        inference_python_version="your_python_version",
        X_sample=X_train,
        y_sample=y_train,
    )

    # Verify generated artifacts. Payload looks like this: [[2, 4, 6, 8], [1, 4, 50, 60]]
    catboost_model.verify(X_test, auto_serialize_data=True)

    # Register CatBoostRegressor model
    model_id = catboost_model.save(display_name="CatBoost Model")
    catboost_model.deploy()
    catboost_model.predict(X_test)
    catboost_model.delete_deployment(wait_for_completion=True)
    catboost_model.delete() # delete the model


Example -- Save Your Own Model
==============================

By default, the ``serialize`` in ``GenericModel`` class is True, and it will serialize the model using cloudpickle. However, you can set ``serialize=False`` to disable it. And serialize the model on your own. You just need to copy the serialized model into the ``.artifact_dir``. This example shows step by step how you can do that.
The example is illustrated using a Sklearn model.

.. code-block:: python3

    import tempfile
    from ads import set_auth
    from ads.model import GenericModel
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    set_auth(auth="resource_principal")

    # Load dataset and Prepare train and test split
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Train a LogisticRegression model
    sklearn_estimator = LogisticRegression()
    sklearn_estimator.fit(X_train, y_train)

    # Serialize your model. You can choose your own way to serialize your model.
    import cloudpickle
    with open("./model.pkl", "wb") as f:
        cloudpickle.dump(sklearn_estimator, f)

    model = GenericModel(sklearn_estimator, artifact_dir = "model_artifact_folder", serialize=False)
    model.prepare(inference_conda_env="generalml_p38_cpu_v1",force_overwrite=True, model_file_name="model.pkl", X_sample=X_test)

Now copy the model.pkl file and paste into the ``model_artifact_folder`` folder. And open the score.py in the ``model_artifact_folder`` folder to add implementation of the ``load_model`` function. You can also add your preprocessing steps in ``pre_inference`` function and postprocessing steps in ``post_inference`` function. Below is an example implementation of the score.py.
Replace your score.py with the code below.

.. code-block:: python3
    :emphasize-lines: 28, 29, 30, 31, 123

    # score.py 1.0 generated by ADS 2.8.2 on 20230301_065458
    import os
    import sys
    import json
    from functools import lru_cache

    model_name = 'model.pkl'


    """
    Inference script. This script is used for prediction by scoring server when schema is known.
    """

    @lru_cache(maxsize=10)
    def load_model(model_file_name=model_name):
        """
        Loads model from the serialized format

        Returns
        -------
        model:  a model instance on which predict API can be invoked
        """
        model_dir = os.path.dirname(os.path.realpath(__file__))
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        contents = os.listdir(model_dir)
        if model_file_name in contents:
            import cloudpickle
            with open(os.path.join(model_dir, model_name), "rb") as f:
                model = cloudpickle.load(f)
            return model
        else:
            raise Exception(f'{model_file_name} is not found in model directory {model_dir}')

    @lru_cache(maxsize=1)
    def fetch_data_type_from_schema(input_schema_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "input_schema.json")):
        """
        Returns data type information fetch from input_schema.json.

        Parameters
        ----------
        input_schema_path: path of input schema.

        Returns
        -------
        data_type: data type fetch from input_schema.json.

        """
        data_type = {}
        if os.path.exists(input_schema_path):
            schema = json.load(open(input_schema_path))
            for col in schema['schema']:
                data_type[col['name']] = col['dtype']
        else:
            print("input_schema has to be passed in in order to recover the same data type. pass `X_sample` in `ads.model.framework.sklearn_model.SklearnModel.prepare` function to generate the input_schema. Otherwise, the data type might be changed after serialization/deserialization.")
        return data_type

    def deserialize(data, input_schema_path):
        """
        Deserialize json serialization data to data in original type when sent to predict.

        Parameters
        ----------
        data: serialized input data.
        input_schema_path: path of input schema.

        Returns
        -------
        data: deserialized input data.

        """

        import pandas as pd
        import numpy as np
        import base64
        from io import BytesIO
        if isinstance(data, bytes):
            return data

        data_type = data.get('data_type', '') if isinstance(data, dict) else ''
        json_data = data.get('data', data) if isinstance(data, dict) else data

        if "numpy.ndarray" in data_type:
            load_bytes = BytesIO(base64.b64decode(json_data.encode('utf-8')))
            return np.load(load_bytes, allow_pickle=True)
        if "pandas.core.series.Series" in data_type:
            return pd.Series(json_data)
        if "pandas.core.frame.DataFrame" in data_type or isinstance(json_data, str):
            return pd.read_json(json_data, dtype=fetch_data_type_from_schema(input_schema_path))
        if isinstance(json_data, dict):
            return pd.DataFrame.from_dict(json_data)
        return json_data

    def pre_inference(data, input_schema_path):
        """
        Preprocess data

        Parameters
        ----------
        data: Data format as expected by the predict API of the core estimator.
        input_schema_path: path of input schema.

        Returns
        -------
        data: Data format after any processing.

        """
        return deserialize(data, input_schema_path)

    def post_inference(yhat):
        """
        Post-process the model results

        Parameters
        ----------
        yhat: Data format after calling model.predict.

        Returns
        -------
        yhat: Data format after any processing.

        """
        return yhat.tolist()

    def predict(data, model=load_model(), input_schema_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "input_schema.json")):
        """
        Returns prediction given the model and data to predict

        Parameters
        ----------
        model: Model instance returned by load_model API.
        data: Data format as expected by the predict API of the core estimator. For eg. in case of sckit models it could be numpy array/List of list/Pandas DataFrame.
        input_schema_path: path of input schema.

        Returns
        -------
        predictions: Output from scoring server
            Format: {'prediction': output from model.predict method}

        """
        features = pre_inference(data, input_schema_path)
        yhat = post_inference(
            model.predict(features)
        )
        return {'prediction': yhat}

Save the score.py and now call ``.verify()`` to check if it works locally.

.. code-block:: python3

    model.verify(X_test[:2], auto_serialize_data=True)

After verify run successfully, you can save the model to model catalog, deploy and call predict to invoke the endpoint.

.. code-block:: python3

    model_id = model.save(display_name='Demo Sklearn model')
    deploy = model.deploy(display_name='Demo Sklearn deployment')
    model.predict(X_test[:2].tolist())

You can also use the shortcut ``.prepare_save_deploy()`` instead of calling ``.prepare()``, ``.save()`` and ``.deploy()`` seperately.

.. code-block:: python3

    import tempfile
    from ads.catalog.model import ModelCatalog
    from ads.model.generic_model import GenericModel

    class Toy:
        def predict(self, x):
            return x ** 2
    estimator = Toy()

    model = GenericModel(estimator=estimator)
    model.summary_status()
    # If you are running the code inside a notebook session and using a service pack, `inference_conda_env` can be omitted.
    model.prepare_save_deploy(inference_conda_env="dataexpl_p37_cpu_v3")
    model.verify(2)
    model.predict(2)
    model.delete_deployment(wait_for_completion=True)
    model.delete()
