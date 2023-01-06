Quick Start
***********

Deployment Examples
===================

The following sections provide sample code to create and deploy a model.

AutoMLModel
-----------

Create a model, prepare it, verify that it works, save it to the model catalog, deploy it, make a prediction, and then delete the deployment.

.. code-block:: python3

    import logging
    import tempfile
    import warnings
    from ads.automl.driver import AutoML
    from ads.automl.provider import OracleAutoMLProvider
    from ads.catalog.model import ModelCatalog
    from ads.common.model_metadata import UseCaseType
    from ads.dataset.dataset_browser import DatasetBrowser
    from ads.model.framework.automl_model import AutoMLModel

    ds = DatasetBrowser.sklearn().open("wine").set_target("target")
    train, test = ds.train_test_split(test_size=0.1, random_state = 42)

    ml_engine = OracleAutoMLProvider(n_jobs=-1, loglevel=logging.ERROR)
    oracle_automl = AutoML(train, provider=ml_engine)
    model, baseline = oracle_automl.train(
            model_list=['LogisticRegression', 'DecisionTreeClassifier'],
            random_state = 42,
            time_budget = 500
        )

    artifact_dir = tempfile.mkdtemp()
    automl_model = AutoMLModel(estimator=model, artifact_dir=artifact_dir)
    automl_model.prepare(inference_conda_env="generalml_p37_cpu_v1",
                         training_conda_env="generalml_p37_cpu_v1",
                         use_case_type=UseCaseType.BINARY_CLASSIFICATION,
                         X_sample=test.X,
                         force_overwrite=True)
    automl_model.verify(test.X.iloc[:10])
    model_id = automl_model.save(display_name='Demo AutoMLModel model')
    deploy = automl_model.deploy(display_name='Demo AutoMLModel deployment')
    automl_model.predict(test.X.iloc[:10])
    automl_model.delete_deployment(wait_for_completion=True)
    ModelCatalog(compartment_id=os.environ['NB_SESSION_COMPARTMENT_OCID']).delete_model(model_id)

GenericModel
------------

Create a model, prepare it, verify that it works, save it to the model catalog, deploy it, make a prediction, and then delete the deployment.

.. code-block:: python3

    import tempfile
    from ads.catalog.model import ModelCatalog
    from ads.model.generic_model import GenericModel

    class Toy:
        def predict(self, x):
            return x ** 2
    estimator = Toy()

    model = GenericModel(estimator=estimator, artifact_dir=tempfile.mkdtemp())
    model.summary_status()
    model.prepare(inference_conda_env="dataexpl_p37_cpu_v3")
    model.verify(2)
    model_id = model.save()
    model.deploy()
    model.predict(2)
    model.delete_deployment(wait_for_completion=True)
    ModelCatalog(compartment_id=os.environ['NB_SESSION_COMPARTMENT_OCID']).delete_model(model_id)


LightGBMModel
-------------

Create a model, prepare it, verify that it works, save it to the model catalog, deploy it, make a prediction, and then delete the deployment.

.. code-block:: python3

    import lightgbm as lgb
    import tempfile
    from ads.catalog.model import ModelCatalog
    from ads.model.framework.lightgbm_model import LightGBMModel
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    train = lgb.Dataset(X_train, label=y_train)
    param = {
      'objective': 'multiclass', 'num_class': 3,
    }
    lightgbm_estimator = lgb.train(param, train)
    lightgbm_model = LightGBMModel(estimator=lightgbm_estimator, artifact_dir=tempfile.mkdtemp())
    lightgbm_model.prepare(inference_conda_env="generalml_p37_cpu_v1")
    lightgbm_model.verify(X_test)
    model_id = lightgbm_model.save()
    model_deployment = lightgbm_model.deploy()
    lightgbm_model.predict(X_test)
    lightgbm_model.delete_deployment(wait_for_completion=True)
    ModelCatalog(compartment_id=os.environ['NB_SESSION_COMPARTMENT_OCID']).delete_model(model_id)


PyTorchModel
------------

Create a model, prepare it, verify that it works, save it to the model catalog, deploy it, make a prediction, and then delete the deployment.

.. code-block:: python3


    import tempfile
    import torch
    import torchvision
    from ads.catalog.model import ModelCatalog
    from ads.model.framework.pytorch_model import PyTorchModel

    torch_estimator = torchvision.models.resnet18(pretrained=True)
    torch_estimator.eval()

    # create fake test data
    test_data = torch.randn(1, 3, 224, 224)

    artifact_dir = tempfile.mkdtemp()
    torch_model = PyTorchModel(torch_estimator, artifact_dir=artifact_dir)
    torch_model.prepare(inference_conda_env="generalml_p37_cpu_v1")

    # Update ``score.py`` by constructing the model class instance first.
    added_line = """
    import torchvision
    the_model = torchvision.models.resnet18()
    """
    with open(artifact_dir + "/score.py", 'r+') as f:
          content = f.read()
          f.seek(0, 0)
          f.write(added_line.rstrip('\r\n') + '\n' + content)

    # continue to save and deploy the model.
    torch_model.verify(test_data)
    model_id = torch_model.save()
    model_deployment = torch_model.deploy()
    torch_model.predict(test_data)
    torch_model.delete_deployment(wait_for_completion=True)
    ModelCatalog(compartment_id=os.environ['NB_SESSION_COMPARTMENT_OCID']).delete_model(model_id)


SklearnModel
------------

Create a model, prepare it, verify that it works, save it to the model catalog, deploy it, make a prediction, and then delete the deployment.

.. code-block:: python3

    import tempfile
    from ads.catalog.model import ModelCatalog
    from ads.model.framework.sklearn_model import SklearnModel
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    sklearn_estimator = LogisticRegression()
    sklearn_estimator.fit(X_train, y_train)

    sklearn_model = SklearnModel(estimator=sklearn_estimator, artifact_dir=tempfile.mkdtemp())
    sklearn_model.prepare(inference_conda_env="dataexpl_p37_cpu_v3")
    sklearn_model.verify(X_test)
    model_id = sklearn_model.save()
    model_deployment = sklearn_model.deploy()
    sklearn_model.predict(X_test)
    sklearn_model.delete_deployment(wait_for_completion=True)
    ModelCatalog(compartment_id=os.environ['NB_SESSION_COMPARTMENT_OCID']).delete_model(model_id)


TensorFlowModel
---------------

Create a model, prepare it, verify that it works, save it to the model catalog, deploy it, make a prediction, and then delete the deployment.

.. code-block:: python3

    from ads.catalog.model import ModelCatalog
    from ads.model.framework.tensorflow_model import TensorFlowModel
    import tempfile
    import tensorflow as tf

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    tf_estimator = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10),
            ]
        )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    tf_estimator.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    tf_estimator.fit(x_train, y_train, epochs=1)

    tf_model = TensorFlowModel(tf_estimator, artifact_dir=tempfile.mkdtemp())
    tf_model.prepare(inference_conda_env="generalml_p37_cpu_v1")
    tf_model.verify(x_test[:1])
    model_id = tf_model.save()
    model_deployment = tf_model.deploy()
    tf_model.predict(x_test[:1])
    tf_model.delete_deployment(wait_for_completion=True)
    ModelCatalog(compartment_id=os.environ['NB_SESSION_COMPARTMENT_OCID']).delete_model(model_id)


XGBoostModel
------------

Create a model, prepare it, verify that it works, save it to the model catalog, deploy it, make a prediction, and then delete the deployment.

.. code-block:: python3

    import tempfile
    import xgboost as xgb
    from ads.catalog.model import ModelCatalog
    from ads.model.framework.xgboost_model import XGBoostModel
    from sklearn.datasets import load_iris
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    xgboost_estimator = xgb.XGBClassifier()
    xgboost_estimator.fit(X_train, y_train)
    xgboost_model = XGBoostModel(estimator=xgboost_estimator, artifact_dir=tempfile.mkdtemp())
    xgboost_model.prepare(inference_conda_env="generalml_p37_cpu_v1")
    xgboost_model.verify(X_test)
    model_id = xgboost_model.save()
    model_deployment = xgboost_model.deploy()
    xgboost_model.predict(X_test)
    xgboost_model.delete_deployment(wait_for_completion=True)
    ModelCatalog(compartment_id=os.environ['NB_SESSION_COMPARTMENT_OCID']).delete_model(model_id)

Shortcut
========
.. versionadded:: 2.6.3

Create a model and call the ``prepare_save_deploy`` method to prepare, save, and deploy in one step, make a prediction, and then delete the deployment.

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
    ModelCatalog(compartment_id=os.environ['NB_SESSION_COMPARTMENT_OCID']).delete_model(model.model_id)


Logging
=======

Model deployments have the option to log access and prediction traffic. The access log, logs requests to the model deployment endpoint. The prediction logs record the predictions that the model endpoint makes. Logs must belong to a log group.

The following example uses the ``OCILogGroup`` class to create a log group and two logs (access and predict). When a model is deployed, the OCIDs of these resources are passed to the ``.deploy()`` method.

You can access logs through APIs, the ``oci`` CLI, or the Console. The following example uses the ADS ``.show_logs()`` method, to access the predict and access log objects in the ``model_deployment`` module.

.. code-block:: python3

    import tempfile
    from ads.common.oci_logging import OCILogGroup
    from ads.model.generic_model import GenericModel

    # Create a log group and logs
    log_group = OCILogGroup(display_name="Model Deployment Log Group").create()
    access_log = log_group.create_log("Model Deployment Access Log")
    predict_log = log_group.create_log("Model Deployment Predict Log")

    # Create a generic model that will be deployed
    class Toy:
        def predict(self, x):
            return x ** 2

    model = Toy()

    # Deploy the model
    model = GenericModel(estimator=model, artifact_dir=tempfile.mkdtemp())
    model.summary_status()
    model.prepare(inference_conda_env="dataexpl_p37_cpu_v3")
    model.verify(2)
    model.save()
    model.deploy(
        deployment_log_group_id=log_group.id,
        deployment_access_log_id=access_log.id,
        deployment_predict_log_id=predict_log.id,
    )

    # Make a prediction and view the logs
    model.predict(2)
    model.model_deployment.show_logs(log_type="predict")
    model.model_deployment.show_logs(log_type="access")
    model.model_deployment.access_log.tail()
    model.model_deployment.predict_log.tail()

