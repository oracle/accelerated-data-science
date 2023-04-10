Quick Start
***********

ADS can auto generate the required files to register and deploy your models. Checkout the examples below to learn how to deploy models of different frameworks.

Sklearn
-------

.. code-block:: python3

    import tempfile

    import ads
    from ads.model.framework.sklearn_model import SklearnModel
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split


    ads.set_auth(auth="resource_principal")

    # Load dataset and Prepare train and test split
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Train a LogisticRegression model
    sklearn_estimator = LogisticRegression()
    sklearn_estimator.fit(X_train, y_train)

    # Instantiate ads.model.framework.sklearn_model.SklearnModel using the sklearn LogisticRegression model
    sklearn_model = SklearnModel(
        estimator=sklearn_estimator, artifact_dir=tempfile.mkdtemp()
    )

    # Autogenerate score.py, serialized model, runtime.yaml, input_schema.json and output_schema.json
    sklearn_model.prepare(
        inference_conda_env="dbexp_p38_cpu_v1",
        X_sample=X_train,
        y_sample=y_train,
    )

    # Verify generated artifacts
    sklearn_model.verify(X_test)

    # Register scikit-learn model
    model_id = sklearn_model.save(display_name="Sklearn Model")


XGBoost
-------

Create a model, prepare it, verify that it works, save it to the model catalog, deploy it, make a prediction, and then delete the deployment.

.. code-block:: python3

    import tempfile

    import ads
    import xgboost as xgb
    from ads.model.framework.xgboost_model import XGBoostModel
    from sklearn.datasets import load_iris
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split


    ads.set_auth(auth="resource_principal")

    # Load dataset and Prepare train and test split
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Train a XBoost Classifier  model
    xgboost_estimator = xgb.XGBClassifier()
    xgboost_estimator.fit(X_train, y_train)

    # Instantiate ads.model.framework.xgboost_model.XGBoostModel using the trained XGBoost Model
    xgboost_model = XGBoostModel(estimator=xgboost_estimator, artifact_dir=tempfile.mkdtemp())

    # Autogenerate score.py, serialized model, runtime.yaml, input_schema.json and output_schema.json
    xgboost_model.prepare(
        inference_conda_env="generalml_p38_cpu_v1",
        X_sample=X_train,
        y_sample=y_train,
    )

    # Verify generated artifacts
    xgboost_model.verify(X_test)

    # Register XGBoost model
    model_id = xgboost_model.save(display_name="XGBoost Model")

LightGBM
--------

Create a model, prepare it, verify that it works, save it to the model catalog, deploy it, make a prediction, and then delete the deployment.

.. code-block:: python3

    import tempfile

    import ads
    import lightgbm as lgb
    from ads.model.framework.lightgbm_model import LightGBMModel
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    ads.set_auth(auth="resource_principal")

    # Load dataset and Prepare train and test split
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Train a XBoost Classifier  model
    train = lgb.Dataset(X_train, label=y_train)
    param = {
      'objective': 'multiclass', 'num_class': 3,
    }
    lightgbm_estimator = lgb.train(param, train)

    # Instantiate ads.model.lightgbm_model.XGBoostModel using the trained LGBM Model
    lightgbm_model = LightGBMModel(estimator=lightgbm_estimator, artifact_dir=tempfile.mkdtemp())

    # Autogenerate score.py, serialized model, runtime.yaml, input_schema.json and output_schema.json
    lightgbm_model.prepare(
        inference_conda_env="generalml_p38_cpu_v1",
        X_sample=X_train,
        y_sample=y_train,
    )

    # Verify generated artifacts
    lightgbm_model.verify(X_test)

    # Register LightGBM model
    model_id = lightgbm_model.save(display_name="LightGBM Model")


PyTorch
-------

Create a model, prepare it, verify that it works, save it to the model catalog, deploy it, make a prediction, and then delete the deployment.

.. code-block:: python3


    import tempfile

    import ads
    import torch
    import torchvision
    from ads.model.framework.pytorch_model import PyTorchModel

    ads.set_auth(auth="resource_principal")

    # Load a pre-trained resnet model
    torch_estimator = torchvision.models.resnet18(pretrained=True)
    torch_estimator.eval()

    # create random test data
    test_data = torch.randn(1, 3, 224, 224)

    # Instantiate ads.model.framework.pytorch_model.PyTorchModel using the pre-trained PyTorch Model
    artifact_dir=tempfile.mkdtemp()
    torch_model = PyTorchModel(torch_estimator, artifact_dir=artifact_dir)

    # Autogenerate score.py, serialized model, runtime.yaml
    # Set `use_torch_script` to `True` to save the model as Torchscript program.
    torch_model.prepare(inference_conda_env="pytorch110_p38_cpu_v1", use_torch_script=True)

    # Verify generated artifacts
    torch_model.verify(test_data)

    # Register PyTorch model
    model_id = torch_model.save(display_name="PyTorch Model")


Spark Pipeline
--------------

Create a model, prepare it, verify that it works, save it to the model catalog, deploy it, make a prediction, and then delete the deployment.

.. code-block:: python3

    import os
    import tempfile

    import ads
    from ads.model.framework.spark_model import SparkPipelineModel
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import HashingTF, Tokenizer
    from ads.model.framework.spark_model import SparkPipelineModel

    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .getOrCreate()

    # create data
    training = spark.createDataFrame(
        [
            (0, "a b c d e spark", 1.0),
            (1, "b d", 0.0),
            (2, "spark f g h", 1.0),
            (3, "hadoop mapreduce", 0.0),
        ],
        ["id", "text", "label"],
    )
    test = spark.createDataFrame(
        [
            (4, "spark i j k"),
            (5, "l m n"),
            (6, "spark hadoop spark"),
            (7, "apache hadoop"),
        ],
        ["id", "text"],
    )

    # Train a Spark Pipeline model
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.001)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
    model = pipeline.fit(training)

    # Instantite ads.model.framework.spark_model.SparkPipelineModel using the pre-trained Spark Pipeline Model
    spark_model = SparkPipelineModel(estimator=model, artifact_dir=tempfile.mkdtemp())
    spark_model.prepare(inference_conda_env="pyspark32_p38_cpu_v2",
                        X_sample = training,
                        force_overwrite=True)

    # Verify generated artifacts
    prediction = spark_model.verify(test)

    #Register Spark model
    spark_model.save(display_name="Spark Pipeline Model")


TensorFlow
----------

Create a model, prepare it, verify that it works, save it to the model catalog, deploy it, make a prediction, and then delete the deployment.

.. code-block:: python3

    from ads.model.framework.tensorflow_model import TensorFlowModel
    import tensorflow as tf
    from uuid import uuid4

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

    # Instantite ads.model.framework.tensorflow_model.TensorFlowModel using the pre-trained TensorFlow Model
    tf_model = TensorFlowModel(tf_estimator, artifact_dir=f"./model-artifact-{str(uuid4())}")

    # Autogenerate score.py, pickled model, runtime.yaml, input_schema.json and output_schema.json
    tf_model.prepare(inference_conda_env="tensorflow28_p38_cpu_v1")

    # Verify generated artifacts
    tf_model.verify(x_test[:1])

    #Register TensorFlow model
    model_id = tf_model.save(display_name="TensorFlow Model")


HuggingFace Pipelines
---------------------

.. code-block:: python3

    from transformers import pipeline
    from ads.model import HuggingFacePipelineModel

    import tempfile
    import PIL.Image
    from ads.common.auth import default_signer
    import requests
    import cloudpickle

    ## download the image
    image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
    image = PIL.Image.open(requests.get(image_url, stream=True).raw)

    ## download the pretrained model
    classifier = pipeline(model="openai/clip-vit-large-patch14")
    classifier(
            images=image,
            candidate_labels=["animals", "humans", "landscape"],
        )

    ## Initiate a HuggingFacePipelineModel instance
    zero_shot_image_classification_model = HuggingFacePipelineModel(classifier, artifact_dir=tempfile.mkdtemp())

    # Autogenerate score.py, serialized model, runtime.yaml
    conda_pack_path = "oci://bucket@namespace/path/to/conda/pack"
    python_version = "3.x" # Remember to update 3.x with your actual python version, e.g. 3.8
    zero_shot_image_classification_model.prepare(inference_conda_env=conda_pack_path, inference_python_version = python_version, force_overwrite=True)

    ## Convert payload to bytes
    data = {"images": image, "candidate_labels": ["animals", "humans", "landscape"]}
    body = cloudpickle.dumps(data) # convert image to bytes

    # Verify generated artifacts
    zero_shot_image_classification_model.verify(data=data)
    zero_shot_image_classification_model.verify(data=body)

    # Register HuggingFace Pipeline model
    zero_shot_image_classification_model.save()

    ## Deploy
    log_group_id = "<log_group_id>"
    log_id = "<log_id>"
    zero_shot_image_classification_model.deploy(deployment_bandwidth_mbps=100,
                    wait_for_completion=False,
                    deployment_log_group_id = log_group_id,
                    deployment_access_log_id = log_id,
                    deployment_predict_log_id = log_id)
    zero_shot_image_classification_model.predict(data)
    zero_shot_image_classification_model.predict(body)

    ### Invoke the model by sending bytes
    auth = default_signer()['signer']
    endpoint = zero_shot_image_classification_model.model_deployment.url + "/predict"
    headers = {"Content-Type": "application/octet-stream"}
    requests.post(endpoint, data=body, auth=auth, headers=headers).json()


Other Frameworks
----------------

.. code-block:: python3

    import tempfile
    from ads.model.generic_model import GenericModel

    # Create custom framework model
    class Toy:
        def predict(self, x):
            return x ** 2
    model = Toy()

    # Instantite ads.model.generic_model.GenericModel using the trained Custom Model
    generic_model = GenericModel(estimator=model, artifact_dir=tempfile.mkdtemp())
    generic_model.summary_status()

    # Autogenerate score.py, pickled model, runtime.yaml, input_schema.json and output_schema.json
    generic_model.prepare(
            inference_conda_env="dbexp_p38_cpu_v1",
            model_file_name="toy_model.pkl",
            force_overwrite=True
         )

    # Check if the artifacts are generated correctly.
    # The verify method invokes the ``predict`` function defined inside ``score.py`` in the artifact_dir
    generic_model.verify([2])

    # Register the model
    model_id = generic_model.save(display_name="Custom Framework Model")


With Model Version Set
----------------------
.. code-block:: python3

    import tempfile
    from ads.model.generic_model import GenericModel

    # Create custom framework model
    class Toy:
        def predict(self, x):
            return x ** 2
    model = Toy()

    # Instantite ads.model.generic_model.GenericModel using the trained Custom Model
    generic_model = GenericModel(estimator=model, artifact_dir=tempfile.mkdtemp())
    generic_model.summary_status()

    
    # Within the context manager, you can save the :ref:`Model Serialization` model without specifying the ``model_version_set`` parameter because it's taken from the model context manager. If the model version set doesn't exist in the model catalog, the example creates a model version set named ``my_model_version_set``.  If the model version set exists in the model catalog, the models are saved to that model version set.
    with ads.model.experiment(name="my_model_version_set", create_if_not_exists=True):

        # Autogenerate score.py, pickled model, runtime.yaml, input_schema.json and output_schema.json
        generic_model.prepare(
                inference_conda_env="dbexp_p38_cpu_v1",
                model_file_name="toy_model.pkl",
                force_overwrite=True
            )

        # Check if the artifacts are generated correctly.
        # The verify method invokes the ``predict`` function defined inside ``score.py`` in the artifact_dir
        generic_model.verify([2])

        # Register the model
        model_id = generic_model.save(display_name="Custom Framework Model")

ADS CLI
-------

**Prerequisites**

1. :doc:`Install ADS CLI<../cli/quickstart>`
2. :doc:`Configure Defaults<../cli/opctl/configure>`

Deploy
~~~~~~

To deploy a model, provide the path to the model deployment YAML file with the ``--file`` option

.. code-block:: shell

  ads opctl run --file <path_to_model_deployment_yaml>


Monitor
~~~~~~~

To monitor a model deployment, provide the model deployment OCID and provide the log type with the ``-l`` option

Below is an example to stream the access log

.. code-block:: shell

  ads opctl watch <model_deployment_ocid> -l access

.. admonition:: Tip

  The allowed values for ``-l`` option are ``access``, ``predict``, or ``None``.


Activate
~~~~~~~~

To activate a model deployment, provide the model deployment OCID

.. code-block:: shell

  ads opctl activate <model_deployment_ocid>

Data Science Model Deployments can only be activated when they are in the `INACTIVE` state.


Deactivate
~~~~~~~~~~

To deactivate a model deployment, provide the model deployment OCID

.. code-block:: shell

  ads opctl deactivate <model_deployment_ocid>

Data Science Model Deployments can only be deactivated when they are in the `ACTIVE` state.


Delete
~~~~~~

To delete a model deployment, provide the model deployment OCID

.. code-block:: shell

  ads opctl delete <model_deployment_ocid>

Data Science Model Deployments can only be deleted when they are in the `ACTIVE`, `INACTIVE`, or `FAILED` state.
