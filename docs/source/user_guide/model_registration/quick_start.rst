Quick Start
***********

ADS can auto generate the required files to register and deploy your models. Checkout the examples below to learn how to deploy models of different frameworks.

Sklearn
-------

.. code-block:: python3

    import tempfile
    from ads.model.framework.sklearn_model import SklearnModel
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Load dataset and Prepare train and test split
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Train a LogisticRegression model
    sklearn_estimator = LogisticRegression()
    sklearn_estimator.fit(X_train, y_train)

    # Instantite ads.model.framework.sklearn_model.SklearnModel using the sklearn LogisticRegression model
    sklearn_model = SklearnModel(
        estimator=sklearn_estimator, artifact_dir=tempfile.mkdtemp()
    )

    # Autogenerate score.py, pickled model, runtime.yaml, input_schema.json and output_schema.json
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
    import xgboost as xgb
    from ads.model.framework.xgboost_model import XGBoostModel
    from sklearn.datasets import load_iris
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Load dataset and Prepare train and test split
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Train a XBoost Classifier  model
    xgboost_estimator = xgb.XGBClassifier()
    xgboost_estimator.fit(X_train, y_train)

    # Instantite ads.model.framework.xgboost_model.XGBoostModel using the trained XGBoost Model
    xgboost_model = XGBoostModel(estimator=xgboost_estimator, artifact_dir=tempfile.mkdtemp())

    # Autogenerate score.py, pickled model, runtime.yaml, input_schema.json and output_schema.json
    xgboost_model.prepare(inference_conda_env="generalml_p38_cpu_v1")

    # Verify generated artifacts
    xgboost_model.verify(X_test)

    # Register XGBoost model
    model_id = xgboost_model.save(display_name="XGBoost Model")

LightGBM
--------

Create a model, prepare it, verify that it works, save it to the model catalog, deploy it, make a prediction, and then delete the deployment.

.. code-block:: python3

    import lightgbm as lgb
    import tempfile
    from ads.model.framework.lightgbm_model import LightGBMModel
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

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

    # Instantite ads.model.lightgbm_model.XGBoostModel using the trained LGBM Model
    lightgbm_model = LightGBMModel(estimator=lightgbm_estimator, artifact_dir=tempfile.mkdtemp())

    # Autogenerate score.py, pickled model, runtime.yaml, input_schema.json and output_schema.json
    lightgbm_model.prepare(inference_conda_env="generalml_p38_cpu_v1")

    # Verify generated artifacts
    lightgbm_model.verify(X_test)

    # Register LightGBM model
    model_id = lightgbm_model.save(display_name="LightGBM Model")


PyTorch
-------

Create a model, prepare it, verify that it works, save it to the model catalog, deploy it, make a prediction, and then delete the deployment.

.. code-block:: python3


    import tempfile
    import torch
    import torchvision
    from ads.model.framework.pytorch_model import PyTorchModel

    # Load a pre-trained resnet model
    torch_estimator = torchvision.models.resnet18(pretrained=True)
    torch_estimator.eval()

    # create random test data
    test_data = torch.randn(1, 3, 224, 224)

    # Instantite ads.model.framework.pytorch_model.PyTorchModel using the pre-trained PyTorch Model
    artifact_dir=tempfile.mkdtemp()
    torch_model = PyTorchModel(torch_estimator, artifact_dir=artifact_dir)

    # Autogenerate score.py, pickled model, runtime.yaml, input_schema.json and output_schema.json
    # Set `save_entire_model` to `True` to save the model as Torchscript program.
    torch_model.prepare(inference_conda_env="pytorch110_p38_cpu_v1", use_torch_script=True)

    # Verify generated artifacts
    torch_model.verify(test_data)

    #Register PyTorch model
    model_id = torch_model.save(display_name="PyTorch Model")


Spark Pipeline
--------------

Create a model, prepare it, verify that it works, save it to the model catalog, deploy it, make a prediction, and then delete the deployment.

.. code-block:: python3

    import tempfile
    import os
    from pyspark.sql import SparkSession
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

    # Instantite ads.model.framework.tensorflow_model.TensorFlowModel using the pre-trained TensorFlow Model
    tf_model = TensorFlowModel(tf_estimator, artifact_dir=tempfile.mkdtemp())

    # Autogenerate score.py, pickled model, runtime.yaml, input_schema.json and output_schema.json
    tf_model.prepare(inference_conda_env="tensorflow28_p38_cpu_v1")

    # Verify generated artifacts
    tf_model.verify(x_test[:1])

    #Register TensorFlow model
    model_id = tf_model.save(display_name="TensorFlow Model")

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

