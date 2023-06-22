#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.model.framework.spark_model import SparkPipelineModel

import tempfile
from pyspark.ml.linalg import Vectors
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer


spark = SparkSession.builder.appName("Python Spark SQL basic example").getOrCreate()
artifact_dir = tempfile.mkdtemp()


def generate_data1():
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

    return training, test


def build_spark_pipeline1(training, test):
    # Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.001)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

    # Fit the pipeline to training documents.
    model = pipeline.fit(training)
    prediction = model.transform(test).toPandas()["prediction"].to_list()
    return model, prediction


def spark_pipeline_script1():
    training, test = generate_data1()
    model, prediction = build_spark_pipeline1(training, test)

    return {
        "framework": SparkPipelineModel,
        "estimator": model,
        "artifact_dir": "./artifact_folder/spark",
        "inference_conda_env": "pyspark30_p37_cpu_v5",
        "inference_python_version": "3.7",
        "model_file_name": "model",
        "data": test,
        "y_true": prediction,
        "onnx_data": None,
        "prepare_args": dict(),
        "local_pred": prediction,
        "score_py_path": "scripts/spark_pipeline_local_score.py",
    }


def generate_data2():
    training = spark.createDataFrame(
        [
            (1.0, Vectors.dense([0.0, 1.1, 0.1])),
            (0.0, Vectors.dense([2.0, 1.0, -1.0])),
            (0.0, Vectors.dense([2.0, 1.3, 1.0])),
            (1.0, Vectors.dense([0.0, 1.2, -0.5])),
        ],
        ["label", "features"],
    )

    return training, training


def build_spark_pipeline2(training, test):
    lr = LogisticRegression(maxIter=10, regParam=0.01)
    pipeline = Pipeline(stages=[lr])
    model = pipeline.fit(training)
    prediction = model.transform(test).toPandas()["prediction"].to_list()
    return model, prediction


def spark_pipeline_script2():
    training, test = generate_data2()
    model, prediction = build_spark_pipeline2(training, test)

    return {
        "framework": SparkPipelineModel,
        "estimator": model,
        "artifact_dir": "./artifact_folder/spark",
        "inference_conda_env": "pyspark30_p37_cpu_v5",
        "inference_python_version": "3.7",
        "model_file_name": "model",
        "data": test,
        "y_true": prediction,
        "onnx_data": None,
        "prepare_args": dict(),
        "local_pred": prediction,
        "score_py_path": "scripts/spark_pipeline_local_score.py",
    }


def spark_pipeline_script3():
    training, test = generate_data2()
    model, prediction = build_spark_pipeline2(training, test)

    return {
        "framework": SparkPipelineModel,
        "estimator": model,
        "artifact_dir": "./artifact_folder/spark",
        "inference_conda_env": "pyspark30_p37_cpu_v5",
        "inference_python_version": "3.7",
        "model_file_name": "model",
        "data": test.take(2),
        "y_true": prediction[:2],
        "onnx_data": None,
        "prepare_args": {"X_sample": test.take(2)},
        "local_pred": prediction,
        "score_py_path": "scripts/spark_pipeline_local_score.py",
    }
