#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for model frameworks. Includes tests for:
 - SparkPipelineModel
"""
import os
import shutil
import tempfile
import pytest
import numpy as np
from packaging import version
from ads.model.framework.spark_model import SparkPipelineModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer


spark = SparkSession.builder.appName("Python Spark SQL basic example").getOrCreate()
artifact_dir1 = tempfile.mkdtemp()
artifact_dir2 = tempfile.mkdtemp()


def setup_module():
    os.makedirs(artifact_dir1, exist_ok=True)
    os.makedirs(artifact_dir2, exist_ok=True)


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


class TestSparkPipelineModel:
    """Unittests for the SparkPipelineModel class."""

    inference_conda_env = "pyspark30_p37_cpu_v5"
    inference_python_version = "3.7"
    model_file_name = "model"

    training1, test1 = generate_data1()
    model1, pred1 = build_spark_pipeline1(training1, test1)
    spark_model1 = SparkPipelineModel(estimator=model1, artifact_dir=artifact_dir1)

    training2, test2 = generate_data2()
    model2, pred2 = build_spark_pipeline2(training2, test2)
    spark_model2 = SparkPipelineModel(estimator=model2, artifact_dir=artifact_dir2)

    model_group = [
        {
            "training": training1,
            "test": test1,
            "model": model1,
            "pred": pred1,
            "spark_model": spark_model1,
            "artifact_dir": artifact_dir1,
        },
        {
            "training": training2,
            "test": test2,
            "model": model2,
            "pred": pred2,
            "spark_model": spark_model2,
            "artifact_dir": artifact_dir2,
        },
    ]

    @pytest.mark.parametrize("model_data", model_group)
    def test_serialize_with_incorrect_model_file_name_pt(self, model_data):
        """
        Test wrong model_file_name format.
        """
        test_pytorch_model = SparkPipelineModel(
            model_data["model"],
            model_data["artifact_dir"],
        )
        with pytest.raises(NotImplementedError):
            test_pytorch_model._handle_model_file_name(
                as_onnx=True, model_file_name="model.onnx"
            )

    @pytest.mark.parametrize("model_data", model_group)
    def test_bad_inputs(self, model_data):
        """
        {
        "training": training1,
            "test": test1,
            "model": model1,
            "pred": pred1,
            "spark_model": spark_model1,
            "artifact_dir":artifact_dir1,
        }
        """
        model = model_data["spark_model"]
        test = model_data["test"]
        pred = model_data["pred"]
        model.prepare(
            inference_conda_env=self.inference_conda_env,
            model_file_name=self.model_file_name,
            inference_python_version=self.inference_python_version,
            force_overwrite=True,
            training_id=None,
            X_sample=test,
            y_sample=pred,
        )
        with pytest.raises(AttributeError):
            model.prepare(
                inference_conda_env=self.inference_conda_env,
                model_file_name=self.model_file_name,
                inference_python_version=self.inference_python_version,
                force_overwrite=True,
                training_id=None,
                X_sample=test,
                y_sample=pred,
                as_onnx=True,
            )
        with pytest.raises(TypeError):
            model.prepare(
                inference_conda_env=self.inference_conda_env,
                model_file_name=self.model_file_name,
                inference_python_version=self.inference_python_version,
                force_overwrite=True,
                training_id=None,
            )

        with pytest.raises(ValueError):
            model.prepare(
                inference_conda_env=self.inference_conda_env,
                model_file_name=self.model_file_name,
                inference_python_version=self.inference_python_version,
                force_overwrite=False,
                training_id=None,
                X_sample=test,
                y_sample=pred,
            )

        assert (
            pred == model.verify(test)["prediction"]
        ), "normal verify, normal test is failing"
        assert (
            pred == model.verify(test.take(test.count()))["prediction"]
        ), "spark sql DF sampling not working in verify"
        assert (
            pred == model.verify(test.toPandas())["prediction"]
        ), "spark sql converting to pandas not working in verify"
        if version.parse(spark.version) >= version.parse("3.2.0"):
            assert (
                pred == model.verify(test.to_pandas_on_spark())["prediction"]
            ), "spark sql converting to pandas on spark not working in verify"
        assert (
            pred[:1] == model.verify(test.toJSON().collect()[0])["prediction"]
        ), "failed when passing in a single json serialized row as a str"
        assert (
            pred[:2] == model.verify(test.toPandas().head(2))["prediction"]
        ), "failed when passing in a pandas df"

        with pytest.raises(TypeError):
            model.verify(test.take(0))
        with pytest.raises(Exception):
            model.verify(np.ones(test.toPandas().shape))


def teardown_module():
    shutil.rmtree(artifact_dir1)
    shutil.rmtree(artifact_dir2)
