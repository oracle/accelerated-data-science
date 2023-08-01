#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


from os import path
from os import cpu_count
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.feature import StandardScaler, VectorAssembler


from pyspark.sql.types import StructType, IntegerType, DateType


spark = (
    SparkSession.builder.appName("Python Spark SQL basic example")
    .config("spark.driver.cores", str(max(1, cpu_count() - 1)))
    .config("spark.executor.cores", str(max(1, cpu_count() - 1)))
    .getOrCreate()
)
oracle_attrition = spark.read.csv(
    "oci://mayoor-dev@ociodscdev/oracle_fraud_dataset1.csv",
    header=True,
    inferSchema=True,
)
print(oracle_attrition.columns)
print(oracle_attrition.dtypes)
oracle_attrition.createOrReplaceTempView("ORCL_ATTR")
spark.sql("select col01,col011  from ORCL_ATTR limit 10").show()
from pyspark.sql.functions import countDistinct

oracle_attrition.groupBy("anomalous").count().show()
spark.sql("select distinct anomalous from ORCL_ATTR").show()
data_column = [col for col in oracle_attrition.columns if col != "anomalous"]
print(data_column)
vec_assembler = VectorAssembler(inputCols=data_column, outputCol="features")
oracle_attrition = vec_assembler.transform(oracle_attrition)
train, test = oracle_attrition.randomSplit([0.8, 0.2], seed=42)
train.groupBy("anomalous").count().show()
test.groupBy("anomalous").count().show()
gbt_model = GBTClassifier(maxIter=5, maxDepth=2, labelCol="anomalous", seed=42)
model = gbt_model.fit(train)
predictions = model.transform(test)
predictions.select("prediction").show(10)
predictions.groupBy("prediction", "anomalous").count().show()
evaluator = MulticlassClassificationEvaluator(
    labelCol="anomalous", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(
    "F-1 Score:{}".format(evaluator.evaluate(predictions, {evaluator.metricName: "f1"}))
)
print("Test Error = %g" % (1.0 - accuracy))
print(accuracy)
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics

metrics = BinaryClassificationMetrics(
    predictions.select(
        predictions.prediction.cast("double"), predictions.anomalous.cast("double")
    ).rdd
)
print("AUC", metrics.areaUnderPR)
print("AUC-ROC", metrics.areaUnderROC)
