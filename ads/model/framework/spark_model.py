#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import numpy as np
import pandas as pd
import fsspec
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from ads.common import logger, utils
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common.object_storage_details import ObjectStorageDetails
from ads.model.extractor.spark_extractor import SparkExtractor
from ads.model.serde.model_input import SparkModelInputSerializerType
from ads.model.serde.model_serializer import SparkModelSerializerType
from ads.model.generic_model import (
    FrameworkSpecificModel,
    DEFAULT_MODEL_FOLDER_NAME,
    _prepare_artifact_dir,
)
from ads.model.model_properties import ModelProperties
from ads.model.serde.common import SERDE

SPARK_DATAFRAME_SCHEMA_PATH = "_input_data_schema.json"


class SparkPipelineModel(FrameworkSpecificModel):
    """SparkPipelineModel class for estimators from the pyspark framework.

    Attributes
    ----------
    algorithm: str
        The algorithm of the model.
    artifact_dir: str
        Artifact directory to store the files needed for deployment.
    auth: Dict
        Default authentication is set using the `ads.set_auth` API. To override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create
        an authentication signer to instantiate an IdentityClient object.
    estimator: Callable
        A trained pyspark estimator/model using pyspark.
    framework: str
        "spark", the framework name of the model.
    hyperparameter: dict
        The hyperparameters of the estimator.
    metadata_custom: ModelCustomMetadata
        The model custom metadata.
    metadata_provenance: ModelProvenanceMetadata
        The model provenance metadata.
    metadata_taxonomy: ModelTaxonomyMetadata
        The model taxonomy metadata.
    model_artifact: ModelArtifact
        This is built by calling prepare.
        A ModelDeployment instance.
    model_file_name: str
        Name of the serialized model.
    model_id: str
        The model ID.
    properties: ModelProperties
        ModelProperties object required to save and deploy model.
        For more details, check https://accelerated-data-science.readthedocs.io/en/latest/ads.model.html#module-ads.model.model_properties.
    runtime_info: RuntimeInfo
        A RuntimeInfo instance.
    schema_input: Schema
        Schema describes the structure of the input data.
    schema_output: Schema
        Schema describes the structure of the output data.
    serialize: bool
        Whether to serialize the model to pkl file by default. If False, you need to serialize the model manually,
        save it under artifact_dir and update the score.py manually.
    version: str
        The framework version of the model.

    Methods
    -------
    delete_deployment(...)
        Deletes the current model deployment.
    deploy(..., **kwargs)
        Deploys a model.
    from_model_artifact(uri, model_file_name, artifact_dir, ..., **kwargs)
        Loads model from the specified folder, or zip/tar archive.
    from_model_catalog(model_id, model_file_name, artifact_dir, ..., **kwargs)
        Loads model from model catalog.
    introspect(...)
        Runs model introspection.
    predict(data, ...)
        Returns prediction of input data run against the model deployment endpoint.
    prepare(..., **kwargs)
        Prepare and save the score.py, serialized model and runtime.yaml file.
    reload(...)
        Reloads the model artifact files: `score.py` and the `runtime.yaml`.
    save(..., **kwargs)
        Saves model artifacts to the model catalog.
    summary_status(...)
        Gets a summary table of the current status.
    verify(data, ...)
        Tests if deployment works in local environment.

    Examples
    --------
    >>> import tempfile
    >>> from ads.model.framework.spark_model import SparkPipelineModel
    >>> from pyspark.ml.linalg import Vectors
    >>> from pyspark.ml.classification import LogisticRegression

    >>> training = spark.createDataFrame([
    >>>     (1.0, Vectors.dense([0.0, 1.1, 0.1])),
    >>>     (0.0, Vectors.dense([2.0, 1.0, -1.0])),
    >>>     (0.0, Vectors.dense([2.0, 1.3, 1.0])),
    >>>     (1.0, Vectors.dense([0.0, 1.2, -0.5]))], ["label", "features"])
    >>> lr_estimator = LogisticRegression(maxIter=10, regParam=0.001)
    >>> pipeline = Pipeline(stages=[lr_estimator])
    >>> pipeline_model = pipeline.fit(training)

    >>> spark_model = SparkPipelineModel(estimator=pipeline_model, artifact_dir=tempfile.mkdtemp())
    >>> spark_model.prepare(inference_conda_env="dataexpl_p37_cpu_v3")
    >>> spark_model.verify(training)
    >>> spark_model.save()
    >>> model_deployment = spark_model.deploy()
    >>> spark_model.predict(training)
    >>> spark_model.delete_deployment()
    """

    _PREFIX = "spark"
    model_input_serializer_type = SparkModelInputSerializerType
    model_save_serializer_type = SparkModelSerializerType

    @runtime_dependency(
        module="pyspark",
        short_name="ml",
        object="ml",
        install_from=OptionalDependency.SPARK,
    )
    def __init__(
        self,
        estimator: Callable,
        artifact_dir: Optional[str] = None,
        properties: Optional[ModelProperties] = None,
        auth: Dict = None,
        model_save_serializer: Optional[SERDE] = model_save_serializer_type.SPARK,
        model_input_serializer: Optional[SERDE] = model_input_serializer_type.SPARK,
        **kwargs,
    ):
        """
        Initiates a SparkPipelineModel instance.

        Parameters
        ----------
        estimator: Callable
            SparkPipelineModel
        artifact_dir: str
            The URI for the generated artifact, which can be local path or OCI object storage URI.
        properties: (ModelProperties, optional). Defaults to None.
            ModelProperties object required to save and deploy model.
        auth :(Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        model_save_serializer: (SERDE or str, optional). Defaults to None.
            Instance of ads.model.SERDE. Used for serialize/deserialize model.
        model_input_serializer: (SERDE, optional). Defaults to `ads.model.serde.model_input.SparkModelInputSERDE`.
            Instance of ads.model.SERDE. Used for serialize/deserialize data.

        Returns
        -------
        SparkPipelineModel
            SparkPipelineModel instance.


        Examples
        --------
        >>> import tempfile
        >>> from ads.model.framework.spark_model import SparkPipelineModel
        >>> from pyspark.ml.linalg import Vectors
        >>> from pyspark.ml.classification import LogisticRegression
        >>> from pyspark.ml import Pipeline

        >>> training = spark.createDataFrame([
        >>>     (1.0, Vectors.dense([0.0, 1.1, 0.1])),
        >>>     (0.0, Vectors.dense([2.0, 1.0, -1.0])),
        >>>     (0.0, Vectors.dense([2.0, 1.3, 1.0])),
        >>>     (1.0, Vectors.dense([0.0, 1.2, -0.5]))], ["label", "features"])
        >>> lr_estimator = LogisticRegression(maxIter=10, regParam=0.001)
        >>> pipeline = Pipeline(stages=[lr_estimator])
        >>> pipeline_model = pipeline.fit(training)

        >>> spark_model = SparkPipelineModel(estimator=pipeline_model, artifact_dir=tempfile.mkdtemp())
        >>> spark_model.prepare(inference_conda_env="pyspark30_p37_cpu_v5")
        >>> spark_model.verify(training)
        >>> spark_model.save()
        >>> model_deployment = spark_model.deploy()
        >>> spark_model.predict(training)
        >>> spark_model.delete_deployment()
        """
        if not (type(estimator) in [ml.PipelineModel]):
            raise TypeError(
                f"{str(type(estimator))} is not supported in `SparkPipelineModel`s."
            )
        super().__init__(
            estimator=estimator,
            artifact_dir=artifact_dir,
            properties=properties,
            auth=auth,
            model_save_serializer=model_save_serializer,
            model_input_serializer=model_input_serializer,
            **kwargs,
        )

        self._extractor = SparkExtractor(estimator)
        self.framework = self._extractor.framework
        self.algorithm = self._extractor.algorithm
        self.version = self._extractor.version
        self.hyperparameter = self._extractor.hyperparameter

    @staticmethod
    def _handle_model_file_name(
        as_onnx: bool, model_file_name: Optional[str] = DEFAULT_MODEL_FOLDER_NAME
    ):
        """
        Process folder name for saving model.

        Parameters
        ----------
        as_onnx: bool
            To convert to onnx format
        model_file_name: Optional[str]
            File name for saving model. Default value is `model`.

        Returns
        -------
        str
            Processed file name. (Folder in the case of spark serialization)
        """
        if as_onnx:
            raise NotImplementedError(
                "The Spark to Onnx Conversion is not supported because it is unstable. Please set as_onnx to False (default) to perform a spark model serialization"
            )
        if not model_file_name:
            return DEFAULT_MODEL_FOLDER_NAME
        return model_file_name

    def serialize_model(
        self,
        as_onnx: bool = False,
        X_sample: Optional[
            Union[
                Dict,
                str,
                List,
                np.ndarray,
                pd.core.series.Series,
                pd.core.frame.DataFrame,
                "pyspark.sql.DataFrame",
                "pyspark.pandas.DataFrame",
            ]
        ] = None,
        force_overwrite: bool = False,
        **kwargs,
    ) -> None:
        """
        Serialize and save pyspark model using spark serialization.

        Parameters
        ----------
        force_overwrite: (bool, optional). Defaults to False.
            If set as True, overwrite serialized model if exists.

        Returns
        -------
        None
        """
        if as_onnx:
            raise NotImplementedError(
                "The Spark to Onnx Conversion is not supported because it is unstable. Please set as_onnx to False (default) to perform a spark model serialization"
            )

        super().serialize_model(
            as_onnx=as_onnx,
            force_overwrite=force_overwrite,
            X_sample=X_sample,
            **kwargs,
        )

    @runtime_dependency(
        module="pyspark",
        short_name="sql",
        object="sql",
        install_from=OptionalDependency.SPARK,
    )
    def _prepare_data_for_schema(
        self,
        X_sample: Union[List, Tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        y_sample: Union[List, Tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        **kwargs,
    ):
        """Generate Spark Schema and format Spark DataFrame as Pandas for ADS Schema Generation"""
        input_schema_path = os.path.join(
            self.artifact_dir,
            self.model_file_name + SPARK_DATAFRAME_SCHEMA_PATH,
        )
        (
            X_sample,
            data_type,
            schema,
        ) = self.get_data_serializer()._serialize_via_spark(X_sample)
        if not schema:
            raise TypeError(
                f"Data type: {data_type} unsupported. Please use `pyspark.sql.DataFrame`, `pyspark.pandas.DataFrame`, or `pandas.DataFrame`."
            )
        storage_options = kwargs.get("auth", {})
        with fsspec.open(
            input_schema_path,
            mode="w",
            **(storage_options),
        ) as f:
            f.write(schema.json())

        if isinstance(X_sample, sql.DataFrame):
            X_sample = X_sample.toPandas()

        return X_sample, y_sample
