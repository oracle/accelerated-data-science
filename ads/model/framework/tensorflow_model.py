#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ads.common import logger
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.model.extractor.tensorflow_extractor import TensorflowExtractor
from ads.model.generic_model import FrameworkSpecificModel
from ads.model.model_properties import ModelProperties
from ads.model.serde.model_serializer import TensorflowModelSerializerType
from ads.model.common.utils import DEPRECATE_AS_ONNX_WARNING
from ads.model.serde.common import SERDE


class TensorFlowModel(FrameworkSpecificModel):
    """TensorFlowModel class for estimators from Tensorflow framework.

    Attributes
    ----------
    algorithm: str
        The algorithm of the model.
    artifact_dir: str
        Directory for generate artifact.
    auth: Dict
        Default authentication is set using the `ads.set_auth` API. To override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create
        an authentication signer to instantiate an IdentityClient object.
    estimator: Callable
        A trained tensorflow estimator/model using Tensorflow.
    framework: str
        "tensorflow", the framework name of the model.
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
    model_deployment: ModelDeployment
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
    >>> from ads.model.framework.tensorflow_model import TensorFlowModel
    >>> import tempfile
    >>> import tensorflow as tf

    >>> mnist = tf.keras.datasets.mnist
    >>> (x_train, y_train), (x_test, y_test) = mnist.load_data()
    >>> x_train, x_test = x_train / 255.0, x_test / 255.0

    >>> tf_estimator = tf.keras.models.Sequential(
    ...                [
    ...                    tf.keras.layers.Flatten(input_shape=(28, 28)),
    ...                    tf.keras.layers.Dense(128, activation="relu"),
    ...                    tf.keras.layers.Dropout(0.2),
    ...                    tf.keras.layers.Dense(10),
    ...                ]
    ...            )
    >>> loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    >>> tf_estimator.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    >>> tf_estimator.fit(x_train, y_train, epochs=1)

    >>> tf_model = TensorFlowModel(estimator=tf_estimator,
    ... artifact_dir=tempfile.mkdtemp())
    >>> inference_conda_env = "generalml_p37_cpu_v1"

    >>> tf_model.prepare(inference_conda_env="generalml_p37_cpu_v1", force_overwrite=True)
    >>> tf_model.verify(x_test[:1])
    >>> tf_model.save()
    >>> model_deployment = tf_model.deploy(wait_for_completion=False)
    >>> tf_model.predict(x_test[:1])
    """

    _PREFIX = "tensorflow"
    model_save_serializer_type = TensorflowModelSerializerType

    @runtime_dependency(
        module="tensorflow",
        short_name="tf",
        install_from=OptionalDependency.TENSORFLOW,
    )
    def __init__(
        self,
        estimator: callable,
        artifact_dir: Optional[str] = None,
        properties: Optional[ModelProperties] = None,
        auth: Dict = None,
        model_save_serializer: Optional[SERDE] = model_save_serializer_type.TENSORFLOW,
        model_input_serializer: Optional[SERDE] = None,
        **kwargs,
    ):
        """
        Initiates a TensorFlowModel instance.

        Parameters
        ----------
        estimator: callable
            Any model object generated by tensorflow framework
        artifact_dir: str
            Directory for generate artifact.
        properties: (ModelProperties, optional). Defaults to None.
            ModelProperties object required to save and deploy model.
        auth :(Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        model_save_serializer: (SERDE or str, optional). Defaults to None.
            Instance of ads.model.SERDE. Used for serialize/deserialize model.
        model_input_serializer: (SERDE, optional). Defaults to None.
            Instance of ads.model.SERDE. Used for serialize/deserialize data.

        Returns
        -------
        TensorFlowModel
            TensorFlowModel instance.
        """
        super().__init__(
            estimator=estimator,
            artifact_dir=artifact_dir,
            properties=properties,
            auth=auth,
            model_save_serializer=model_save_serializer,
            model_input_serializer=model_input_serializer,
            **kwargs,
        )
        self._extractor = TensorflowExtractor(estimator)
        self.framework = self._extractor.framework
        self.algorithm = self._extractor.algorithm
        self.version = self._extractor.version
        self.hyperparameter = self._extractor.hyperparameter

        self.version = tf.version.VERSION

    def serialize_model(
        self,
        as_onnx: bool = False,
        X_sample: Optional[
            Union[
                Dict,
                str,
                List,
                Tuple,
                np.ndarray,
                pd.core.series.Series,
                pd.core.frame.DataFrame,
            ]
        ] = None,
        force_overwrite: bool = False,
        **kwargs,
    ) -> None:
        """
        Serialize and save Tensorflow model using ONNX or model specific method.

        Parameters
        ----------
        as_onnx: (bool, optional). Defaults to False.
            If set as True, convert into ONNX model.
        X_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]. Defaults to None.
            A sample of input data that will be used to generate input schema and detect input_signature.
        force_overwrite: (bool, optional). Defaults to False.
            If set as True, overwrite serialized model if exists.
        **kwargs: optional params used to serialize tensorflow model to onnx,
        including the following:
            input_signature: a tuple or a list of tf.TensorSpec objects). default to None.
            Define the shape/dtype of the input so that model(input_signature) is a valid invocation of the model.
            opset_version: int. Defaults to None. Used for the ONNX model.

        Returns
        -------
        None
            Nothing.
        """

        if as_onnx:
            logger.warning(
                "This approach supports converting tensorflow.keras models to "
                "onnx format. If the defined model includes other tensorflow "
                "modules (e.g., tensorflow.function), please use GenericModel instead."
            )
            logger.warning(DEPRECATE_AS_ONNX_WARNING)
            self.set_model_save_serializer(self.model_save_serializer_type.ONNX)

        super().serialize_model(
            as_onnx=as_onnx,
            force_overwrite=force_overwrite,
            X_sample=X_sample,
            **kwargs,
        )

    @runtime_dependency(
        module="tensorflow",
        short_name="tf",
        install_from=OptionalDependency.TENSORFLOW,
    )
    def _to_tensor(self, data):
        data = tf.convert_to_tensor(data)
        return data
