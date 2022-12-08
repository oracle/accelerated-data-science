#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ads.common import logger
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.model.extractor.tensorflow_extractor import TensorflowExtractor
from ads.common.data_serializer import InputDataSerializer
from ads.model.generic_model import FrameworkSpecificModel
from ads.model.model_properties import ModelProperties

ONNX_MODEL_FILE_NAME = "model.onnx"
TENSORFLOW_MODEL_FILE_NAME = "model.h5"


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
    ds_client: DataScienceClient
        The data science client used by model deployment.
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

    @runtime_dependency(
        module="tensorflow",
        short_name="tf",
        install_from=OptionalDependency.TENSORFLOW,
    )
    def __init__(
        self,
        estimator: callable,
        artifact_dir: str,
        properties: Optional[ModelProperties] = None,
        auth: Dict = None,
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
            **kwargs,
        )
        self._extractor = TensorflowExtractor(estimator)
        self.framework = self._extractor.framework
        self.algorithm = self._extractor.algorithm
        self.version = self._extractor.version
        self.hyperparameter = self._extractor.hyperparameter

        self.version = tf.version.VERSION

    def _handle_model_file_name(self, as_onnx: bool, model_file_name: str) -> str:
        """
        Process file name for saving model.
        For ONNX model file name must be ending with ".onnx".
        For joblib model file name must be ending with ".joblib".
        If not specified, use "model.onnx" for ONNX model and "model.joblib" for joblib model.

        Parameters
        ----------
        as_onnx: bool
            If set as True, convert into ONNX model.
        model_file_name: str
            File name for saving model.

        Returns
        -------
        str
            Processed file name.
        """
        if not model_file_name:
            return ONNX_MODEL_FILE_NAME if as_onnx else TENSORFLOW_MODEL_FILE_NAME
        if as_onnx:
            if model_file_name and not model_file_name.endswith(".onnx"):
                raise ValueError(
                    "`model_file_name` has to be ending with `.onnx` for onnx format."
                )
        else:
            if model_file_name and not model_file_name.endswith(".h5"):
                raise ValueError(
                    "`model_file_name` has to be ending with `.h5` "
                    "for Tensorflow model format."
                )
        return model_file_name

    @runtime_dependency(
        module="tensorflow",
        short_name="tf",
        install_from=OptionalDependency.TENSORFLOW,
    )
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

        model_path = os.path.join(self.artifact_dir, self.model_file_name)

        if os.path.exists(model_path) and not force_overwrite:
            raise ValueError(
                f"The {model_path} already exists, set force_overwrite to True if you wish to overwrite."
            )

        os.makedirs(self.artifact_dir, exist_ok=True)

        if as_onnx:
            logger.warning(
                "This approach supports converting tensorflow.keras models to "
                "onnx format. If the defined model includes other tensorflow "
                "modules (e.g., tensorflow.function), please use GenericModel instead."
            )
            opset_version = kwargs.get("opset_version", None)
            input_signature = kwargs.get("input_signature", None)

            self.to_onnx(
                path=model_path,
                input_signature=input_signature,
                X_sample=X_sample,
                opset_version=opset_version,
            )

        else:
            self.estimator.save(model_path)

    @runtime_dependency(module="tf2onnx", install_from=OptionalDependency.ONNX)
    @runtime_dependency(
        module="tensorflow",
        short_name="tf",
        install_from=OptionalDependency.TENSORFLOW,
    )
    def to_onnx(
        self,
        path: str = None,
        input_signature=None,
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
        opset_version=None,
    ):
        """
        Exports the given Tensorflow model into ONNX format.

        Parameters
        ----------
        path: str, default to None
            Path to save the serialized model.
        input_signature: a tuple or a list of tf.TensorSpec objects. default to None.
            Define the shape/dtype of the input so that model(input_signature) is a valid invocation of the model.
        X_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]. Defaults to None.
            A sample of input data that will be used to generate input schema and detect input_signature.
        opset_version: int. Defaults to None.
            The opset to be used for the ONNX model.

        Returns
        -------
        None
            Nothing

        Raises
        ------
        ValueError
            if path is not provided
        """

        if not path:
            raise ValueError(
                "The parameter `path` must be provided to save the model file."
            )
        if input_signature is None:
            if hasattr(self.estimator, "input_shape"):
                if not isinstance(self.estimator.input, list):
                    # single input
                    detected_input_signature = (
                        tf.TensorSpec(
                            self.estimator.input_shape,
                            dtype=self.estimator.input.dtype,
                            name="input",
                        ),
                    )
                else:
                    # multiple input
                    detected_input_signature = []
                    for i in range(len(self.estimator.input)):
                        detected_input_signature.append(
                            tf.TensorSpec(
                                self.estimator.input_shape[i],
                                dtype=self.estimator.input[i].dtype,
                            )
                        )

            elif X_sample is not None and hasattr(X_sample, "shape"):
                logger.warning(
                    "Since `input_signature` is not provided, `input_signature` is "
                    "detected from `X_sample` to export tensorflow model as "
                    "onnx."
                )
                X_sample_shape = list(X_sample.shape)
                X_sample_shape[0] = None
                detected_input_signature = (
                    tf.TensorSpec(X_sample_shape, dtype=X_sample.dtype, name="input"),
                )
            else:
                raise ValueError(
                    "The parameter `input_signature` must be provided to export "
                    "tensorflow model as onnx."
                )
            try:
                tf2onnx.convert.from_keras(
                    self.estimator,
                    input_signature=detected_input_signature,
                    opset=opset_version,
                    output_path=path,
                )
            except:
                raise ValueError(
                    "`input_signature` can not be autodetected. The parameter `input_signature` must be provided to export "
                    "tensorflow model as onnx."
                )

        else:
            tf2onnx.convert.from_keras(
                self.estimator,
                input_signature=input_signature,
                opset=opset_version,
                output_path=path,
            )

    @runtime_dependency(
        module="tensorflow",
        short_name="tf",
        install_from=OptionalDependency.TENSORFLOW,
    )
    def get_data_serializer(
        self,
        data: Union[
            Dict,
            str,
            List,
            np.ndarray,
            pd.core.series.Series,
            pd.core.frame.DataFrame,
            "tf.Tensor",
        ],
        data_type: str = None,
    ):
        """Returns serializable input data.

        Parameters
        ----------
        data: Union[Dict, str, list, numpy.ndarray, pd.core.series.Series,
        pd.core.frame.DataFrame, tf.Tensor]
            Data expected by the model deployment predict API.
        data_type: str
            Type of the data.

        Returns
        -------
        InputDataSerializer
            A class containing serialized input data and original data type information.

        Raises
        ------
        TypeError
            if provided data type is not supported.
        """
        try:
            data_type = data_type or type(data)
            if data_type == "image":
                data = tf.convert_to_tensor(data)
                data_type = str(type(data))
            if isinstance(data, tf.Tensor):
                data = data.numpy()
            return InputDataSerializer(data, data_type=data_type)
        except:
            raise TypeError(
                "The supported data types are Dict, str, list, "
                "numpy.ndarray, pd.core.series.Series, "
                "pd.core.frame.DataFrame, tf.Tensor, bytes. Please "
                "convert to the supported data types first. "
            )
