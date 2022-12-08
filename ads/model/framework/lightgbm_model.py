#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ads.common import logger
from ads.model.extractor.lightgbm_extractor import LightgbmExtractor
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common.data_serializer import InputDataSerializer
from ads.model.generic_model import (
    FrameworkSpecificModel,
    DEFAULT_ONNX_FORMAT_MODEL_FILE_NAME,
    DEFAULT_JOBLIB_FORMAT_MODEL_FILE_NAME,
    DEFAULT_TXT_FORMAT_MODEL_FILE_NAME,
)
from ads.model.model_properties import ModelProperties
from joblib import dump


class LightGBMModel(FrameworkSpecificModel):
    """LightGBMModel class for estimators from Lightgbm framework.

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
    ds_client: DataScienceClient
        The data science client used by model deployment.
    estimator: Callable
        A trained lightgbm estimator/model using Lightgbm.
    framework: str
        "lightgbm", the framework name of the model.
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
    >>> import lightgbm as lgb
    >>> import tempfile
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import load_iris
    >>> from ads.model.framework.lightgbm_model import LightGBMModel

    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target

    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    >>> train = lgb.Dataset(X_train, label=y_train)
    >>> param = {
    ...        'objective': 'multiclass', 'num_class': 3,
    ...        }
    >>> lightgbm_estimator = lgb.train(param, train)

    >>> lightgbm_model = LightGBMModel(estimator=lightgbm_estimator,
    ... artifact_dir=tempfile.mkdtemp())

    >>> lightgbm_model.prepare(inference_conda_env="generalml_p37_cpu_v1", force_overwrite=True)
    >>> lightgbm_model.reload()
    >>> lightgbm_model.verify(X_test)
    >>> lightgbm_model.save()
    >>> model_deployment = lightgbm_model.deploy(wait_for_completion=False)
    >>> lightgbm_model.predict(X_test)
    """

    _PREFIX = "lightgbm"

    def __init__(
        self,
        estimator: Callable,
        artifact_dir: str,
        properties: Optional[ModelProperties] = None,
        auth: Dict = None,
        **kwargs,
    ):
        """
        Initiates a LightGBMModel instance. This class wraps the Lightgbm model as estimator.
        It's primary purpose is to hold the trained model and do serialization.

        Parameters
        ----------
        estimator:
            any model object generated by Lightgbm framework
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
        LightGBMModel
            LightGBMModel instance.

        Raises
        ------
        TypeError: If the input model is not a Lightgbm model or not supported for serialization.


        Examples
        --------
        >>> import lightgbm as lgb
        >>> import tempfile
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.datasets import load_iris
        >>> from ads.model.framework.lightgbm_model import LightGBMModel
        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        >>> train = lgb.Dataset(X_train, label=y_train)
        >>> param = {
        ... 'objective': 'multiclass', 'num_class': 3,
        ... }
        >>> lightgbm_estimator = lgb.train(param, train)
        >>> lightgbm_model = LightGBMModel(estimator=lightgbm_estimator, artifact_dir=tempfile.mkdtemp())
        >>> lightgbm_model.prepare(inference_conda_env="generalml_p37_cpu_v1")
        >>> lightgbm_model.verify(X_test)
        >>> lightgbm_model.save()
        >>> model_deployment = lightgbm_model.deploy()
        >>> lightgbm_model.predict(X_test)
        >>> lightgbm_model.delete_deployment()
        """
        model_type = str(type(estimator))
        if not (
            model_type.startswith("<class 'lightgbm.basic.")
            or model_type.startswith("<class 'lightgbm.sklearn.")
            or model_type.startswith("<class 'onnxruntime.")
        ):
            raise TypeError(f"{model_type} is not supported in LightGBMModel.")
        super().__init__(
            estimator=estimator,
            artifact_dir=artifact_dir,
            properties=properties,
            auth=auth,
            **kwargs,
        )
        self._extractor = LightgbmExtractor(estimator)
        self.framework = self._extractor.framework
        self.algorithm = self._extractor.algorithm
        self.version = self._extractor.version
        self.hyperparameter = self._extractor.hyperparameter

    def _handle_model_file_name(self, as_onnx: bool, model_file_name: str):
        """
        Process file name for saving model.
        For ONNX model file name must be ending with ".onnx".
        For joblib model file name must be ending with ".joblib".
        For TXT model file name must be ending with ".txt".
        If not specified, use "model.onnx" for ONNX model, "model.txt" for TXT model and "model.joblib" for joblib model.

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

        Raises
        ------
        ValueError: If the input model_file_name does not corresponding to serialize format.
        """
        is_sklearn = str(type(self.estimator)).startswith("<class 'lightgbm.sklearn.")
        if not model_file_name:
            if as_onnx:
                return DEFAULT_ONNX_FORMAT_MODEL_FILE_NAME
            elif is_sklearn:
                return DEFAULT_JOBLIB_FORMAT_MODEL_FILE_NAME
            else:
                return DEFAULT_TXT_FORMAT_MODEL_FILE_NAME
        if as_onnx:
            if model_file_name and not model_file_name.endswith(".onnx"):
                raise ValueError(
                    "`model_file_name` has to be ending with `.onnx` for onnx format."
                )
        else:
            if (
                is_sklearn
                and model_file_name
                and not model_file_name.endswith(".joblib")
            ):
                raise ValueError(
                    "`model_file_name` has to be ending with `.joblib` for joblib format."
                )
            if (
                not is_sklearn
                and model_file_name
                and not model_file_name.endswith(".txt")
            ):
                raise ValueError(
                    "`model_file_name` has to be ending with `.txt` for TXT format."
                )
        return model_file_name

    def serialize_model(
        self,
        as_onnx: bool = False,
        initial_types: List[Tuple] = None,
        force_overwrite: bool = False,
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
        **kwargs: Dict,
    ):
        """
        Serialize and save Lightgbm model using ONNX or model specific method.

        Parameters
        ----------
        artifact_dir: str
            Directory for generate artifact.
        as_onnx: (boolean, optional). Defaults to False.
            If set as True, provide initial_types or X_sample to convert into ONNX.
        initial_types: (List[Tuple], optional). Defaults to None.
            Each element is a tuple of a variable name and a type.
        force_overwrite: (boolean, optional). Defaults to False.
            If set as True, overwrite serialized model if exists.
        X_sample: Union[Dict, str, List, np.ndarray, pd.core.series.Series, pd.core.frame.DataFrame,]. Defaults to None.
            Contains model inputs such that model(X_sample) is a valid invocation of the model.
            Used to generate initial_types.

        Returns
        -------
        None
            Nothing.
        """
        model_path = os.path.join(self.artifact_dir, self.model_file_name)
        if os.path.exists(model_path) and not force_overwrite:
            raise ValueError(
                "Model file already exists and will not be overwritten. "
                "Set `force_overwrite` to True if you wish to overwrite."
            )
        else:
            if not os.path.exists(self.artifact_dir):
                os.makedirs(self.artifact_dir)
            if as_onnx:
                onx = self.to_onnx(
                    initial_types=initial_types, X_sample=X_sample, **kwargs
                )
                with open(model_path, "wb") as f:
                    f.write(onx.SerializeToString())
            else:
                if str(type(self.estimator)).startswith("<class 'lightgbm.basic."):
                    self.estimator.save_model(model_path)
                elif str(type(self.estimator)).startswith("<class 'lightgbm.sklearn"):
                    dump(self.estimator, model_path)

    @runtime_dependency(
        module="skl2onnx.common.data_types",
        object="FloatTensorType",
        install_from=OptionalDependency.ONNX,
    )
    @runtime_dependency(
        module="onnxmltools.convert",
        object="convert_lightgbm",
        install_from=OptionalDependency.ONNX,
    )
    def to_onnx(
        self,
        initial_types: List[Tuple] = None,
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
        **kwargs,
    ):
        """
        Produces an equivalent ONNX model of the given Lightgbm model.

        Parameters
        ----------
        initial_types: (List[Tuple], optional). Defaults to None.
            Each element is a tuple of a variable name and a type.
        X_sample: Union[Dict, str, List, np.ndarray, pd.core.series.Series, pd.core.frame.DataFrame,]. Defaults to None.
            Contains model inputs such that model(X_sample) is a valid invocation of the model.
            Used to generate initial_types.

        Returns
        ------
            An ONNX model (type: ModelProto) which is equivalent to the input Lightgbm model.
        """
        auto_generated_initial_types = None
        if not initial_types:
            auto_generated_initial_types = self.generate_initial_types(X_sample)
            try:
                return convert_lightgbm(
                    self.estimator,
                    initial_types=auto_generated_initial_types,
                    target_opset=kwargs.pop("target_opset", None),
                    **kwargs,
                )
            except:
                raise ValueError(
                    "`initial_types` can not be detected. Please directly pass initial_types."
                )
        else:
            return convert_lightgbm(
                self.estimator,
                initial_types=initial_types,
                target_opset=kwargs.pop("target_opset", None),
                **kwargs,
            )

    @runtime_dependency(
        module="skl2onnx.common.data_types",
        object="FloatTensorType",
        install_from=OptionalDependency.ONNX,
    )
    def generate_initial_types(self, X_sample: Any) -> List:
        """Auto generate intial types.

        Parameters
        ----------
        X_sample: (Any)
            Train data.

        Returns
        -------
        List
            Initial types.
        """
        if X_sample is not None and hasattr(X_sample, "shape"):
            auto_generated_initial_types = [
                ("input", FloatTensorType([None, X_sample.shape[1]]))
            ]
        elif hasattr(self.estimator, "num_feature"):
            n_cols = self.estimator.num_feature()
            auto_generated_initial_types = [("input", FloatTensorType([None, n_cols]))]
        elif hasattr(self.estimator, "n_features_in_"):
            n_cols = self.estimator.n_features_in_
            auto_generated_initial_types = [("input", FloatTensorType([None, n_cols]))]
        else:
            raise ValueError(
                "`initial_types` can not be detected. Please directly pass initial_types."
            )
        return auto_generated_initial_types
