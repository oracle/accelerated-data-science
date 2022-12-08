#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ads.common import logger
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common.data_serializer import InputDataSerializer
from ads.model.extractor.xgboost_extractor import XgboostExtractor
from ads.model.generic_model import (
    FrameworkSpecificModel,
    DEFAULT_ONNX_FORMAT_MODEL_FILE_NAME,
    DEFAULT_JSON_FORMAT_MODEL_FILE_NAME,
)
from ads.model.model_properties import ModelProperties


class XGBoostModel(FrameworkSpecificModel):
    """XGBoostModel class for estimators from xgboost framework.

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
        A trained xgboost estimator/model using Xgboost.
    framework: str
        "xgboost", the framework name of the model.
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
    >>> import xgboost as xgb
    >>> import tempfile
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import load_iris
    >>> from ads.model.framework.xgboost_model import XGBoostModel

    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    >>> xgboost_estimator = xgb.XGBClassifier()
    >>> xgboost_estimator.fit(X_train, y_train)

    >>> xgboost_model = XGBoostModel(estimator=xgboost_estimator, artifact_dir=tmp_model_dir)
    >>> xgboost_model.prepare(inference_conda_env="generalml_p37_cpu_v1", force_overwrite=True)
    >>> xgboost_model.reload()
    >>> xgboost_model.verify(X_test)
    >>> xgboost_model.save()
    >>> model_deployment = xgboost_model.deploy(wait_for_completion=False)
    >>> xgboost_model.predict(X_test)
    """

    _PREFIX = "xgboost"

    @runtime_dependency(module="xgboost", install_from=OptionalDependency.BOOSTED)
    def __init__(
        self,
        estimator: callable,
        artifact_dir: str,
        properties: Optional[ModelProperties] = None,
        auth: Dict = None,
        **kwargs,
    ):
        """
        Initiates a XGBoostModel instance. This class wraps the XGBoost model as estimator.
        It's primary purpose is to hold the trained model and do serialization.

        Parameters
        ----------
        estimator:
            XGBoostModel
        artifact_dir: str
            artifact directory to store the files needed for deployment.
        properties: (ModelProperties, optional). Defaults to None.
            ModelProperties object required to save and deploy model.
        auth :(Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        XGBoostModel
            XGBoostModel instance.


        Examples
        --------
        >>> import xgboost as xgb
        >>> import tempfile
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.datasets import load_iris
        >>> from ads.model.framework.xgboost_model import XGBoostModel

        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target

        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        >>> train = xgb.DMatrix(X_train, y_train)
        >>> test = xgb.DMatrix(X_test, y_test)
        >>> xgboost_estimator = XGBClassifier()
        >>> xgboost_estimator.fit(X_train, y_train)
        >>> xgboost_model = XGBoostModel(estimator=xgboost_estimator, artifact_dir=tempfile.mkdtemp())
        >>> xgboost_model.prepare(inference_conda_env="generalml_p37_cpu_v1")
        >>> xgboost_model.verify(X_test)
        >>> xgboost_model.save()
        >>> model_deployment = xgboost_model.deploy()
        >>> xgboost_model.predict(X_test)
        >>> xgboost_model.delete_deployment()
        """
        if not (
            str(type(estimator)).startswith("<class 'xgboost.")
            or str(type(estimator)).startswith("<class 'onnxruntime.")
        ):
            raise TypeError(f"{str(type(estimator))} is not supported in XGBoostModel.")
        super().__init__(
            estimator=estimator,
            artifact_dir=artifact_dir,
            properties=properties,
            auth=auth,
            **kwargs,
        )
        self._extractor = XgboostExtractor(estimator)
        self.framework = self._extractor.framework
        self.algorithm = self._extractor.algorithm
        self.version = self._extractor.version
        self.hyperparameter = self._extractor.hyperparameter

    @staticmethod
    def _handle_model_file_name(as_onnx: bool, model_file_name: str):
        """
        Process file name for saving model.
        For ONNX model file name must be ending with ".onnx".
        For JSON model file name must be ending with ".json".
        If not specified, use "model.onnx" for ONNX model and "model.json" for JSON model.

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
        if not model_file_name:
            return (
                DEFAULT_ONNX_FORMAT_MODEL_FILE_NAME
                if as_onnx
                else DEFAULT_JSON_FORMAT_MODEL_FILE_NAME
            )
        if as_onnx:
            if model_file_name and not model_file_name.endswith(".onnx"):
                raise ValueError(
                    "`model_file_name` has to be ending with `.onnx` for onnx format."
                )
        else:
            if model_file_name and not model_file_name.endswith(".json"):
                raise ValueError(
                    "`model_file_name` has to be ending with `.json` for JSON format."
                )
        return model_file_name

    @runtime_dependency(module="xgboost", install_from=OptionalDependency.BOOSTED)
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
        **kwargs,
    ):
        """
        Serialize and save Xgboost model using ONNX or model specific method.

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
        self.model_file_name = self._handle_model_file_name(
            as_onnx, self.model_file_name
        )
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
                self.estimator.save_model(model_path)

    @runtime_dependency(module="onnx", install_from=OptionalDependency.ONNX)
    @runtime_dependency(module="xgboost", install_from=OptionalDependency.BOOSTED)
    @runtime_dependency(
        module="skl2onnx",
        object="convert_sklearn",
        install_from=OptionalDependency.ONNX,
    )
    @runtime_dependency(
        module="skl2onnx",
        object="update_registered_converter",
        install_from=OptionalDependency.ONNX,
    )
    @runtime_dependency(
        module="skl2onnx.common.data_types",
        object="FloatTensorType",
        install_from=OptionalDependency.ONNX,
    )
    @runtime_dependency(
        module="skl2onnx.common.shape_calculator",
        object="calculate_linear_classifier_output_shapes",
        install_from=OptionalDependency.ONNX,
    )
    @runtime_dependency(
        module="skl2onnx.common.shape_calculator",
        object="calculate_linear_regressor_output_shapes",
        install_from=OptionalDependency.ONNX,
    )
    @runtime_dependency(module="onnxmltools", install_from=OptionalDependency.ONNX)
    @runtime_dependency(
        module="onnxmltools.convert.xgboost.operator_converters.XGBoost",
        object="convert_xgboost",
        install_from=OptionalDependency.ONNX,
    )
    def to_onnx(
        self,
        initial_types: List[Tuple] = None,
        X_sample: Union[list, tuple, pd.DataFrame, pd.Series, np.ndarray] = None,
        **kwargs,
    ):
        """
        Produces an equivalent ONNX model of the given Xgboost model.

        Parameters
        ----------
        initial_types: (List[Tuple], optional). Defaults to None.
            Each element is a tuple of a variable name and a type.
        X_sample: Union[Dict, str, List, np.ndarray, pd.core.series.Series, pd.core.frame.DataFrame,]. Defaults to None.
            Contains model inputs such that model(X_sample) is a valid invocation of the model.
            Used to generate initial_types.

        Returns
        -------
        onnx.onnx_ml_pb2.ModelProto
            An ONNX model (type: ModelProto) which is equivalent to the input xgboost model.
        """
        auto_generated_initial_types = None
        if not initial_types:
            auto_generated_initial_types = self.generate_initial_types(X_sample)

        model_types = []
        if str(type(self.estimator)).startswith("<class 'xgboost.sklearn."):
            model_types.append(type(self.estimator))

        if model_types:
            if xgboost.sklearn.XGBClassifier in model_types:
                update_registered_converter(
                    xgboost.XGBClassifier,
                    "XGBoostXGBClassifier",
                    calculate_linear_classifier_output_shapes,
                    convert_xgboost,
                    options={"nocl": [True, False], "zipmap": [True, False]},
                )
            elif xgboost.sklearn.XGBRegressor in model_types:
                update_registered_converter(
                    xgboost.XGBRegressor,
                    "XGBoostXGBRegressor",
                    calculate_linear_regressor_output_shapes,
                    convert_xgboost,
                )
            if initial_types:
                return convert_sklearn(
                    self.estimator, initial_types=initial_types, **kwargs
                )
            else:
                try:
                    return convert_sklearn(
                        self.estimator,
                        initial_types=auto_generated_initial_types,
                        **kwargs,
                    )
                except:
                    raise ValueError(
                        "`initial_types` can not be autodetected. Please directly pass `initial_types`."
                    )
        else:
            # xgboost api
            if initial_types:
                return onnxmltools.convert_xgboost(
                    self.estimator,
                    initial_types=initial_types,
                    target_opset=kwargs.pop("target_opset", None),
                    targeted_onnx=onnx.__version__,
                    **kwargs,
                )
            else:
                try:
                    return onnxmltools.convert_xgboost(
                        self.estimator,
                        initial_types=auto_generated_initial_types,
                        target_opset=kwargs.pop("target_opset", None),
                        targeted_onnx=onnx.__version__,
                        **kwargs,
                    )
                except:
                    raise ValueError(
                        "`initial_types` can not be autodetected. Please directly pass `initial_types`."
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
        if hasattr(self.estimator, "n_features_in_"):
            # sklearn api
            n_cols = self.estimator.n_features_in_
            return [("input", FloatTensorType([None, n_cols]))]
        elif hasattr(self.estimator, "feature_names") and self.estimator.feature_names:
            # xgboost learning api
            n_cols = len(self.estimator.feature_names)
            return [("input", FloatTensorType([None, n_cols]))]
        if X_sample is None:
            raise ValueError(
                " At least one of `X_sample` or `initial_types` must be provided."
            )
        if (
            X_sample is not None
            and hasattr(X_sample, "shape")
            and len(X_sample.shape) >= 2
        ):
            auto_generated_initial_types = [
                ("input", FloatTensorType([None, X_sample.shape[1]]))
            ]
        else:
            raise ValueError(
                "`initial_types` can not be detected. Please directly pass initial_types."
            )
        return auto_generated_initial_types
