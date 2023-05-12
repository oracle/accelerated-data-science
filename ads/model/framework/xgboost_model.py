#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ads.common import logger
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.model.extractor.xgboost_extractor import XgboostExtractor
from ads.model.generic_model import FrameworkSpecificModel
from ads.model.model_properties import ModelProperties
from ads.model.serde.model_serializer import XgboostModelSerializerType
from ads.model.common.utils import DEPRECATE_AS_ONNX_WARNING
from ads.model.serde.common import SERDE


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
    model_save_serializer_type = XgboostModelSerializerType

    @runtime_dependency(module="xgboost", install_from=OptionalDependency.BOOSTED)
    def __init__(
        self,
        estimator: callable,
        artifact_dir: Optional[str] = None,
        properties: Optional[ModelProperties] = None,
        auth: Dict = None,
        model_save_serializer: Optional[SERDE] = model_save_serializer_type.XGBOOST,
        model_input_serializer: Optional[SERDE] = None,
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
        model_save_serializer: (SERDE or str, optional). Defaults to None.
            Instance of ads.model.SERDE. Used for serialize/deserialize model.
        model_input_serializer: (SERDE, optional). Defaults to None.
            Instance of ads.model.SERDE. Used for serialize/deserialize data.

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
            model_save_serializer=model_save_serializer,
            model_input_serializer=model_input_serializer,
            **kwargs,
        )
        self._extractor = XgboostExtractor(estimator)
        self.framework = self._extractor.framework
        self.algorithm = self._extractor.algorithm
        self.version = self._extractor.version
        self.hyperparameter = self._extractor.hyperparameter

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
        if as_onnx:
            logger.warning(DEPRECATE_AS_ONNX_WARNING)
            self.set_model_save_serializer(self.model_save_serializer_type.ONNX)

        super().serialize_model(
            as_onnx=as_onnx,
            initial_types=initial_types,
            force_overwrite=force_overwrite,
            X_sample=X_sample,
            **kwargs,
        )
