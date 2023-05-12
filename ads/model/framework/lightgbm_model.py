#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ads.common import logger
from ads.model.extractor.lightgbm_extractor import LightgbmExtractor
from ads.model.generic_model import FrameworkSpecificModel
from ads.model.model_properties import ModelProperties
from ads.model.serde.model_serializer import LightGBMModelSerializerType
from ads.model.common.utils import DEPRECATE_AS_ONNX_WARNING
from ads.model.serde.common import SERDE


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
    model_save_serializer_type = LightGBMModelSerializerType

    def __init__(
        self,
        estimator: Callable,
        artifact_dir: Optional[str] = None,
        properties: Optional[ModelProperties] = None,
        auth: Dict = None,
        model_save_serializer: Optional[SERDE] = None,
        model_input_serializer: Optional[SERDE] = None,
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
        model_save_serializer: (SERDE or str, optional). Defaults to None.
            Instance of ads.model.SERDE. Used for serialize/deserialize model.
        model_input_serializer: (SERDE, optional). Defaults to None.
            Instance of ads.model.SERDE. Used for serialize/deserialize data.

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

        default_model_save_serializer = "joblib"
        if model_type.startswith("<class 'lightgbm.basic."):
            default_model_save_serializer = "lightgbm"

        super().__init__(
            estimator=estimator,
            artifact_dir=artifact_dir,
            properties=properties,
            auth=auth,
            model_save_serializer=model_save_serializer
            or default_model_save_serializer,
            model_input_serializer=model_input_serializer,
            **kwargs,
        )
        self._extractor = LightgbmExtractor(estimator)
        self.framework = self._extractor.framework
        self.algorithm = self._extractor.algorithm
        self.version = self._extractor.version
        self.hyperparameter = self._extractor.hyperparameter

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
        Serialize and save Lightgbm model.

        Parameters
        ----------
        as_onnx: (boolean, optional). Defaults to False.
            If set as True, provide `initial_types` or `X_sample` to convert into ONNX.
        initial_types: (List[Tuple], optional). Defaults to None.
            Each element is a tuple of a variable name and a type.
        force_overwrite: (boolean, optional). Defaults to False.
            If set as True, overwrite serialized model if exists.
        X_sample: Union[Dict, str, List, np.ndarray, pd.core.series.Series, pd.core.frame.DataFrame,]. Defaults to None.
            Contains model inputs such that model(`X_sample`) is a valid invocation of the model.
            Used to generate `initial_types`.

        Returns
        -------
        None
            Nothing.
        """
        if as_onnx:
            logger.warning(DEPRECATE_AS_ONNX_WARNING)
            self.set_model_save_serializer("lightgbm_onnx")

        super().serialize_model(
            as_onnx=as_onnx,
            initial_types=initial_types,
            force_overwrite=force_overwrite,
            X_sample=X_sample,
            **kwargs,
        )
