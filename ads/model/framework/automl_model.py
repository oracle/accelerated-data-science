#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import cloudpickle
import numpy as np
import pandas as pd
from ads.common import logger
from ads.model.extractor.automl_extractor import AutoMLExtractor
from ads.model.generic_model import FrameworkSpecificModel
from ads.model.model_properties import ModelProperties

DEFAULT_PKL_FORMAT_MODEL_FILE_NAME = "model.pkl"


class AutoMLModel(FrameworkSpecificModel):
    """AutoMLModel class for estimators from AutoML framework.

    Attributes
    ----------
    algorithm: str
        "ensemble", the algorithm name of the model.
    artifact_dir: str
        Artifact directory to store the files needed for deployment.
    auth: Dict
        Default authentication is set using the `ads.set_auth` API. To override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create
        an authentication signer to instantiate an IdentityClient object.
    ds_client: DataScienceClient
        The data science client used by model deployment.
    estimator: Callable
        A trained automl estimator/model using oracle automl.
    framework: str
        "oracle_automl", the framework name of the estimator.
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
        Name of the serialized model. Default to "model.pkl".
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
    >>> import tempfile
    >>> import logging
    >>> import warnings
    >>> from ads.automl.driver import AutoML
    >>> from ads.automl.provider import OracleAutoMLProvider
    >>> from ads.dataset.dataset_browser import DatasetBrowser
    >>> from ads.model.framework.automl_model import AutoMLModel
    >>> from ads.common.model_metadata import UseCaseType
    >>> ds = DatasetBrowser.sklearn().open("wine").set_target("target")
    >>> train, test = ds.train_test_split(test_size=0.1, random_state = 42)

    >>> ml_engine = OracleAutoMLProvider(n_jobs=-1, loglevel=logging.ERROR)
    >>> oracle_automl = AutoML(train, provider=ml_engine)
    >>> model, baseline = oracle_automl.train(
    ...                model_list=['LogisticRegression', 'DecisionTreeClassifier'],
    ...                random_state = 42,
    ...                time_budget = 500
    ...        )

    >>> automl_model.prepare(inference_conda_env=inference_conda_env, force_overwrite=True)
    >>> automl_model.verify(...)
    >>> automl_model.save()
    >>> model_deployment = automl_model.deploy(wait_for_completion=False)
    """

    _PREFIX = "automl"

    def __init__(
        self,
        estimator: Callable,
        artifact_dir: str,
        properties: Optional[ModelProperties] = None,
        auth: Dict = None,
        **kwargs,
    ):
        """
        Initiates a AutoMLModel instance.

        Parameters
        ----------
        estimator: Callable
            Any model object generated by automl framework.
        artifact_dir: str
            Directory for generate artifact.
        properties: (ModelProperties, optional). Defaults to None.
            ModelProperties object required to save and deploy model.
        auth: (Dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        AutoMLModel
            AutoMLModel instance.

        Raises
        ------
        TypeError
            If the input model is not an AutoML model.
        """
        if not str(type(estimator)).startswith("<class 'ads.common.model.ADSModel'"):
            raise TypeError(f"{str(type(estimator))} is not supported in AutoMLModel.")
        super().__init__(
            estimator=estimator,
            artifact_dir=artifact_dir,
            properties=properties,
            auth=auth,
            **kwargs,
        )
        self._extractor = AutoMLExtractor(estimator)
        self.framework = self._extractor.framework
        self.algorithm = self._extractor.algorithm
        self.version = self._extractor.version
        self.hyperparameter = self._extractor.hyperparameter

    @staticmethod
    def _handle_model_file_name(as_onnx: bool, model_file_name: str):
        if as_onnx:
            raise NotImplementedError(
                "AutoML framework does not support onnx serialization."
            )

        if not model_file_name:
            return DEFAULT_PKL_FORMAT_MODEL_FILE_NAME

        if model_file_name and not model_file_name.endswith(".pkl"):
            raise ValueError(
                "`model_file_name` has to be ending with `.pkl` for pkl format."
            )
        return model_file_name

    def serialize_model(
        self,
        force_overwrite: Optional[bool] = False,
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
        Serialize and save AutoML model using pkl.

        Parameters
        ----------
        force_overwrite: (bool, optional). Defaults to False.
            If set as True, overwrite serialized model if exists.
        X_sample: Union[Dict, str, List, np.ndarray, pd.core.series.Series, pd.core.frame.DataFrame,]. Defaults to None.
            Contains model inputs such that model(X_sample) is a valid invocation of the model.
            Used to generate input schema.

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

            with open(model_path, "wb") as outfile:
                cloudpickle.dump(self.estimator, outfile)
