#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ads.common import logger
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.model.extractor.sklearn_extractor import SklearnExtractor
from ads.common.data_serializer import InputDataSerializer
from ads.model.generic_model import (
    FrameworkSpecificModel,
    DEFAULT_ONNX_FORMAT_MODEL_FILE_NAME,
    DEFAULT_JOBLIB_FORMAT_MODEL_FILE_NAME,
)
from ads.model.model_properties import ModelProperties
from joblib import dump
from pandas.api.types import is_numeric_dtype, is_string_dtype


class SklearnModel(FrameworkSpecificModel):
    """SklearnModel class for estimators from sklearn framework.

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
        A trained sklearn estimator/model using scikit-learn.
    framework: str
        "scikit-learn", the framework name of the model.
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
    >>> import tempfile
    >>> from sklearn.model_selection import train_test_split
    >>> from ads.model.framework.sklearn_model import SklearnModel
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import load_iris

    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    >>> sklearn_estimator = LogisticRegression()
    >>> sklearn_estimator.fit(X_train, y_train)

    >>> sklearn_model = SklearnModel(estimator=sklearn_estimator,
    ... artifact_dir=tmp_model_dir)

    >>> sklearn_model.prepare(inference_conda_env="generalml_p37_cpu_v1", force_overwrite=True)
    >>> sklearn_model.reload()
    >>> sklearn_model.verify(X_test)
    >>> sklearn_model.save()
    >>> model_deployment = sklearn_model.deploy(wait_for_completion=False)
    >>> sklearn_model.predict(X_test)
    """

    _PREFIX = "sklearn"

    def __init__(
        self,
        estimator: Callable,
        artifact_dir: str,
        properties: Optional[ModelProperties] = None,
        auth: Dict = None,
        **kwargs,
    ):
        """
        Initiates a SklearnModel instance.

        Parameters
        ----------
        estimator: Callable
            Sklearn Model
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
        SklearnModel
            SklearnModel instance.


        Examples
        --------
        >>> import tempfile
        >>> from sklearn.model_selection import train_test_split
        >>> from ads.model.framework.sklearn_model import SklearnModel
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import load_iris

        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        >>> sklearn_estimator = LogisticRegression()
        >>> sklearn_estimator.fit(X_train, y_train)

        >>> sklearn_model = SklearnModel(estimator=sklearn_estimator, artifact_dir=tempfile.mkdtemp())
        >>> sklearn_model.prepare(inference_conda_env="dataexpl_p37_cpu_v3")
        >>> sklearn_model.verify(X_test)
        >>> sklearn_model.save()
        >>> model_deployment = sklearn_model.deploy()
        >>> sklearn_model.predict(X_test)
        >>> sklearn_model.delete_deployment()
        """
        if not (
            str(type(estimator)).startswith("<class 'sklearn.")
            or str(type(estimator)).startswith("<class 'onnxruntime.")
        ):
            raise TypeError(f"{str(type(estimator))} is not supported in SklearnModel.")
        super().__init__(
            estimator=estimator,
            artifact_dir=artifact_dir,
            properties=properties,
            auth=auth,
            **kwargs,
        )
        self._extractor = SklearnExtractor(estimator)
        self.framework = self._extractor.framework
        self.algorithm = self._extractor.algorithm
        self.version = self._extractor.version
        self.hyperparameter = self._extractor.hyperparameter

    @staticmethod
    def _handle_model_file_name(as_onnx: bool, model_file_name: str):
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

        Raises
        ------
        ValueError: If the input model_file_name does not corresponding to serialize format.
        """

        if not model_file_name:
            return (
                DEFAULT_ONNX_FORMAT_MODEL_FILE_NAME
                if as_onnx
                else DEFAULT_JOBLIB_FORMAT_MODEL_FILE_NAME
            )
        if as_onnx:
            if model_file_name and not model_file_name.endswith(".onnx"):
                raise ValueError(
                    "`model_file_name` has to be ending with `.onnx` for onnx format."
                )
        else:
            if model_file_name and not model_file_name.endswith(".joblib"):
                raise ValueError(
                    "`model_file_name` has to be ending with `.joblib` for joblib format."
                )
        return model_file_name

    def serialize_model(
        self,
        as_onnx: Optional[bool] = False,
        initial_types: Optional[List[Tuple]] = None,
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
        Serialize and save scikit-learn model using ONNX or model specific method.

        Parameters
        ----------
        as_onnx: (bool, optional). Defaults to False.
            If set as True, provide initial_types or X_sample to convert into ONNX.
        initial_types: (List[Tuple], optional). Defaults to None.
            Each element is a tuple of a variable name and a type.
        force_overwrite: (bool, optional). Defaults to False.
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
                dump(self.estimator, model_path)

    @runtime_dependency(module="onnx", install_from=OptionalDependency.ONNX)
    @runtime_dependency(module="xgboost", install_from=OptionalDependency.BOOSTED)
    @runtime_dependency(module="lightgbm", install_from=OptionalDependency.BOOSTED)
    @runtime_dependency(module="skl2onnx", install_from=OptionalDependency.ONNX)
    @runtime_dependency(module="onnxmltools", install_from=OptionalDependency.ONNX)
    @runtime_dependency(
        module="onnxmltools.convert.xgboost.operator_converters.XGBoost",
        object="convert_xgboost",
        install_from=OptionalDependency.ONNX,
    )
    @runtime_dependency(
        module="onnxmltools.convert.lightgbm.operator_converters.LightGbm",
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
        Produces an equivalent ONNX model of the given scikit-learn model.

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
            An ONNX model (type: ModelProto) which is equivalent to the input scikit-learn model.
        """
        auto_generated_initial_types = None
        if not initial_types:
            if X_sample is None:
                raise ValueError(
                    " At least one of `X_sample` or `initial_types` must be provided."
                )
            auto_generated_initial_types = self.generate_initial_types(X_sample)
        if str(type(self.estimator)).startswith("<class 'sklearn.pipeline"):
            model_types = []
            model_types = [type(val[1]) for val in self.estimator.steps]
            if xgboost.sklearn.XGBClassifier in model_types:
                skl2onnx.update_registered_converter(
                    xgboost.XGBClassifier,
                    "XGBoostXGBClassifier",
                    skl2onnx.common.shape_calculator.calculate_linear_classifier_output_shapes,
                    convert_xgboost,
                    options=kwargs.pop(
                        "options", {"nocl": [True, False], "zipmap": [True, False]}
                    ),
                )

            if xgboost.sklearn.XGBRegressor in model_types:
                skl2onnx.update_registered_converter(
                    xgboost.XGBRegressor,
                    "XGBoostXGBRegressor",
                    skl2onnx.common.shape_calculator.calculate_linear_regressor_output_shapes,
                    convert_xgboost,
                )

            if lightgbm.sklearn.LGBMClassifier in model_types:
                skl2onnx.update_registered_converter(
                    lightgbm.LGBMClassifier,
                    "LightGbmLGBMClassifier",
                    skl2onnx.common.shape_calculator.calculate_linear_classifier_output_shapes,
                    convert_lightgbm,
                    options=kwargs.pop(
                        "options",
                        {"nocl": [True, False], "zipmap": [True, False, "columns"]},
                    ),
                )

            if lightgbm.sklearn.LGBMRegressor in model_types:

                def skl2onnx_convert_lightgbm(scope, operator, container):
                    options = scope.get_options(operator.raw_operator)
                    if "split" in options:
                        if StrictVersion(onnxmltools.__version__) < StrictVersion(
                            "1.9.2"
                        ):
                            logger.warnings(
                                "Option split was released in version 1.9.2 but %s is "
                                "installed. It will be ignored."
                                % onnxmltools.__version__
                            )
                        operator.split = options["split"]
                    else:
                        operator.split = None
                    convert_lightgbm(scope, operator, container)

                skl2onnx.update_registered_converter(
                    lightgbm.LGBMRegressor,
                    "LightGbmLGBMRegressor",
                    skl2onnx.common.shape_calculator.calculate_linear_regressor_output_shapes,
                    skl2onnx_convert_lightgbm,
                    options=kwargs.pop("options", {"split": None}),
                )
            if initial_types:
                return skl2onnx.convert_sklearn(
                    self.estimator, initial_types=initial_types, **kwargs
                )
            else:
                try:
                    return skl2onnx.convert_sklearn(
                        self.estimator,
                        initial_types=auto_generated_initial_types,
                        target_opset=None,
                        **kwargs,
                    )
                except Exception as e:
                    raise ValueError(
                        "`initial_types` can not be autodetected. Please directly pass `initial_types`."
                    )
        else:
            if initial_types:
                return onnxmltools.convert_sklearn(
                    self.estimator,
                    initial_types=initial_types,
                    targeted_onnx=onnx.__version__,
                    **kwargs,
                )
            else:
                try:
                    return onnxmltools.convert_sklearn(
                        self.estimator,
                        initial_types=auto_generated_initial_types,
                        targeted_onnx=onnx.__version__,
                        **kwargs,
                    )
                except Exception as e:
                    raise ValueError(
                        "`initial_types` can not be detected. Please directly pass initial_types."
                    )

    @runtime_dependency(module="skl2onnx", install_from=OptionalDependency.ONNX)
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
        if SklearnModel._is_all_numerical_array_dataframe(X_sample):
            # if it's a dataframe and all the columns are numerical. Or
            # it's not a dataframe, also try this.
            if hasattr(X_sample, "shape") and len(X_sample.shape) >= 2:
                auto_generated_initial_types = [
                    (
                        "input",
                        skl2onnx.common.data_types.FloatTensorType(
                            [None, X_sample.shape[1]]
                        ),
                    )
                ]
            elif hasattr(self.estimator, "n_features_in_"):
                n_cols = self.estimator.n_features_in_
                auto_generated_initial_types = [
                    (
                        "input",
                        skl2onnx.common.data_types.FloatTensorType([None, n_cols]),
                    )
                ]
            else:
                raise ValueError(
                    "`initial_types` can not be detected. Please directly pass initial_types."
                )
        elif SklearnModel.is_either_numerical_or_string_dataframe(X_sample):
            # for dataframe and not all the columns are numerical, then generate
            # the input types of all the columns one by one.
            auto_generated_initial_types = []

            for i, col in X_sample.iteritems():
                if is_numeric_dtype(col.dtypes):
                    auto_generated_initial_types.append(
                        (
                            col.name,
                            skl2onnx.common.data_types.FloatTensorType([None, 1]),
                        )
                    )
                else:
                    auto_generated_initial_types.append(
                        (
                            col.name,
                            skl2onnx.common.data_types.StringTensorType([None, 1]),
                        )
                    )
        else:
            try:
                auto_generated_initial_types = (
                    skl2onnx.common.data_types.guess_data_type(
                        np.array(X_sample) if isinstance(X_sample, list) else X_sample
                    )
                )
            except:
                auto_generated_initial_types = None
        return auto_generated_initial_types

    @staticmethod
    def _is_all_numerical_array_dataframe(
        data: Union[pd.DataFrame, np.ndarray]
    ) -> bool:
        """Check whether all the columns are numerical for numpy array and dataframe.
        For data with any other data types, it will return False.

        Parameters
        ----------
        data: Union[pd.DataFrame, np.ndarray]

        Returns
        -------
        bool
            Whether all the columns in a pandas dataframe or numpy array are all numerical.
        """
        return (
            isinstance(data, pd.DataFrame)
            and all([is_numeric_dtype(dtype) for dtype in data.dtypes])
            or (isinstance(data, np.ndarray) and is_numeric_dtype(data.dtype))
        )

    @staticmethod
    def is_either_numerical_or_string_dataframe(data: pd.DataFrame) -> bool:
        """Check whether all the columns are either numerical or string for dataframe."""
        return isinstance(data, pd.DataFrame) and all(
            [
                is_numeric_dtype(col.dtypes) or is_string_dtype(col.dtypes)
                for _, col in data.iteritems()
            ]
        )
