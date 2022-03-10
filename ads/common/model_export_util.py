#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
import os
import sys
from typing import Union, Any

import cloudpickle
import numpy as np
import onnx
import onnxmltools
import pandas as pd
from ads.common import logger, utils
from ads.common.data import ADSData
from ads.common.function.fn_util import (
    generate_fn_artifacts,
    get_function_config,
    write_score,
)
from ads.common.model_artifact import ModelArtifact
from ads.common.model_metadata import UseCaseType
from ads.feature_engineering.schema import DataSizeTooWide
from pkg_resources import get_distribution, DistributionNotFound
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes,
)
from sklearn.preprocessing import LabelEncoder

pd.options.mode.chained_assignment = None

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ADS")

OnnxConvertibleModels = [
    "sklearn",
    "automl",
    "lightgbm",
    "xgboost",
    "torch",
    "tensorflow",
    "keras",
    "mxnet",
]
NoVerifyModels = ["automl", "torch", "mxnet", "lightgbm", "xgboost"]
AlreadyWrittenModels = ["torch", "mxnet", "automl"]
TransformableData = ["automl"]
Progress_Steps_W_Fn = 6
Progress_Steps_Wo_Fn = 4


def prepare_generic_model(
    model_path: str,
    fn_artifact_files_included: bool = False,
    fn_name: str = "model_api",
    force_overwrite: bool = False,
    model: Any = None,
    data_sample: ADSData = None,
    use_case_type=None,
    X_sample: Union[
        list,
        tuple,
        pd.Series,
        np.ndarray,
        pd.DataFrame,
    ] = None,
    y_sample: Union[
        list,
        tuple,
        pd.Series,
        np.ndarray,
        pd.DataFrame,
    ] = None,
    **kwargs,
) -> ModelArtifact:
    """
    Generates template files to aid model deployment.
    The model could be accompanied by other artifacts all of which can be dumped at `model_path`.
    Following files are generated:
    * func.yaml
    * func.py
    * requirements.txt
    * score.py

    Parameters
    ----------
    model_path : str
        Path where the artifacts must be saved.
        The serialized model object and any other associated files/objects must
        be saved in the `model_path` directory
    fn_artifact_files_included : bool
        Default is False, if turned off, function artifacts are not generated.
    fn_name : str
        Opional parameter to specify the function name
    force_overwrite : bool
        Opional parameter to specify if the model_artifact should overwrite the existing model_path (if it exists)
    model : (Any, optional). Defaults to None.
        This is an optional model object which is only used to extract taxonomy metadata.
        Supported models: automl, keras, lightgbm, pytorch, sklearn, tensorflow, and xgboost.
        If the model is not under supported frameworks, then extracting taxonomy metadata will be skipped.
        The alternative way is using `atifact.populate_metadata(model=model, usecase_type=UseCaseType.REGRESSION)`.
    data_sample : ADSData
        A sample of the test data that will be provided to predict() API of scoring script
        Used to generate schema_input and schema_output
    use_case_type: str
        The use case type of the model
    X_sample : Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame, dask.dataframe.core.Series, dask.dataframe.core.DataFrame]
        A sample of input data that will be provided to predict() API of scoring script
        Used to generate input schema.
    y_sample : Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame, dask.dataframe.core.Series, dask.dataframe.core.DataFrame]
        A sample of output data that is expected to be returned by predict() API of scoring script,
        corresponding to X_sample
        Used to generate output schema.

    **kwargs
    ________
    data_science_env : bool, default: False
        If set to True, the datascience environment represented by the slug in the training conda environment will be used.
    inference_conda_env : str, default: None
        Conda environment to use within the model deployment service for inferencing. For example, oci://bucketname@namespace/path/to/conda/env
    ignore_deployment_error : bool, default: False
        If set to True, the prepare method will ignore all the errors that may impact model deployment.
    underlying_model : str, default: 'UNKNOWN'
        Underlying Model Type, could be "automl", "sklearn", "h2o", "lightgbm", "xgboost", "torch", "mxnet", "tensorflow", "keras", "pyod" and etc.
    model_libs : dict, default: {}
        Model required libraries where the key is the library names and the value is the library versions.
        For example, {numpy: 1.21.1}.
    progress : int, default: None
        max number of progress.
    inference_python_version: str, default:None.
        If provided will be added to the generated runtime yaml
    max_col_num: (int, optional). Defaults to utils.DATA_SCHEMA_MAX_COL_NUM.
        The maximum column size of the data that allows to auto generate schema.

    Examples
    --------
    >>> import cloudpickle
    >>> import os
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> import ads
    >>> from ads.common.model_export_util import prepare_generic_model
    >>> import yaml
    >>>
    >>> ads.set_auth('api_key', profile='DEFAULT')
    >>> model_artifact_location = os.path.expanduser('~/myusecase/model/')
    >>> inference_conda_env="oci://my-bucket@namespace/conda_environments/cpu/Data Exploration and Manipulation for CPU Python 3.7/2.0/dataexpl_p37_cpu_v2"
    >>> inference_python_version = "3.7"
    >>> if not os.path.exists(model_artifact_location):
    ...     os.makedirs(model_artifact_location)
    >>> X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
    >>> lrmodel = LogisticRegression().fit(X, y)
    >>> with open(os.path.join(model_artifact_location, 'model.pkl'), "wb") as mfile:
    ...     cloudpickle.dump(lrmodel, mfile)
    >>> modelartifact = prepare_generic_model(
    ...     model_artifact_location,
    ...     model = lrmodel,
    ...     force_overwrite=True,
    ...     inference_conda_env=inference_conda_env,
    ...     ignore_deployment_error=True,
    ...     inference_python_version=inference_python_version
    ... )
    >>> modelartifact.reload() # Call reload to update the ModelArtifact object with the generated score.py
    >>> assert len(modelartifact.predict(X[:5])['prediction']) == 5 #Test the generated score.py works. This may require customization.
    >>> with open(os.path.join(model_artifact_location, "runtime.yaml")) as rf:
    ...     content = yaml.load(rf, Loader=yaml.FullLoader)
    ...     assert content['MODEL_DEPLOYMENT']['INFERENCE_CONDA_ENV']['INFERENCE_ENV_PATH'] == inference_conda_env
    ...     assert content['MODEL_DEPLOYMENT']['INFERENCE_CONDA_ENV']['INFERENCE_PYTHON_VERSION'] == inference_python_version
    >>> # Save Model to model artifact
    >>> ocimodel = modelartifact.save(
    ...     project_id="oci1......", # OCID of the project to which the model to be associated
    ...     compartment_id="oci1......", # OCID of the compartment where the model will reside
    ...     display_name="LRModel_01",
    ...     description="My Logistic Regression Model",
    ...     ignore_pending_changes=True,
    ...     timeout=100,
    ...     ignore_introspection=True,
    ... )
    >>> print(f"The OCID of the model is: {ocimodel.id}")

    Returns
    -------
    model_artifact: ads.model_artifact.model_artifact
        A generic model artifact
    """
    if "function_artifacts" in kwargs:
        if (
            fn_artifact_files_included
            or fn_artifact_files_included != kwargs["function_artifacts"]
        ):
            raise ValueError(
                "Parameter 'function_artifacts' and 'fn_artifact_files_included' cannot be used at "
                "the same time. Parameter 'function_artifacts' is deprecated and removed in future releases."
            )
        else:
            logger.warning(
                "Parameter 'function_artifacts' is deprecated and removed in future releases. Use 'fn_artifact_files_included' instead."
            )
            fn_artifact_files_included = kwargs["function_artifacts"]

    assert model_path, "Required folder path for saving artifacts"

    # If this is being called from ADSModel.prepare, grab relevant data
    underlying_model = kwargs.get("underlying_model", "UNKNOWN")
    model_libs = kwargs.get(
        "model_libs", {}
    )  # utils.extract_lib_dependencies_from_model(self.est)
    progress = kwargs.get("progress", None)
    max_col_num = kwargs.get("max_col_num", utils.DATA_SCHEMA_MAX_COL_NUM)
    artifact_type_generic = progress is not None

    from ads.common.model import ADSModel

    if isinstance(model, ADSModel) and underlying_model != "automl":
        raise ValueError(
            "Only generic model can be used to generate generic model artifact."
        )

    if use_case_type and use_case_type not in UseCaseType:
        raise ValueError(f"Invalid usecase type. Choose from {UseCaseType.values()}")

    with progress if artifact_type_generic else utils.get_progress_bar(
        Progress_Steps_W_Fn if fn_artifact_files_included else Progress_Steps_Wo_Fn
    ) as progress:
        if not artifact_type_generic:
            progress.update("Preparing Model Artifact Directory")
            if os.path.exists(model_path):
                if force_overwrite:
                    logger.warning(
                        f"As force_overwrite is set to True, all the existing files in the {model_path} will be removed"
                    )
                else:
                    raise ValueError(
                        "Directory already exists, set force_overwrite to True if you wish to overwrite."
                    )

            os.makedirs(model_path, exist_ok=True)

        progress.update("Updating requirements.txt")
        if fn_artifact_files_included:
            # fdk removed from dependency list in setup.py (fn deployments deprecated)
            # before we request versions we want to check if fdk installed by user
            # and provide support in error message, if not installed
            try:
                get_distribution('fdk')
            except Exception as e:
                if isinstance(e, DistributionNotFound):
                    error_message = "fdk library not installed in current environment, it is required " \
                                    "for deployment with fn. Install fdk with 'pip install fdk'."
                    logger.error(str(error_message))
                    raise
            else:
                required_fn_libs = get_function_config()["requires"]["functions"]
                [
                    model_libs.update({lib: get_distribution(lib).version})
                    for lib in required_fn_libs
                ]
                required_model_libs = get_function_config()["requires"][
                    kwargs.get("serializer", "default")
                ]
                [
                    model_libs.update({lib: get_distribution(lib).version})
                    for lib in required_model_libs
                ]
                utils.generate_requirement_file(
                    requirements=model_libs, file_path=model_path
                )

        model_artifact_args = {}
        if "inference_conda_env" in kwargs:
            model_artifact_args["inference_conda_env"] = kwargs["inference_conda_env"]
        if "inference_python_version" in kwargs:
            model_artifact_args["inference_python_version"] = kwargs[
                "inference_python_version"
            ]
        if "data_science_env" in kwargs:
            model_artifact_args["data_science_env"] = kwargs["data_science_env"]
        if "ignore_deployment_error" in kwargs:
            model_artifact_args["ignore_deployment_error"] = kwargs[
                "ignore_deployment_error"
            ]
        model_artifact = ModelArtifact(
            model_path,
            reload=False,
            create=True,
            progress=progress,
            **model_artifact_args,
        )

        model_kwargs = {"_underlying_model": underlying_model, "progress": progress}
        kwargs.update(model_kwargs)
        if fn_artifact_files_included:
            generate_fn_artifacts(
                model_path,
                fn_name,
                artifact_type_generic=artifact_type_generic,
                **kwargs,
            )
        if progress:
            progress.update("Writing score.py")
        write_score(model_path, **kwargs)
        if model is None:
            logger.warning(
                "Taxonomy metadata was not extracted. "
                "To auto-populate taxonomy metadata the model must be provided. "
                "Pass the model as a parameter to .prepare_generic_model(model=model, usecase_type=UseCaseType.REGRESSION). "
                "Alternative way is using atifact.populate_metadata(model=model, usecase_type=UseCaseType.REGRESSION)."
            )
        try:
            model_artifact.populate_metadata(model, use_case_type)
        except:
            logger.warning("Failed to populate the custom and taxonomy metadata.")

        try:
            model_artifact.populate_schema(
                data_sample,
                X_sample,
                y_sample,
                max_col_num,
            )
        except DataSizeTooWide:
            logger.warning(
                f"The data has too many columns and "
                f"the maximum allowable number of columns is `{max_col_num}`. "
                "The schema was not auto generated. increase allowable number of columns."
            )

        return model_artifact


def serialize_model(
    model=None, target_dir=None, X=None, y=None, model_type=None, **kwargs
):
    """
    Parameters
    ----------
    model : ads.Model
        A model to be serialized
    target_dir : str, optional
        directory to output the serialized model
    X : Union[pandas.DataFrame, dask.DataFrame, numpy.ndarray, scipy.sparse.csr.csr_matrix]
        The X data
    y : Union[list, pandas.DataFrame, dask.DataFrame, pandas.Series, dask.Series, numpy.ndarray]
        Tbe Y data
    model_type : str, optional
        A string corresponding to the model type

    Returns
    -------
    model_kwargs: Dict
        A dictionary of model kwargs for the serialized model
    """
    model_kwargs = {}
    try:
        assert model_type in OnnxConvertibleModels
        assert X is not None, (
            "WARNING: In order to convert model to onnnx format, you will need to provide a data "
            "sample "
        )
        # Try to serialize the model based off of type
        onx = None
        if model_type == "sklearn":
            onx = _sklearn_to_onnx(model=model)
        elif model_type == "automl":
            _automl_to_pkl(model=model, target_dir=target_dir)
        elif model_type == "lightgbm":
            onx = _lightgbm_to_onnx(model=model, X=X, y=y)
        elif model_type == "torch":
            _torch_to_onnx(model=model, target_dir=target_dir, X=X, y=y)
        elif model_type == "xgboost":
            onx = _xgboost_to_onnx(model=model, X=X, y=y)
        elif model_type == "tensorflow":
            onx = _tf_to_onnx(model=model, X=X, y=y)
        elif model_type == "keras":
            onx = _keras_to_onnx(model=model, X=X, y=y)
        elif model_type == "mxnet":
            _mxnet_to_onnx(model=model, X=X, y=y)
        else:
            raise Exception(
                f"ADS does not have method to convert models of type: {model_type} to onnx"
            )

        # Save the model to the target file
        if model_type not in AlreadyWrittenModels:
            with open(os.path.join(target_dir, "model.onnx"), "wb") as f:
                f.write(onx.SerializeToString())

        # Lastly, set the necessary env variables for the remaining workflow
        if model_type == "automl":
            model_kwargs["model_name"] = "model.pkl"
            model_kwargs["serializer"] = "pkl"
        else:
            model_kwargs["model_name"] = "model.onnx"
            model_kwargs["serializer"] = "onnx"
    except Exception as e:
        logger.error(
            f"Failed to serialize the model as ONNX returned the error message::{e}."
        )
    return model_kwargs


def _sklearn_to_onnx(model=None, target_dir=None, X=None, y=None, **kwargs):
    n_cols = model.est.n_features_in_
    initial_types = [("input", FloatTensorType([None, n_cols]))]
    return onnxmltools.convert_sklearn(
        model.est,
        name=None,
        initial_types=initial_types,
        doc_string="",
        target_opset=None,
        targeted_onnx=onnx.__version__,
        custom_conversion_functions=None,
        custom_shape_calculators=None,
    )


def _automl_to_pkl(model=None, target_dir=None, **kwargs):

    with open(os.path.join(target_dir, "model.pkl"), "wb") as outfile:
        cloudpickle.dump(model, outfile)


def _lightgbm_to_onnx(model=None, target_dir=None, X=None, y=None, **kwargs):
    in_types = [("input", FloatTensorType([None, len(X.columns)]))]
    import lightgbm as lgb

    if str(type(model.est)) == "<class 'sklearn.pipeline.Pipeline'>":
        model_est_types = [type(val[1]) for val in model.est.steps]
    else:
        model_est_types = [type(model.est)]

    if lgb.sklearn.LGBMClassifier in model_est_types:
        from lightgbm import LGBMClassifier
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
            convert_lightgbm,
        )

        update_registered_converter(
            LGBMClassifier,
            "LightGbmLGBMClassifier",
            calculate_linear_classifier_output_shapes,
            convert_lightgbm,
            options={"nocl": [True, False], "zipmap": [True, False]},
        )
    elif lgb.sklearn.LGBMRegressor in model_est_types:
        from lightgbm import LGBMRegressor
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
            convert_lightgbm,
        )

        update_registered_converter(
            LGBMRegressor,
            "LightGbmLGBMRegressor",
            calculate_linear_regressor_output_shapes,
            convert_lightgbm,
            options={"nocl": [True, False], "zipmap": [True, False]},
        )
    else:
        return onnxmltools.convert_lightgbm(
            model.est,
            name=None,
            initial_types=in_types,
            doc_string="",
            target_opset=None,
            targeted_onnx=onnx.__version__,
            custom_conversion_functions=None,
            custom_shape_calculators=None,
        )
    return convert_sklearn(model.est, initial_types=in_types)


def _xgboost_to_onnx(model=None, target_dir=None, X=None, y=None, **kwargs):
    in_types = [("input", FloatTensorType([None, len(X.columns)]))]
    import xgboost as xgb

    if str(type(model.est)) == "<class 'sklearn.pipeline.Pipeline'>":
        model_est_types = [type(val[1]) for val in model.est.steps]
    else:
        model_est_types = [type(model.est)]
    if xgb.sklearn.XGBClassifier in model_est_types:

        from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
            convert_xgboost,
        )
        from xgboost import XGBClassifier

        update_registered_converter(
            XGBClassifier,
            "XGBoostXGBClassifier",
            calculate_linear_classifier_output_shapes,
            convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False]},
        )
    elif xgb.sklearn.XGBRegressor in model_est_types:
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
            convert_xgboost,
        )
        from xgboost import XGBRegressor

        update_registered_converter(
            XGBRegressor,
            "XGBoostXGBRegressor",
            calculate_linear_regressor_output_shapes,
            convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False]},
        )
    else:
        from onnxmltools.convert import convert_xgboost

        return convert_xgboost(
            model,
            name=None,
            initial_types=in_types,
            doc_string="",
            target_opset=None,
            targeted_onnx=onnx.__version__,
            custom_conversion_functions=None,
            custom_shape_calculators=None,
        )
    return convert_sklearn(model.est, initial_types=in_types)


def _torch_to_onnx(model=None, target_dir=None, X=None, y=None, **kwargs):
    import torch

    assert hasattr(torch, "onnx"), (
        f"This version of pytorch {torch.__version__} does not appear to support onnx "
        f"conversion "
    )
    # Add variable batch size to the beginning of the shape
    sample_input = torch.randn([1] + [X[:1].shape[1]], requires_grad=True)
    torch.onnx.export(
        model.est,
        sample_input,
        f=os.path.join(target_dir, "model.onnx"),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def _tf_to_onnx(model=None, target_dir=None, X=None, y=None, **kwargs):
    try:
        from onnxmltools.convert import convert_tensorflow

        return convert_tensorflow(
            model,
            name=None,
            input_names=None,
            output_names=None,
            doc_string="",
            target_opset=None,
            channel_first_inputs=None,
            debug_mode=False,
            custom_op_conversions=None,
        )
    except Exception as e:
        _log_automatic_serialization_keras(e)


def _keras_to_onnx(model=None, target_dir=None, X=None, y=None, **kwargs):
    try:
        from onnxmltools.convert import convert_keras

        return convert_keras(
            model,
            name=None,
            initial_types=None,
            doc_string="",
            target_opset=None,
            targeted_onnx=onnx.__version__,
            channel_first_inputs=None,
            custom_conversion_functions=None,
            custom_shape_calculators=None,
            default_batch_size=1,
        )
    except Exception as e:
        _log_automatic_serialization_keras(e)


def _mxnet_to_onnx(model=None, target_dir=None, X=None, y=None, **kwargs):
    assert onnx.__version__ == "1.3.0", "Mxnet can only export to onnx version 1.3.0"
    from mxnet.contrib import onnx as onnx_mxnet

    sym = kwargs.get("sym", None)
    params = kwargs.get("params", None)
    assert sym is not None and params is not None, (
        "Pass `sym` [Path to the json file or Symbol object] and "
        "`params` [Path to the params file or params dictionary. "
        "(Including both arg_params and aux_params)] to kwargs"
    )
    input_shape = X.shape
    onnx_file = "model.onnx"
    onnx_mxnet.export_model(
        sym=sym,
        params=params,
        input_shape=[input_shape],
        input_type=np.float32,
        onnx_file_path=onnx_file,
    )


def _log_automatic_serialization_keras(e):
    logger.error(
        f"The following error occured: {e}. "
        f"This may be because automatic serialization for Keras models is not supported. "
        f"Use `prepare_generic_model()` method to manually serialize the model."
    )


# # Note to developers: If you make any changes to this class, copy and paste those changes over to
# templates/score.jinja2. We do not yet have an automatic way of doing this.
class ONNXTransformer(object):
    """
    This is a transformer to convert X [pandas.Dataframe, dask.Dataframe, equivalent] and y [array like] data into Onnx
    readable dtypes and formats. It is Serializable, so it can be reloaded at another time.

    Parameters
    ----------
    task: str
        Either "classification" or "regression". This determines if y should be label encoded

    Examples
    --------
    >>> from ads.common.model_export_util import ONNXTransformer
    >>> onnx_data_transformer = ONNXTransformer(task="classification")
    >>> train_transformed = onnx_data_transformer.fit_transform(train.X, train.y)
    >>> test_transformed = onnx_data_transformer.transform(test.X, test.y)
    """

    def __init__(self, task=None):
        assert task in ["classification", "regression"], (
            "The ONNXTransformer only supports the following task types:"
            " [regression, classification]."
        )
        self.task = task
        self.cat_impute_values = {}
        self.cat_unique_values = {}
        self.label_encoder = None
        self.dtypes = None
        self._fitted = False

    def _handle_dtypes(self, X):
        # Data type cast could be expensive doing it in a for loop
        # Especially with wide datasets
        # So cast the numerical columns first, without loop
        # Then impute categorical columns
        dict_astype = {}
        for k, v in zip(X.columns, X.dtypes):
            if "int" in str(v) or "float" in str(v):
                dict_astype[k] = "float32"
        _X = X.astype(dict_astype)
        for k in _X.columns[_X.dtypes != "float32"]:
            # SimpleImputer is not available for strings in ONNX-ML specifications
            # Replace NaNs with the most frequent category
            self.cat_impute_values[k] = _X[k].value_counts().idxmax()
            _X[k] = _X[k].fillna(self.cat_impute_values[k])
            # Sklearn's OrdinalEncoder and LabelEncoder don't support unseen categories in test data
            # Label encode them to identify new categories in test data
            self.cat_unique_values[k] = _X[k].unique().tolist()
        return _X

    def fit(self, X, y=None):
        """
        Fits the OnnxTransformer on the dataset
        Parameters
        ----------
        X : Union[pandas.DataFrame, dask.DataFrame, numpy.ndarray, scipy.sparse.csr.csr_matrix]
            The Dataframe for the training data
        y: Union[list, pandas.DataFrame, dask.DataFrame, pandas.Series, dask.Series, numpy.ndarray]
            The labels

        Returns
        -------
        Self: ads.Model
            The fitted estimator
        """
        _X = self._handle_dtypes(X)
        self.dtypes = _X.dtypes
        if self.task == "classification" and y is not None:
            # Label encoding is required for SVC's onnx converter
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        self._fitted = True
        return self

    def transform(self, X, y=None):
        """
        Transforms the data for the OnnxTransformer
        Parameters
        ----------
        X: Union[pandas.DataFrame, dask.DataFrame, numpy.ndarray, scipy.sparse.csr.csr_matrix]
            The Dataframe for the training data
        y: Union[list, pandas.DataFrame, dask.DataFrame, pandas.Series, dask.Series, numpy.ndarray]
            The labels

        Returns
        -------
        _X: Array
            The transformed X data
        y: Array
            The transformed list of labels

        """
        assert self._fitted, "Call fit_transform first!"
        _X = X.astype(self.dtypes)
        for k in _X.columns[_X.dtypes != "float32"]:
            # Replace unseen categories with NaNs and impute them
            _X.loc[~_X[k].isin(self.cat_unique_values[k]), k] = np.nan
            # SimpleImputer is not available for strings in ONNX-ML specifications
            # Replace NaNs with the most frequent category
            _X[k] = _X[k].fillna(self.cat_impute_values[k])

        if self.label_encoder is not None and y is not None:
            y = self.label_encoder.transform(y)

        return _X, y

    def fit_transform(self, X, y=None):
        """
        Fits, then transforms the data
        Parameters
        ----------
        X: Union[pandas.DataFrame, dask.DataFrame, numpy.ndarray, scipy.sparse.csr.csr_matrix]
            The Dataframe for the training data
        y: Union[list, pandas.DataFrame, dask.DataFrame, pandas.Series, dask.Series, numpy.ndarray]
            The labels

        Returns
        -------
        _X: Array
            The fitted and transformed X data
        y: Array
            The fitted and transformed list of labels
        """
        return self.fit(X, y).transform(X, y)

    def save(self, filename, **kwargs):
        """
        Saves the Onnx model to disk
        Parameters
        ----------
        filename: Str
            The filename location for where the model should be saved

        Returns
        -------
        filename: Str
            The filename where the model was saved
        """
        export_dict = {
            "task": {"value": self.task, "dtype": str(type(self.task))},
            "cat_impute_values": {
                "value": self.cat_impute_values,
                "dtype": str(type(self.cat_impute_values)),
            },
            "cat_unique_values": {
                "value": self.cat_unique_values,
                "dtype": str(type(self.cat_unique_values)),
            },
            "label_encoder": {
                "value": {
                    "params": self.label_encoder.get_params()
                    if hasattr(self.label_encoder, "get_params")
                    else {},
                    "classes_": self.label_encoder.classes_.tolist()
                    if hasattr(self.label_encoder, "classes_")
                    else [],
                },
                "dtype": str(type(self.label_encoder)),
            },
            "dtypes": {
                "value": {
                    "index": list(self.dtypes.index),
                    "values": [str(val) for val in self.dtypes.values],
                }
                if self.dtypes is not None
                else {},
                "dtype": str(type(self.dtypes)),
            },
            "_fitted": {"value": self._fitted, "dtype": str(type(self._fitted))},
        }
        with open(filename, "w") as f:
            json.dump(export_dict, f, sort_keys=True, indent=4, separators=(",", ": "))
        return filename

    @staticmethod
    def load(filename, **kwargs):
        """
        Loads the Onnx model to disk
        Parameters
        ----------
        filename: Str
            The filename location for where the model should be loaded

        Returns
        -------
        onnx_transformer: ONNXTransformer
            The loaded model
        """
        # Make sure you have  pandas, numpy, and sklearn imported
        with open(filename, "r") as f:
            export_dict = json.load(f)
        try:
            onnx_transformer = ONNXTransformer(task=export_dict["task"]["value"])
        except Exception as e:
            print(f"No task set in ONNXTransformer at {filename}")
            raise e
        for key in export_dict.keys():
            if key not in ["task", "label_encoder", "dtypes"]:
                try:
                    setattr(onnx_transformer, key, export_dict[key]["value"])
                except Exception as e:
                    print(
                        f"Warning: Failed to reload {key} from {filename} to OnnxTransformer."
                    )
                    raise e
        onnx_transformer.dtypes = pd.Series(
            data=[np.dtype(val) for val in export_dict["dtypes"]["value"]["values"]],
            index=export_dict["dtypes"]["value"]["index"],
        )
        le = LabelEncoder()
        le.set_params(**export_dict["label_encoder"]["value"]["params"])
        le.classes_ = np.asarray(export_dict["label_encoder"]["value"]["classes_"])
        onnx_transformer.label_encoder = le
        return onnx_transformer


if __name__ == "__main__":
    pass
