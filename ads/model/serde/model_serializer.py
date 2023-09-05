#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import cloudpickle
import numpy as np
import pandas as pd
from ads.model.serde.common import Serializer, Deserializer
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common import logger
from pandas.api.types import is_numeric_dtype, is_string_dtype
from typing import Any, Dict, List, Optional, Tuple, Union
from joblib import dump


MODEL_SERIALIZATION_TYPE_ONNX = "onnx"
MODEL_SERIALIZATION_TYPE_CLOUDPICKLE = "cloudpickle"
MODEL_SERIALIZATION_TYPE_TORHCSCRIPT = "torchscript"
MODEL_SERIALIZATION_TYPE_TORCH = "torch"
MODEL_SERIALIZATION_TYPE_TORCH_ONNX = "torch_onnx"
MODEL_SERIALIZATION_TYPE_TF = "tf"
MODEL_SERIALIZATION_TYPE_TF_ONNX = "tf_onnx"
MODEL_SERIALIZATION_TYPE_JOBLIB = "joblib"
MODEL_SERIALIZATION_TYPE_SKLEARN_ONNX = "sklearn_onnx"
MODEL_SERIALIZATION_TYPE_LIGHTGBM = "lightgbm"
MODEL_SERIALIZATION_TYPE_LIGHTGBM_ONNX = "lightgbm_onnx"
MODEL_SERIALIZATION_TYPE_XGBOOST = "xgboost"
MODEL_SERIALIZATION_TYPE_XGBOOST_UBJ = "xgboost_ubj"
MODEL_SERIALIZATION_TYPE_XGBOOST_TXT = "xgboost_txt"
MODEL_SERIALIZATION_TYPE_XGBOOST_ONNX = "xgboost_onnx"
MODEL_SERIALIZATION_TYPE_SPARK = "spark"
MODEL_SERIALIZATION_TYPE_HUGGINGFACE = "huggingface"


SUPPORTED_MODEL_SERIALIZERS = [
    MODEL_SERIALIZATION_TYPE_ONNX,
    MODEL_SERIALIZATION_TYPE_CLOUDPICKLE,
    MODEL_SERIALIZATION_TYPE_TORHCSCRIPT,
    MODEL_SERIALIZATION_TYPE_TORCH,
    MODEL_SERIALIZATION_TYPE_TORCH_ONNX,
    MODEL_SERIALIZATION_TYPE_TF,
    MODEL_SERIALIZATION_TYPE_TF_ONNX,
    MODEL_SERIALIZATION_TYPE_JOBLIB,
    MODEL_SERIALIZATION_TYPE_SKLEARN_ONNX,
    MODEL_SERIALIZATION_TYPE_LIGHTGBM,
    MODEL_SERIALIZATION_TYPE_LIGHTGBM_ONNX,
    MODEL_SERIALIZATION_TYPE_XGBOOST,
    MODEL_SERIALIZATION_TYPE_XGBOOST_ONNX,
    MODEL_SERIALIZATION_TYPE_SPARK,
    MODEL_SERIALIZATION_TYPE_HUGGINGFACE,
]


class ModelSerializerType:
    CLOUDPICKLE = MODEL_SERIALIZATION_TYPE_CLOUDPICKLE
    ONNX = MODEL_SERIALIZATION_TYPE_ONNX


class PyTorchModelSerializerType:
    TORCH = MODEL_SERIALIZATION_TYPE_TORCH
    TORCHSCRIPT = MODEL_SERIALIZATION_TYPE_TORHCSCRIPT
    ONNX = MODEL_SERIALIZATION_TYPE_TORCH_ONNX


class TensorflowModelSerializerType:
    TENSORFLOW = MODEL_SERIALIZATION_TYPE_TF
    ONNX = MODEL_SERIALIZATION_TYPE_TF_ONNX


class LightGBMModelSerializerType:
    LIGHTGBM = MODEL_SERIALIZATION_TYPE_LIGHTGBM
    ONNX = MODEL_SERIALIZATION_TYPE_LIGHTGBM_ONNX


class SklearnModelSerializerType:
    JOBLIB = MODEL_SERIALIZATION_TYPE_JOBLIB
    CLOUDPICKLE = MODEL_SERIALIZATION_TYPE_CLOUDPICKLE
    ONNX = MODEL_SERIALIZATION_TYPE_SKLEARN_ONNX


class XgboostModelSerializerType:
    XGBOOST = MODEL_SERIALIZATION_TYPE_XGBOOST
    ONNX = MODEL_SERIALIZATION_TYPE_XGBOOST_ONNX


class SparkModelSerializerType:
    SPARK = MODEL_SERIALIZATION_TYPE_SPARK


class HuggingFaceSerializerType:
    HUGGINGFACE = MODEL_SERIALIZATION_TYPE_HUGGINGFACE


class ModelSerializer(Serializer):
    """Base class for creation of new model serializers."""

    def __init__(self, model_file_suffix):
        super().__init__()
        self.model_file_suffix = model_file_suffix


class ModelDeserializer(Deserializer):
    """Base class for creation of new model deserializers."""

    def deserialize(self, **kwargs):
        raise NotImplementedError


class CloudPickleModelSerializer(ModelSerializer):
    """Uses `Cloudpickle` to save model."""

    def __init__(self, model_file_suffix="pkl"):
        super().__init__(model_file_suffix=model_file_suffix)

    def serialize(self, estimator, model_path, **kwargs):
        """Uses `cloudpickle.dump` to save model. See https://docs.python.org/3/library/pickle.html#pickle.dump for more details.

        Args:
            estimator: The model to be saved.
            model_path: The file object or path of the model in which it is to be stored.
            kwargs:
                model_save: (dict, optional).
                    The dictionary where contains the availiable options to be passed to `cloudpickle.dump`.
        """
        cloudpickle_kwargs = kwargs.pop("model_save", {})
        with open(model_path, "wb") as f:
            cloudpickle.dump(estimator, f, **cloudpickle_kwargs)


class JobLibModelSerializer(ModelSerializer):
    """Uses `Joblib` to save model."""

    def __init__(self, model_file_suffix="joblib"):
        super().__init__(model_file_suffix=model_file_suffix)

    def serialize(self, estimator, model_path, **kwargs):
        """Uses `joblib.dump` to save model. See https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html for more details.

        Args:
            estimator: The model to be saved.
            model_path: The file object or path of the model in which it is to be stored.
            kwargs:
                model_save: (dict, optional).
                    The dictionary where contains the availiable options to be passed to `joblib.dump`.
        """
        joblib_kwargs = kwargs.pop("model_save", {})
        dump(estimator, model_path, **joblib_kwargs)


class SparkModelSerializer(ModelSerializer):
    """Save Spark Model."""

    def __init__(self, model_file_suffix=""):
        super().__init__(model_file_suffix=model_file_suffix)

    def serialize(self, estimator, model_path, **kwargs):
        estimator.write().overwrite().save(model_path)


class PyTorchModelSerializer(ModelSerializer):
    """Save PyTorch Model using torch.save(). See https://pytorch.org/docs/stable/generated/torch.save.html for more details."""

    def __init__(self, model_file_suffix="pt"):
        super().__init__(model_file_suffix=model_file_suffix)

    @runtime_dependency(module="torch", install_from=OptionalDependency.PYTORCH)
    def serialize(self, estimator, model_path, **kwarg):
        torch.save(estimator.state_dict(), model_path)


class TorchScriptModelSerializer(ModelSerializer):
    """Save PyTorch Model using torchscript. See https://pytorch.org/tutorials/beginner/saving_loading_models.html#export-load-model-in-torchscript-format for more details."""

    def __init__(self, model_file_suffix="pt"):
        super().__init__(model_file_suffix=model_file_suffix)

    @runtime_dependency(module="torch", install_from=OptionalDependency.PYTORCH)
    def serialize(self, estimator, model_path, **kwargs):
        compiled_model = torch.jit.script(estimator)
        torch.jit.save(compiled_model, model_path)


class LightGBMModelSerializer(ModelSerializer):
    """Save LightGBM Model through save_model into txt."""

    def __init__(self, model_file_suffix="txt"):
        super().__init__(model_file_suffix=model_file_suffix)

    def serialize(self, estimator, model_path, **kwargs):
        estimator.save_model(model_path)


class XgboostJsonModelSerializer(ModelSerializer):
    """Save Xgboost Model through xgboost.save_model into JSON."""

    def __init__(self, model_file_suffix="json"):
        super().__init__(model_file_suffix=model_file_suffix)

    def serialize(self, estimator, model_path, **kwargs):
        """Save Xgboost Model through xgboost.save_model .See
        https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.save_model
        for more details.

        Args:
            estimator: The model to be saved.
            model_path: The file object or path of the model in which it is to be stored.
        """
        estimator.save_model(model_path)


class XgboostTxtModelSerializer(ModelSerializer):
    """Save Xgboost Model through xgboost.save_model into txt."""

    def __init__(self, model_file_suffix="txt"):
        super().__init__(model_file_suffix=model_file_suffix)

    def serialize(self, estimator, model_path, **kwargs):
        """Save Xgboost Model through xgboost.save_model .See
        https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.save_model
        for more details.

        Args:
            estimator: The model to be saved.
            model_path: The file object or path of the model in which it is to be stored.
        """
        estimator.save_model(model_path)


class XgboostUbjModelSerializer(ModelSerializer):
    """Save Xgboost Model through xgboost.save_model into binary JSON."""

    def __init__(self, model_file_suffix="ubj"):
        super().__init__(model_file_suffix=model_file_suffix)

    def serialize(self, estimator, model_path, **kwargs):
        """Save Xgboost Model through xgboost.save_model .See
        https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.save_model
        for more details.

        Args:
            estimator: The model to be saved.
            model_path: The file object or path of the model in which it is to be stored.
        """
        estimator.save_model(model_path)


class TensorFlowModelSerializer(ModelSerializer):
    """Save Tensorflow Model."""

    def __init__(self, model_file_suffix="h5"):
        super().__init__(model_file_suffix=model_file_suffix)

    def serialize(self, estimator, model_path, **kwargs):
        estimator.save(model_path)


class HuggingFaceModelSerializer(ModelSerializer):
    """Save HuggingFace Pipeline."""

    def __init__(self, model_file_suffix=""):
        super().__init__(model_file_suffix=model_file_suffix)

    def serialize(self, estimator, model_path, **kwargs):
        estimator.save_pretrained(save_directory=model_path)
        estimator.model.config.use_pretrained_backbone = False
        estimator.model.config.save_pretrained(save_directory=model_path)


class OnnxModelSerializer(ModelSerializer):
    """Base class for creation of onnx converter for each model framework."""

    def __init__(self, model_file_suffix="onnx"):
        super().__init__(model_file_suffix=model_file_suffix)

    def serialize(
        self,
        estimator,
        model_path,
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
        """Save model into onnx format.

        Args:
            estimator: The model to be saved.
            model_path: The file object or path of the model in which it is to be stored.
            initial_types: (List[Tuple], optional)
                a python list. Each element is a tuple of a variable name and a data type.
            X_sample: (any, optional). Defaults to None.
                Contains model inputs such that model(X_sample) is a valid
                invocation of the model, used to valid model input type.
        """
        self.estimator = estimator
        onx = self._to_onnx(
            initial_types=initial_types,
            X_sample=X_sample,
            **kwargs,
        )
        with open(model_path, "wb") as f:
            f.write(onx.SerializeToString())

    def _to_onnx(
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
        raise NotImplementedError


class SklearnOnnxModelSerializer(OnnxModelSerializer):
    """Converts Skearn Model into Onnx."""

    def __init__(self):
        super().__init__()

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
    def _to_onnx(
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
            auto_generated_initial_types = self._generate_initial_types(X_sample)
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
    def _generate_initial_types(self, X_sample: Any) -> List:
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
        if self._is_all_numerical_array_dataframe(X_sample):
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
        elif self.is_either_numerical_or_string_dataframe(X_sample):
            # for dataframe and not all the columns are numerical, then generate
            # the input types of all the columns one by one.
            auto_generated_initial_types = []

            for i, col in X_sample.items():
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
                for _, col in data.items()
            ]
        )


class LightGBMOnnxModelSerializer(OnnxModelSerializer):
    """Converts LightGBM model into onnx format."""

    def __init__(self):
        super().__init__()

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
    def _to_onnx(
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
        Produces an equivalent ONNX model of the given LightGBM model.

        Parameters
        ----------
        initial_types: (List[Tuple], optional). Defaults to None.
            Each element is a tuple of a variable name and a type.
        X_sample: Union[Dict, str, List, np.ndarray, pd.core.series.Series, pd.core.frame.DataFrame,]. Defaults to None.
            Contains model inputs such that model(X_sample) is a valid invocation of the model.
            Used to generate initial_types.

        Returns
        ------
            An ONNX model (type: ModelProto) which is equivalent to the input LightGBM model.
        """
        auto_generated_initial_types = None
        if not initial_types:
            auto_generated_initial_types = self._generate_initial_types(X_sample)
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
    def _generate_initial_types(self, X_sample: Any) -> List:
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


class XgboostOnnxModelSerializer(OnnxModelSerializer):
    """Converts Xgboost model into onnx format."""

    def __init__(self):
        super().__init__()

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
    def _to_onnx(
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
            auto_generated_initial_types = self._generate_initial_types(X_sample)

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
    def _generate_initial_types(self, X_sample: Any) -> List:
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


class PytorchOnnxModelSerializer(OnnxModelSerializer):
    """Converts Pytorch model into onnx format."""

    def __init__(self):
        super().__init__()

    @runtime_dependency(module="torch", install_from=OptionalDependency.PYTORCH)
    def serialize(
        self,
        estimator,
        model_path: str,
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
        Exports the given Pytorch model into ONNX format.

        Parameters
        ----------
        path: str, default to None
            Path to save the serialized model.
        onnx_args: (tuple or torch.Tensor), default to None
            Contains model inputs such that model(onnx_args) is a valid
            invocation of the model. Can be structured either as: 1) ONLY A
            TUPLE OF ARGUMENTS; 2) A TENSOR; 3) A TUPLE OF ARGUMENTS ENDING
            WITH A DICTIONARY OF NAMED ARGUMENTS
        X_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]. Defaults to None.
            A sample of input data that will be used to generate input schema and detect onnx_args.
        kwargs:
            input_names: (List[str], optional). Defaults to ["input"].
                Names to assign to the input nodes of the graph, in order.
            output_names: (List[str], optional). Defaults to ["output"].
                Names to assign to the output nodes of the graph, in order.
            dynamic_axes: (dict, optional). Defaults to None.
                Specify axes of tensors as dynamic (i.e. known only at run-time).

        Returns
        -------
        None
            Nothing

        Raises
        ------
        AssertionError
            if onnx module is not support by the current version of torch
        ValueError
            if X_sample is not provided
            if path is not provided
        """
        onnx_args = kwargs.get("onnx_args", None)
        input_names = kwargs.get("input_names", ["input"])
        output_names = kwargs.get("output_names", ["output"])
        dynamic_axes = kwargs.get("dynamic_axes", None)

        assert hasattr(torch, "onnx"), (
            f"This version of pytorch {torch.__version__} does not appear to support onnx "
            "conversion."
        )

        if onnx_args is None:
            if X_sample is not None:
                logger.warning(
                    "Since `onnx_args` is not provided, `onnx_args` is "
                    "detected from `X_sample` to export pytorch model as onnx."
                )
                onnx_args = X_sample
            else:
                raise ValueError(
                    "`onnx_args` can not be detected. The parameter `onnx_args` must be provided to export pytorch model as onnx."
                )

        if not model_path:
            raise ValueError(
                "The parameter `model_path` must be provided to save the model file."
            )

        torch.onnx.export(
            estimator,
            args=onnx_args,
            f=model_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )


class TensorFlowOnnxModelSerializer(OnnxModelSerializer):
    """Converts Tensorflow model into onnx format."""

    def __init__(self):
        super().__init__()

    @runtime_dependency(module="tf2onnx", install_from=OptionalDependency.ONNX)
    @runtime_dependency(
        module="tensorflow",
        short_name="tf",
        install_from=OptionalDependency.TENSORFLOW,
    )
    def serialize(
        self,
        estimator,
        model_path: str = None,
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
        Exports the given Tensorflow model into ONNX format.

        Parameters
        ----------
        model_path: str, default to None
            Path to save the serialized model.
        X_sample: Union[list, tuple, pd.Series, np.ndarray, pd.DataFrame]. Defaults to None.
            A sample of input data that will be used to generate input schema and detect input_signature.


        Returns
        -------
        None
            Nothing

        Raises
        ------
        ValueError
            if model_path is not provided
        """
        opset_version = kwargs.get("opset_version", None)
        input_signature = kwargs.get("input_signature", None)

        if not model_path:
            raise ValueError(
                "The parameter `model_path` must be provided to save the model file."
            )
        if input_signature is None:
            if hasattr(estimator, "input_shape"):
                if not isinstance(estimator.input, list):
                    # single input
                    detected_input_signature = (
                        tf.TensorSpec(
                            estimator.input_shape,
                            dtype=estimator.input.dtype,
                            name="input",
                        ),
                    )
                else:
                    # multiple input
                    detected_input_signature = []
                    for i in range(len(estimator.input)):
                        detected_input_signature.append(
                            tf.TensorSpec(
                                estimator.input_shape[i],
                                dtype=estimator.input[i].dtype,
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
                    estimator,
                    input_signature=detected_input_signature,
                    opset=opset_version,
                    output_path=model_path,
                )
            except:
                raise ValueError(
                    "`input_signature` can not be autodetected. The parameter `input_signature` must be provided to export "
                    "tensorflow model as onnx."
                )

        else:
            tf2onnx.convert.from_keras(
                estimator,
                input_signature=input_signature,
                opset=opset_version,
                output_path=model_path,
            )


class OnnxModelSaveSERDE(OnnxModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_ONNX


class CloudpickleModelSaveSERDE(CloudPickleModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_CLOUDPICKLE


class JoblibModelSaveSERDE(JobLibModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_JOBLIB


class SparkModelSaveSERDE(SparkModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_SPARK


class HuggingFacePipelineSaveSERDE(HuggingFaceModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_HUGGINGFACE


class TorchScriptModelSaveSERDE(TorchScriptModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_TORHCSCRIPT


class PyTorchModelSaveSERDE(PyTorchModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_TORCH


class PyTorchOnnxModelSaveSERDE(PytorchOnnxModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_TORCH_ONNX


class TensorFlowModelSaveSERDE(TensorFlowModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_TF


class TensorFlowOnnxModelSaveSERDE(TensorFlowOnnxModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_TF_ONNX


class SklearnOnnxModelSaveSERDE(SklearnOnnxModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_SKLEARN_ONNX


class LightGBMModelSaveSERDE(LightGBMModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_LIGHTGBM


class LightGBMOnnxModelSaveSERDE(LightGBMOnnxModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_LIGHTGBM_ONNX


class XgboostJsonModelSaveSERDE(XgboostJsonModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_XGBOOST


class XgboostUbjModelSaveSERDE(XgboostUbjModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_XGBOOST_UBJ


class XgboostTxtModelSaveSERDE(XgboostTxtModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_XGBOOST_TXT


class XgboostOnnxModelSaveSERDE(XgboostOnnxModelSerializer, ModelDeserializer):
    name = MODEL_SERIALIZATION_TYPE_XGBOOST_ONNX


class ModelSerializerFactory:
    """Model Serializer Factory.

    Returns
    -------
    model_save_serde: Intance of `ads.model.SERDE`".
    """

    _factory = {}
    _factory[MODEL_SERIALIZATION_TYPE_CLOUDPICKLE] = CloudpickleModelSaveSERDE
    _factory[MODEL_SERIALIZATION_TYPE_ONNX] = OnnxModelSaveSERDE
    _factory[MODEL_SERIALIZATION_TYPE_TORHCSCRIPT] = TorchScriptModelSaveSERDE
    _factory[MODEL_SERIALIZATION_TYPE_TORCH] = PyTorchModelSaveSERDE
    _factory[MODEL_SERIALIZATION_TYPE_TORCH_ONNX] = PyTorchOnnxModelSaveSERDE
    _factory[MODEL_SERIALIZATION_TYPE_TF] = TensorFlowModelSaveSERDE
    _factory[MODEL_SERIALIZATION_TYPE_TF_ONNX] = TensorFlowOnnxModelSaveSERDE
    _factory[MODEL_SERIALIZATION_TYPE_JOBLIB] = JoblibModelSaveSERDE
    _factory[MODEL_SERIALIZATION_TYPE_SKLEARN_ONNX] = SklearnOnnxModelSaveSERDE
    _factory[MODEL_SERIALIZATION_TYPE_LIGHTGBM] = LightGBMModelSaveSERDE
    _factory[MODEL_SERIALIZATION_TYPE_LIGHTGBM_ONNX] = LightGBMOnnxModelSaveSERDE
    _factory[MODEL_SERIALIZATION_TYPE_XGBOOST] = XgboostJsonModelSaveSERDE
    _factory[MODEL_SERIALIZATION_TYPE_XGBOOST_UBJ] = XgboostUbjModelSaveSERDE
    _factory[MODEL_SERIALIZATION_TYPE_XGBOOST_TXT] = XgboostTxtModelSaveSERDE
    _factory[MODEL_SERIALIZATION_TYPE_XGBOOST_ONNX] = XgboostOnnxModelSaveSERDE
    _factory[MODEL_SERIALIZATION_TYPE_SPARK] = SparkModelSaveSERDE
    _factory[MODEL_SERIALIZATION_TYPE_HUGGINGFACE] = HuggingFacePipelineSaveSERDE

    @classmethod
    def get(cls, se: str):
        serde = cls._factory.get(se, None)
        if serde:
            return serde()
        else:
            raise ValueError(
                f"This {se} format is not supported."
                f"Currently support the following format: {SUPPORTED_MODEL_SERIALIZERS}."
            )
