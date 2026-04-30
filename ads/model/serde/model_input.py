#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import cloudpickle
import numpy as np
import pandas as pd
import base64
import os
import json
from io import BytesIO
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common import logger
from ads.model.serde.common import Serializer, Deserializer
from functools import lru_cache
from typing import Dict, List, Union

SUPPORTED_MODEL_INPUT_SERIALIZERS = ["json", "cloudpickle", "spark", "huggingface"]
_HF_TYPE_KEY = "__ads_huggingface_type__"
_TORCH_TENSOR_KEY = "__ads_torch_tensor__"
_TORCH_TENSOR_FORMAT_VERSION = 1


def _reject_object_array(data):
    if getattr(data, "dtype", None) is not None and data.dtype.hasobject:
        raise TypeError(
            "Object dtype arrays are not supported by the JSON model input "
            "serializer. Use numeric, string, boolean, or datetime arrays, or "
            "provide a custom serializer for this input."
        )


def _serialize_numpy_array(data):
    _reject_object_array(data)
    np_bytes = BytesIO()
    np.save(np_bytes, data, allow_pickle=False)
    return base64.b64encode(np_bytes.getvalue()).decode("utf-8")


def _deserialize_numpy_array(data):
    load_bytes = BytesIO(base64.b64decode(data.encode("utf-8")))
    loaded = np.load(load_bytes, allow_pickle=False)
    _reject_object_array(loaded)
    return loaded


class ModelInputSerializerType:
    JSON = "json"
    CLOUDPICKLE = "cloudpickle"
    HUGGINGFACE = "huggingface"


class SparkModelInputSerializerType:
    SPARK = "spark"


class ModelInputSerializer(Serializer):
    """Abstract base class for creation of new data serializers."""

    def __init__(self):
        super().__init__()

    def serialize(self, data):
        return data


class JsonModelInputSerializer(ModelInputSerializer):
    """
    ADS data serializer. Serialize data of various formats to into a
    dictionary containing serialized input data and original data type
    information.


    Examples
    --------
    >>> from ads.model.serde.model_input import JsonModelInputSerializer

    >>> # numpy array will be converted to base64 encoded string,
    >>> # while `data_type` will record its original type: `numpy.ndarray`
    >>> import numpy as np
    >>> input_data = np.array([1, 2, 3])
    >>> serialized_data = JsonModelInputSerializer().serialize(data=input_data)
    >>> serialized_data
    {
    'data': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZS
    wgJ3NoYXBlJzogKDMsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI
    CAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAAIAAAAAAAAAAwAAAAAAAAA=',
    'data_type': "<class 'numpy.ndarray'>"
    }

    >>> # `pd.core.frame.DataFrame` will be converted to json by `.to_json()`
    >>> # while `data_type` will record its original type: `pandas.core.frame.DataFrame`
    >>> import pandas as pd
    >>> df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    >>> serialized_data = JsonModelInputSerializer().serialize(data=df)
    >>> serialized_data
    {
    'data': '{"col1":{"0":1,"1":2},"col2":{"0":3,"1":4}}',
    'data_type': "<class 'pandas.core.frame.DataFrame'>"
    }

    >>> # `pandas.core.series.Series` will be converted to list by `.tolist()`
    >>> # while `data_type` will record its original type: `pandas.core.series.Series`
    >>> ser = pd.Series(data={'a': 1, 'b': 2, 'c': 3}, index=['a', 'b', 'c'])
    >>> serialized_data = JsonModelInputSerializer().serialize(data=ser)
    >>> serialized_data
    {
    'data': [1, 2, 3],
    'data_type': "<class 'pandas.core.series.Series'>"
    }

    >>> # `torch.Tensor` will be converted to a JSON-compatible dictionary,
    >>> # while `data_type` will record its original type: `torch.Tensor`
    >>> import torch
    >>> tt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    >>> serialized_data = JsonModelInputSerializer().serialize(data=tt)
    >>> serialized_data
    {
    'data': {
        '__ads_torch_tensor__': 1,
        'data': [[1, 2, 3], [4, 5, 6]],
        'dtype': 'torch.int64',
        'shape': [2, 3],
        'device': 'cpu',
    },
    'data_type': "<class 'torch.Tensor'>"
    }

    >>> # `tensorflow.Tensor` will be converted to base64 encoded string,
    >>> # while `data_type` will record its original type: `tensorflow.python.framework.ops.EagerTensor`.
    >>> import torch
    >>> c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    >>> serialized_data = JsonModelInputSerializer().serialize(data=c)
    >>> serialized_data
    {
    'data': 'k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXB
    lJzogKDIsIDIpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA
    gICAgICAgIAoAAIA/AAAAQAAAQEAAAIBA',
    'data_type': "<class 'tensorflow.python.framework.ops.EagerTensor'>"
    }

    >>> # dict, str, list, tuple will be saved as orignal type
    >>> # and `data_type` will record its type.
    >>> mystring = "this is a string."
    >>> serialized_data = JsonModelInputSerializer().serialize(data=mystring)
    >>> serialized_data
    {
    'data': 'this is a string.',
    'data_type': "<class 'str'>"
    }
    """

    def __init__(self):
        super().__init__()

    @runtime_dependency(module="torch", install_from=OptionalDependency.PYTORCH)
    def _convert_torch_tensor(self, data):
        tensor = data.detach().cpu()
        if tensor.is_complex():
            raise TypeError(
                "Complex torch.Tensor inputs are not supported by the JSON "
                "model input serializer. Provide a custom serializer for this "
                "input."
            )
        return {
            _TORCH_TENSOR_KEY: _TORCH_TENSOR_FORMAT_VERSION,
            "data": tensor.tolist(),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "device": str(data.device),
        }

    @runtime_dependency(
        module="tensorflow",
        short_name="tf",
        install_from=OptionalDependency.TENSORFLOW,
    )
    def _convert_tf_tensor(self, data):
        data = data.numpy()
        return data

    def serialize(
        self,
        data: Union[
            Dict,
            str,
            List,
            np.ndarray,
            pd.core.series.Series,
            pd.core.frame.DataFrame,
        ],
    ):
        """Serialize data into a dictionary containing serialized input
        data and original data type information.

        Parameters
        ----------
        data: Union[Dict, str, list, numpy.ndarray, pd.core.series.Series,
        pd.core.frame.DataFrame, bytes]
            Data expected by the model deployment predict API.

        Returns
        -------
        Dict
            A dictionary containing serialized input data and original data type
            information.

        Raises
        ------
        TypeError
            if provided data type is not supported.
        """
        data_type = str(type(data))
        if "torch.Tensor" in data_type:
            data = self._convert_torch_tensor(data)
        if "tensorflow.python.framework.ops.EagerTensor" in data_type:
            data = self._convert_tf_tensor(data)

        if isinstance(data, np.ndarray):
            data = _serialize_numpy_array(data)
        elif isinstance(data, pd.core.series.Series):
            data = data.tolist()
        elif isinstance(data, pd.core.frame.DataFrame):
            data = data.to_json()
        elif (
            isinstance(data, dict)
            or isinstance(data, str)
            or isinstance(data, list)
            or isinstance(data, tuple)
            or isinstance(data, bytes)
        ):
            pass
        else:
            raise TypeError(
                "The supported data types are Dict, str, list, bytes, tuple, "
                "numpy.ndarray, pd.core.series.Series, tf.Tensor, torch.Tensor, "
                "pd.core.frame.DataFrame. Please convert to the supported data "
                "types first. "
            )

        data_dict = {
            "data": data,
            "data_type": data_type,
        }
        return data_dict


class CloudpickleModelInputSerializer(ModelInputSerializer):
    """Serialize data of various formats to bytes for custom input workflows.

    ADS-generated scoring artifacts do not deserialize cloudpickle request
    payloads. This serializer is kept for existing deployments or custom
    scoring code that handles the payload explicitly.
    """

    def __init__(self):
        super().__init__()

    def serialize(self, data):
        """Serialize data into bytes.

        Parameters
        ----------
        data (object): Data to be serialized.

        Returns
        -------
        object: Serialized data used for explicit legacy input mode.
        """
        serialized_data = cloudpickle.dumps(data)
        return serialized_data


class HuggingFaceModelInputSerializer(ModelInputSerializer):
    """Serialize HuggingFace pipeline inputs into JSON-compatible values."""

    def __init__(self):
        super().__init__()

    def serialize(self, data):
        """Serialize HuggingFace pipeline inputs without pickle payloads."""
        return {
            "data": self._serialize_value(data),
            "data_type": "huggingface",
        }

    def _serialize_value(self, value):
        data_type = str(type(value))
        if data_type.startswith("<class 'PIL."):
            from PIL import Image

            if isinstance(value, Image.Image):
                image_format = value.format or "PNG"
                image_bytes = BytesIO()
                value.save(image_bytes, format=image_format)
                return {
                    _HF_TYPE_KEY: "pil_image",
                    "format": image_format,
                    "data": base64.b64encode(image_bytes.getvalue()).decode("utf-8"),
                }

        if isinstance(value, bytes):
            return {
                _HF_TYPE_KEY: "bytes",
                "data": base64.b64encode(value).decode("utf-8"),
            }
        if isinstance(value, tuple):
            return {
                _HF_TYPE_KEY: "tuple",
                "items": [self._serialize_value(item) for item in value],
            }
        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._serialize_value(item) for key, item in value.items()}

        try:
            json.dumps(value)
        except TypeError as exc:
            raise TypeError(
                f"Data type {type(value)} is not supported by the HuggingFace input serializer. "
                "Use JSON-compatible values, PIL images, bytes, lists, tuples, or dictionaries."
            ) from exc
        return value


class SparkModelInputSerializer(JsonModelInputSerializer):
    """[An internal class]
    Defines the contract for input data to spark pipeline models.

    """

    def __init__(self):
        super().__init__()

    @runtime_dependency(
        module="pyspark",
        short_name="sql",
        object="sql",
        install_from=OptionalDependency.SPARK,
    )
    def serialize(
        self,
        data: Union[
            Dict,
            str,
            List,
            np.ndarray,
            pd.core.series.Series,
            pd.core.frame.DataFrame,
        ],
    ):
        """
        Parameters
        ----------
        data: Union[Dict, str, list, numpy.ndarray, pd.core.series.Series,
        pd.core.frame.DataFrame]
            Data expected by the model deployment predict API.

        """
        data, _, _ = self._serialize_via_spark(data)
        if isinstance(data, sql.DataFrame):
            data = data.toJSON().collect()

        try:
            data = super().serialize(data=data)
        except:
            raise TypeError(
                f"Data type: {type(data)} unsupported. Please use `pyspark.sql.DataFrame`, `pyspark.pandas.DataFrame`, `pandas.DataFrame`."
            )
        return data

    @runtime_dependency(
        module="pyspark",
        short_name="sql",
        object="sql",
        install_from=OptionalDependency.SPARK,
    )
    def _serialize_via_spark(self, data):
        """
        If data is either a spark SQLDataFrames and spark.pandas dataframe/series
            Return pandas version and data type of original
        Else
            Return data and None
        """
        try:  # runtime_dependency could not import this for unknown reason
            import pyspark.pandas as ps

            ps_available = True
        except:
            ps_available = False

        def _get_or_create_spark_session():
            return sql.SparkSession.builder.appName(
                "Convert pandas to spark"
            ).getOrCreate()

        if isinstance(data, sql.DataFrame):
            data_type = type(data)
        elif ps_available and (
            isinstance(data, ps.DataFrame) or isinstance(data, ps.Series)
        ):
            data_type = type(data)
            data = data.to_spark()
        elif isinstance(data, sql.types.Row):
            spark_session = _get_or_create_spark_session()
            data = spark_session.createDataFrame(data)
            data_type = type(data)
        elif isinstance(data, pd.core.frame.DataFrame):
            data_type = type(data)
            spark_session = _get_or_create_spark_session()
            data = spark_session.createDataFrame(data)
        elif isinstance(data, list):
            if not len(data):
                raise TypeError(
                    f"Data cannot be empty. Provided data parameter is: {data}"
                )
            if isinstance(data[0], sql.types.Row):
                spark_session = _get_or_create_spark_session()
                data = spark_session.createDataFrame(data)
                data_type = type(data)
            else:
                logger.warn(
                    f"ADS does not serialize data type: {type(data)} for Spark Models. User should proceed at their own risk. ADS supported data types are: `pyspark.sql.DataFrame`, `pandas.DataFrame`, and `pyspark.pandas.DataFrame`."
                )
                return data, type(data), None
        else:
            logger.warn(
                f"ADS does not serialize data type: {type(data)} for Spark Models. User should proceed at their own risk. ADS supported data types are: `pyspark.sql.DataFrame`, `pandas.DataFrame`, and `pyspark.pandas.DataFrame`."
            )
            return data, type(data), None
        return data, data_type, data.schema


class ModelInputDeserializer(Deserializer):
    """Abstract base class for creation of new data deserializers."""

    def __init__(self, name="customized"):
        super().__init__()
        self.name = name

    def deserialize(self, data):
        return data


class JsonModelInputDeserializer(ModelInputDeserializer):
    """ADS data deserializer. Deserialize data to into its original type."""

    def __init__(self, name="json"):
        super().__init__(name=name)

    @lru_cache(maxsize=1)
    def _fetch_data_type_from_schema(
        input_schema_path=os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "input_schema.json"
        )
    ):
        """
        Returns data type information fetch from input_schema.json.

        Parameters
        ----------
        input_schema_path: path of input schema.

        Returns
        -------
        data_type: data type fetch from input_schema.json.

        """
        data_type = {}
        if os.path.exists(input_schema_path):
            schema = json.load(open(input_schema_path))
            for col in schema["schema"]:
                data_type[col["name"]] = col["dtype"]
        else:
            print(
                "input_schema has to be passed in in order to recover the same data type. pass `X_sample` in `ads.model.framework.pytorch_model.PyTorchModel.prepare` function to generate the input_schema. Otherwise, the data type might be changed after serialization/deserialization."
            )
        return data_type

    def deserialize(self, data: dict):
        """Deserialize data into its original type.

        Parameters
        ----------
        data: (dict)

        Returns
        -------
        objects: Deserialized data used for a prediction.

        Raises
        ------
        TypeError
            if provided data type is not supported.
        """
        if isinstance(data, bytes):
            logger.warning(
                "bytes are passed directly to the model. If the model expects a specific data format, you need to write the conversion logic in `deserialize()` yourself."
            )
            return data

        data_type = data.get("data_type", "") if isinstance(data, dict) else ""
        json_data = data.get("data", data) if isinstance(data, dict) else data
        if "torch.Tensor" in data_type:
            return self._load_torch_tensor(json_data)
        if "tensorflow.python.framework.ops.EagerTensor" in data_type:
            return self._load_tf_tensor(json_data)
        if "numpy.ndarray" in data_type:
            return _deserialize_numpy_array(json_data)
        if "pandas.core.series.Series" in data_type:
            return pd.Series(json_data)
        if "pandas.core.frame.DataFrame" in data_type or isinstance(json_data, str):
            return pd.read_json(json_data, dtype=self._fetch_data_type_from_schema())
        if isinstance(json_data, dict):
            return pd.DataFrame.from_dict(json_data)

        return json_data

    @runtime_dependency(module="torch", install_from=OptionalDependency.PYTORCH)
    def _load_torch_tensor(self, data):
        if not isinstance(data, dict) or data.get(_TORCH_TENSOR_KEY) != 1:
            raise TypeError(
                "Torch tensor inputs must use the structured JSON tensor "
                "format generated by `JsonModelInputSerializer`."
            )

        dtype_name = data.get("dtype")
        if not isinstance(dtype_name, str) or not dtype_name.startswith("torch."):
            raise TypeError("Torch tensor input payload is missing a valid dtype.")

        dtype = getattr(torch, dtype_name.split(".", 1)[1], None)
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Unsupported torch tensor dtype: {dtype_name}.")

        tensor = torch.tensor(data.get("data"), dtype=dtype)
        shape = data.get("shape")
        if shape is not None:
            tensor = tensor.reshape(tuple(shape))
        return tensor

    @runtime_dependency(
        module="tensorflow",
        short_name="tf",
        install_from=OptionalDependency.TENSORFLOW,
    )
    def _load_tf_tensor(self, data):
        return tf.convert_to_tensor(_deserialize_numpy_array(data))


class SparkModelInputDeserializer(ModelInputDeserializer):
    def __init__(self, name="spark"):
        super().__init__(name=name)

    def deserialize(data):
        """
        Not implement. See spark template.
        """
        pass


class CloudpickleModelInputDeserializer(ModelInputDeserializer):
    """Reject cloudpickle request payload deserialization in ADS-owned code."""

    def __init__(self, name="cloudpickle"):
        super().__init__(name=name)

    def deserialize(self, data):
        """Reject cloudpickle request payload deserialization.

        Parameters
        ----------
        data (object): Data to be deserialized.
        """
        raise RuntimeError(
            "ADS does not deserialize cloudpickle request payloads. Use a "
            "JSON-compatible input serializer, or provide a custom `score.py` "
            "for trusted clients that require a different request format."
        )


class HuggingFaceModelInputDeserializer(ModelInputDeserializer):
    """Deserialize HuggingFace JSON-compatible pipeline inputs."""

    def __init__(self, name="huggingface"):
        super().__init__(name=name)

    def deserialize(self, data):
        json_data = data.get("data", data) if isinstance(data, dict) else data
        return self._deserialize_value(json_data)

    def _deserialize_value(self, value):
        if isinstance(value, dict):
            value_type = value.get(_HF_TYPE_KEY)
            if value_type == "pil_image":
                from PIL import Image

                image_bytes = BytesIO(base64.b64decode(value["data"].encode("utf-8")))
                return Image.open(image_bytes).copy()
            if value_type == "bytes":
                return base64.b64decode(value["data"].encode("utf-8"))
            if value_type == "tuple":
                return tuple(self._deserialize_value(item) for item in value["items"])
            return {key: self._deserialize_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._deserialize_value(item) for item in value]
        return value


class JsonModelInputSERDE(JsonModelInputSerializer, JsonModelInputDeserializer):
    name = "json"


class CloudpickleModelInputSERDE(
    CloudpickleModelInputSerializer, CloudpickleModelInputDeserializer
):
    name = "cloudpickle"


class HuggingFaceModelInputSERDE(
    HuggingFaceModelInputSerializer, HuggingFaceModelInputDeserializer
):
    name = "huggingface"


class SparkModelInputSERDE(SparkModelInputSerializer, SparkModelInputDeserializer):
    name = "spark"


class ModelInputSerializerFactory:
    """Data Serializer Factory.

    Examples
    --------
    >>> serializer, deserializer = ModelInputSerializerFactory.get("json")
    """

    _factory = {}
    _factory["json"] = JsonModelInputSERDE
    _factory["cloudpickle"] = CloudpickleModelInputSERDE
    _factory["spark"] = SparkModelInputSERDE
    _factory["huggingface"] = HuggingFaceModelInputSERDE

    @classmethod
    def get(cls, se: str = "json"):
        """Gets data serializer and corresponding deserializer.

        Parameters
        ----------
        se (str):
            The name of the required serializer.

        Raises
        ------
        ValueError:
            Raises when input is unsupported format.

        Returns
        -------
        serde (ads.model.serde.common.SERDE):
            Intance of `ads.model.serde.common.SERDE".
        """
        serde_cls = cls._factory.get(se, None)
        if serde_cls:
            return serde_cls()
        else:
            raise ValueError(
                f"This {se} format is not supported."
                f"Currently support the following format: {SUPPORTED_MODEL_INPUT_SERIALIZERS}."
            )
