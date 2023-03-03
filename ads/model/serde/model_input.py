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
from pickle import UnpicklingError
from typing import Dict, List, Union

SUPPORTED_MODEL_INPUT_SERIALIZERS = ["json", "cloudpickle", "spark"]


class ModelInputSerializerType:
    JSON = "json"
    CLOUDPICKLE = "cloudpickle"


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

    >>> # `torch.Tensor` will be converted to base64 encoded string,
    >>> # while `data_type` will record its original type: `torch.Tensor`
    >>> import torch
    >>> tt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    >>> serialized_data = JsonModelInputSerializer().serialize(data=tt)
    >>> serialized_data
    {
    'data': 'UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaW
    lpaWlpaWlpaWoACY3RvcmNoLl91dGlscwpfcmVidWlsZF90ZW5zb3JfdjIKcQAoKFgHAAAAc3RvcmFnZXEB
    Y3RvcmNoCkxvbmdTdG9yYWdlCnECWAEAAAAwcQNYAwAAAGNwdXEESwZ0cQVRSwBLAksDhnEGSwNLAYZxB4l
    jY29sbGVjdGlvbnMKT3JkZXJlZERpY3QKcQgpUnEJdHEKUnELLlBLBwim2iAhmQAAAJkAAABQSwMEAAAICA
    AAAAAAAAAAAAAAAAAAAAAAAA4AKwBhcmNoaXZlL2RhdGEvMEZCJwBaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWl
    paWlpaWlpaWlpaWlpaWloBAAAAAAAAAAIAAAAAAAAAAwAAAAAAAAAEAAAAAAAAAAUAAAAAAAAABgAAAAAAA
    ABQSwcI9z/uVjAAAAAwAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAPABMAYXJjaGl2ZS92ZXJzaW9u
    RkIPAFpaWlpaWlpaWlpaWlpaWjMKUEsHCNGeZ1UCAAAAAgAAAFBLAQIAAAAACAgAAAAAAACm2iAhmQAAAJk
    AAAAQAAAAAAAAAAAAAAAAAAAAAABhcmNoaXZlL2RhdGEucGtsUEsBAgAAAAAICAAAAAAAAPc/7lYwAAAAMA
    AAAA4AAAAAAAAAAAAAAAAA6QAAAGFyY2hpdmUvZGF0YS8wUEsBAgAAAAAICAAAAAAAANGeZ1UCAAAAAgAAA
    A8AAAAAAAAAAAAAAAAAgAEAAGFyY2hpdmUvdmVyc2lvblBLBgYsAAAAAAAAAB4DLQAAAAAAAAAAAAMAAAAA
    AAAAAwAAAAAAAAC3AAAAAAAAANIBAAAAAAAAUEsGBwAAAACJAgAAAAAAAAEAAABQSwUGAAAAAAMAAwC3AAA
    A0gEAAAAA',
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
        buffer = BytesIO()
        torch.save(data, buffer)
        data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return data

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
            np_bytes = BytesIO()
            np.save(np_bytes, data, allow_pickle=True)
            data = base64.b64encode(np_bytes.getvalue()).decode("utf-8")
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
    """Serialize data of various formats to bytes."""

    def __init__(self):
        super().__init__()

    def serialize(self, data):
        """Serialize data into bytes.

        Parameters
        ----------
        data (object): Data to be serialized.

        Returns
        -------
        object: Serialized data used for a request.
        """
        serialized_data = cloudpickle.dumps(data)
        return serialized_data


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
            load_bytes = BytesIO(base64.b64decode(json_data.encode("utf-8")))
            return np.load(load_bytes, allow_pickle=True)
        if "pandas.core.series.Series" in data_type:
            return pd.Series(json_data)
        if "pandas.core.frame.DataFrame" in data_type or isinstance(json_data, str):
            return pd.read_json(json_data, dtype=self._fetch_data_type_from_schema())
        if isinstance(json_data, dict):
            return pd.DataFrame.from_dict(json_data)

        return json_data

    @runtime_dependency(module="torch", install_from=OptionalDependency.PYTORCH)
    def _load_torch_tensor(self, data):
        load_bytes = BytesIO(base64.b64decode(data.encode("utf-8")))
        return torch.load(load_bytes)

    @runtime_dependency(
        module="tensorflow",
        short_name="tf",
        install_from=OptionalDependency.TENSORFLOW,
    )
    def _load_tf_tensor(self, data):
        load_bytes = BytesIO(base64.b64decode(data.encode("utf-8")))
        return tf.convert_to_tensor(np.load(load_bytes, allow_pickle=True))


class SparkModelInputDeserializer(ModelInputDeserializer):
    def __init__(self, name="spark"):
        super().__init__(name=name)

    def deserialize(data):
        """
        Not implement. See spark template.
        """
        pass


class CloudpickleModelInputDeserializer(ModelInputDeserializer):
    """Use cloudpickle to deserialize data into its original type."""

    def __init__(self, name="cloudpickle"):
        super().__init__(name=name)

    def deserialize(self, data):
        """Deserialize data into its original type.

        Parameters
        ----------
        data (object): Data to be deserialized.

        Returns
        -------
        object: deserialized data used for a prediction.
        """
        deserialized_data = data
        try:
            deserialized_data = cloudpickle.loads(data)
        except TypeError:
            pass
        except UnpicklingError:
            logger.warning(
                "bytes are passed directly to the model. If the model expects a specific data format, you need to write the conversion logic in `deserialize()` yourself."
            )
        return deserialized_data


class JsonModelInputSERDE(JsonModelInputSerializer, JsonModelInputDeserializer):
    name = "json"


class CloudpickleModelInputSERDE(
    CloudpickleModelInputSerializer, CloudpickleModelInputDeserializer
):
    name = "cloudpickle"


class SparkModelInputSERDE(SparkModelInputSerializer, SparkModelInputDeserializer):
    name = "spark"


class ModelInputSerializerFactory:
    """Data Serializer Factory.

    Examples
    --------
    >>> serializer, deserializer = ModelInputSerializerFactory.get("cloudpickle")
    """

    _factory = {}
    _factory["json"] = JsonModelInputSERDE
    _factory["cloudpickle"] = CloudpickleModelInputSERDE
    _factory["spark"] = SparkModelInputSERDE

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
