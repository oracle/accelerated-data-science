#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import absolute_import, print_function

import base64
from io import BytesIO
from typing import Dict, List, Union
import requests
import json
import numpy as np
import pandas as pd

DEFAULT_CONTENT_TYPE_JSON = "application/json"
DEFAULT_CONTENT_TYPE_BYTES = "application/octet-stream"


class InputDataSerializer:
    """[An internal class]
    Defines the contract for input data

    """

    def __init__(
        self,
        data: Union[
            Dict,
            str,
            List,
            np.ndarray,
            pd.core.series.Series,
            pd.core.frame.DataFrame,
        ],
        data_type=None,
    ):
        """
        Parameters
        ----------
        data: Union[Dict, str, list, numpy.ndarray, pd.core.series.Series,
        pd.core.frame.DataFrame]
            Data expected by the model deployment predict API.
        data_type: Any, defaults to None.
            Type of the data. If not provided, it will be checked against data.

        """
        if not data_type:
            data_type = type(data)
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
                "The supported data types are Dict, str, list, "
                "numpy.ndarray, pd.core.series.Series, "
                "pd.core.frame.DataFrame, bytes. Please "
                "convert to the supported data types first. "
            )

        self._data = data
        self._data_type = str(data_type)

    @property
    def data(self):
        return self._data

    @property
    def data_type(self):
        return self._data_type

    def to_dict(self):
        return {
            "data": self._data,
            "data_type": self._data_type,
        }

    def is_bytes(self):
        return "bytes" in self.data_type

    def send(self, endpoint: str, dry_run: bool = False, **kwargs):
        headers = dict()
        if self.is_bytes():
            headers["Content-Type"] = (
                kwargs.get("content_type") or DEFAULT_CONTENT_TYPE_BYTES
            )
            request_kwargs = {"data": self.data}  # should pass bytes when using data
        else:
            headers["Content-Type"] = (
                kwargs.get("content_type") or DEFAULT_CONTENT_TYPE_JSON
            )
            request_kwargs = {"json": self.to_dict()}
        request_kwargs["headers"] = headers

        if dry_run:
            request_kwargs["headers"]["Accept"] = "*/*"
            req = requests.Request("POST", endpoint, **request_kwargs).prepare()
            if self.is_bytes():
                return req.body
            return json.loads(req.body)
        else:
            request_kwargs["auth"] = kwargs.get("signer")
            return requests.post(endpoint, **request_kwargs).json()
