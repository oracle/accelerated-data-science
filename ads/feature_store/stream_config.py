#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from copy import deepcopy
from typing import Dict, List

from ads.jobs.builders.base import Builder


class StreamConfig(Builder):
    """Sets the Statistics Config Details.
    Methods
    -------
        with_enabled(self, enabled: bool) -> "StatisticsConfig"
        Sets True/False for enabled
        with_columns(self, columns: List[str]) -> "StatisticsConfig"
        Sets the column names for the statistics config
    """
    CONST_STREAMING_ENDPOINT = "streamingEndpoint"
    CONST_STREAM_POOL_ID = "streamPoolId"
    CONST_STREAM_NAME = "streamName"
    CONST_AVRO_SCHEMA = "avroSchema"

    attribute_map = {
        CONST_STREAMING_ENDPOINT: "streaming_endpoint",
        CONST_STREAM_POOL_ID: "stream_pool_id",
        CONST_STREAM_NAME: "stream_name",
        CONST_AVRO_SCHEMA: "avro_schema"
    }

    def __init__(self, stream_pool_id: str = None, stream_name: str = None, avro_schema: str = None) -> None:
        super().__init__()
        self.with_stream_pool_id(stream_pool_id)
        self.with_stream_name(stream_name)
        self.with_avro_schema(avro_schema)

    @property
    def stream_name(self) -> str:
        return self.get_spec(self.CONST_STREAM_NAME)

    @stream_name.setter
    def stream_name(self, stream_name: str):
        self.with_stream_name(stream_name)

    def with_stream_name(self, stream_name: str) -> "StreamConfig":
        """Sets True/False for enabled

        Parameters
        ----------
        stream_name

        Returns
        -------
        StreamConfig
            The StreamConfig instance (self)
        """
        return self.set_spec(self.CONST_STREAM_NAME, stream_name)

    @property
    def streaming_endpoint(self) -> str:
        return self.get_spec(self.CONST_STREAMING_ENDPOINT)

    @streaming_endpoint.setter
    def streaming_endpoint(self, streaming_endpoint: str):
        self.with_streaming_endpoint(streaming_endpoint)

    def with_streaming_endpoint(self, streaming_endpoint: str) -> "StreamConfig":
        """Sets True/False for enabled

        Parameters
        ----------

        Returns
        -------
        StreamConfig
            The StreamConfig instance (self)
        """
        return self.set_spec(self.CONST_STREAMING_ENDPOINT, streaming_endpoint)
    @property
    def stream_pool_id(self) -> str:
        return self.get_spec(self.CONST_STREAM_POOL_ID)

    @stream_pool_id.setter
    def stream_pool_id(self, stream_pool_id: str):
        self.with_stream_pool_id(stream_pool_id)

    def with_stream_pool_id(self, stream_pool_id: str) -> "StreamConfig":
        """Sets True/False for enabled

        Parameters
        ----------
        stream_pool_id

        Returns
        -------
        StreamConfig
            The StreamConfig instance (self)
        """
        return self.set_spec(self.CONST_STREAM_POOL_ID, stream_pool_id)

    @property
    def avro_schema(self) -> str:
        return self.get_spec(self.CONST_AVRO_SCHEMA)

    @avro_schema.setter
    def avro_schema(self, avro_schema: str):
        self.with_avro_schema(avro_schema)

    def with_avro_schema(self, avro_schema: str) -> "StreamConfig":
        """Sets True/False for enabled

        Parameters
        ----------
        avro_schema

        Returns
        -------
        StreamConfig
            The StreamConfig instance (self)
        """
        return self.set_spec(self.CONST_AVRO_SCHEMA, avro_schema)

    @property
    def kind(self) -> str:
        """The kind of the object as showing in a YAML."""
        return "statistics_config"

    def to_dict(self) -> Dict:
        """Serializes rule to a dictionary.

        Returns
        -------
        dict
            The rule resource serialized as a dictionary.
        """

        spec = deepcopy(self._spec)
        return spec
