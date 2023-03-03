#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import abc


class Serializer:
    """Abstract base class for creation of new serializers."""

    def serialize(self, **kwargs):
        """Serialize data/model into specific type.

        Returns
        -------
        object: Serialized data/model.
        """
        raise NotImplementedError("`serialize()` method needs to be implemented.")


class Deserializer:
    """Abstract base class for creation of new deserializers."""

    def deserialize(self, **kwargs):
        """Deserialize data/model into original type.

        Returns
        -------
        object: deserialized data/model.
        """
        raise NotImplementedError("`deserialize()` method needs to be implemented.")


class SERDE(Serializer, Deserializer):
    """A layer contains two groups which can interact with each other to serialize and
    deserialize supported data structure using supported data format.
    """

    name = ""
