#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/from oci.decorators import init_model_state_from_kwargs

from oci.decorators import init_model_state_from_kwargs
from oci.util import formatted_flat_dict

BEARER_TOKEN_USERNAME = "BEARER_TOKEN"


@init_model_state_from_kwargs
class BearerToken(object):
    """
    List container image results.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new ContainerImageCollection object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param items:
            The value to assign to the items property of this ContainerImageCollection.
        :type items: list[oci.artifacts.models.ContainerImageSummary]

        :param remaining_items_count:
            The value to assign to the remaining_items_count property of this ContainerImageCollection.
        :type remaining_items_count: int

        """
        self.swagger_types = {
            "token": "str",
            "access_token": "str",
            "expires_in": "int",
            "scope": "str",
        }

        self.attribute_map = {
            "token": "token",
            "access_token": "access_token",
            "expires_in": "expires_in",
            "scope": "scope",
        }

        self._token = None
        self._access_token = None
        self._expires_in = None
        self._scope = None

    @property
    def token(self):
        return self._token

    @token.setter
    def token(self, token):
        self._token = token

    @property
    def access_token(self):
        return self._access_token

    @access_token.setter
    def access_token(self, access_token: str):
        self._access_token = access_token

    @property
    def scope(self):
        return self._scope

    @scope.setter
    def scope(self, scope: str):
        self._scope = scope

    @property
    def expires_in(self):
        return self._expires_in

    @expires_in.setter
    def expires_in(self, expires_in: int):
        self._expires_in = expires_in

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
