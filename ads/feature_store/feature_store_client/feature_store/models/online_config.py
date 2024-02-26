# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class OnlineConfig(object):
    """
    Online configuration related information of FeatureStore.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new OnlineConfig object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param redis_id:
            The value to assign to the redis_id property of this OnlineConfig.
        :type redis_id: str

        :param open_search_id:
            The value to assign to the open_search_id property of this OnlineConfig.
        :type open_search_id: str

        """
        self.swagger_types = {
            'redis_id': 'str',
            'open_search_id': 'str'
        }

        self.attribute_map = {
            'redis_id': 'redisId',
            'open_search_id': 'openSearchId'
        }

        self._redis_id = None
        self._open_search_id = None

    @property
    def redis_id(self):
        """
        **[Required]** Gets the redis_id of this OnlineConfig.
        Redis Cluster identifier.


        :return: The redis_id of this OnlineConfig.
        :rtype: str
        """
        return self._redis_id

    @redis_id.setter
    def redis_id(self, redis_id):
        """
        Sets the redis_id of this OnlineConfig.
        Redis Cluster identifier.


        :param redis_id: The redis_id of this OnlineConfig.
        :type: str
        """
        self._redis_id = redis_id

    @property
    def open_search_id(self):
        """
        Gets the open_search_id of this OnlineConfig.
        elastic Search identifier.


        :return: The open_search_id of this OnlineConfig.
        :rtype: str
        """
        return self._open_search_id

    @open_search_id.setter
    def open_search_id(self, open_search_id):
        """
        Sets the open_search_id of this OnlineConfig.
        elastic Search identifier.


        :param open_search_id: The open_search_id of this OnlineConfig.
        :type: str
        """
        self._open_search_id = open_search_id

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
