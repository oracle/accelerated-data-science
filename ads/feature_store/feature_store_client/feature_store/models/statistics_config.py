# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class StatisticsConfig(object):
    """
    enable/disable feature statistic computation and specify the columns for which the stats to be computed
    """

    def __init__(self, **kwargs):
        """
        Initializes a new StatisticsConfig object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param is_enabled:
            The value to assign to the is_enabled property of this StatisticsConfig.
        :type is_enabled: bool

        :param columns:
            The value to assign to the columns property of this StatisticsConfig.
        :type columns: list[str]

        """
        self.swagger_types = {
            'is_enabled': 'bool',
            'columns': 'list[str]'
        }

        self.attribute_map = {
            'is_enabled': 'isEnabled',
            'columns': 'columns'
        }

        self._is_enabled = None
        self._columns = None

    @property
    def is_enabled(self):
        """
        **[Required]** Gets the is_enabled of this StatisticsConfig.
        enable/disable feature statistic computation


        :return: The is_enabled of this StatisticsConfig.
        :rtype: bool
        """
        return self._is_enabled

    @is_enabled.setter
    def is_enabled(self, is_enabled):
        """
        Sets the is_enabled of this StatisticsConfig.
        enable/disable feature statistic computation


        :param is_enabled: The is_enabled of this StatisticsConfig.
        :type: bool
        """
        self._is_enabled = is_enabled

    @property
    def columns(self):
        """
        Gets the columns of this StatisticsConfig.
        column names for which the statistics to be calculated


        :return: The columns of this StatisticsConfig.
        :rtype: list[str]
        """
        return self._columns

    @columns.setter
    def columns(self, columns):
        """
        Sets the columns of this StatisticsConfig.
        column names for which the statistics to be calculated


        :param columns: The columns of this StatisticsConfig.
        :type: list[str]
        """
        self._columns = columns

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
