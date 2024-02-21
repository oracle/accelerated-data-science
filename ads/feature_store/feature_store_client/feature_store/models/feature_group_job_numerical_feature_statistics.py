# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.

from .feature_group_job_statistics_item import FeatureGroupJobStatisticsItem
from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class FeatureGroupJobNumericalFeatureStatistics(FeatureGroupJobStatisticsItem):
    """
    Feature statistics for numerical type feature
    """

    def __init__(self, **kwargs):
        """
        Initializes a new FeatureGroupJobNumericalFeatureStatistics object with values from keyword arguments. The default value of the :py:attr:`~oci.feature_store.models.FeatureGroupJobNumericalFeatureStatistics.feature_type` attribute
        of this class is ``NUMERICAL`` and it should not be changed.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param feature:
            The value to assign to the feature property of this FeatureGroupJobNumericalFeatureStatistics.
        :type feature: str

        :param feature_type:
            The value to assign to the feature_type property of this FeatureGroupJobNumericalFeatureStatistics.
            Allowed values for this property are: "NUMERICAL", "CATEGORICAL"
        :type feature_type: str

        :param min:
            The value to assign to the min property of this FeatureGroupJobNumericalFeatureStatistics.
        :type min: float

        :param max:
            The value to assign to the max property of this FeatureGroupJobNumericalFeatureStatistics.
        :type max: float

        :param median:
            The value to assign to the median property of this FeatureGroupJobNumericalFeatureStatistics.
        :type median: float

        :param mean:
            The value to assign to the mean property of this FeatureGroupJobNumericalFeatureStatistics.
        :type mean: float

        """
        self.swagger_types = {
            'feature': 'str',
            'feature_type': 'str',
            'min': 'float',
            'max': 'float',
            'median': 'float',
            'mean': 'float'
        }

        self.attribute_map = {
            'feature': 'feature',
            'feature_type': 'featureType',
            'min': 'min',
            'max': 'max',
            'median': 'median',
            'mean': 'mean'
        }

        self._feature = None
        self._feature_type = None
        self._min = None
        self._max = None
        self._median = None
        self._mean = None
        self._feature_type = 'NUMERICAL'

    @property
    def min(self):
        """
        Gets the min of this FeatureGroupJobNumericalFeatureStatistics.
        The minimum value of the feature


        :return: The min of this FeatureGroupJobNumericalFeatureStatistics.
        :rtype: float
        """
        return self._min

    @min.setter
    def min(self, min):
        """
        Sets the min of this FeatureGroupJobNumericalFeatureStatistics.
        The minimum value of the feature


        :param min: The min of this FeatureGroupJobNumericalFeatureStatistics.
        :type: float
        """
        self._min = min

    @property
    def max(self):
        """
        Gets the max of this FeatureGroupJobNumericalFeatureStatistics.
        The maximum value of the feature


        :return: The max of this FeatureGroupJobNumericalFeatureStatistics.
        :rtype: float
        """
        return self._max

    @max.setter
    def max(self, max):
        """
        Sets the max of this FeatureGroupJobNumericalFeatureStatistics.
        The maximum value of the feature


        :param max: The max of this FeatureGroupJobNumericalFeatureStatistics.
        :type: float
        """
        self._max = max

    @property
    def median(self):
        """
        Gets the median of this FeatureGroupJobNumericalFeatureStatistics.
        The median value of the feature


        :return: The median of this FeatureGroupJobNumericalFeatureStatistics.
        :rtype: float
        """
        return self._median

    @median.setter
    def median(self, median):
        """
        Sets the median of this FeatureGroupJobNumericalFeatureStatistics.
        The median value of the feature


        :param median: The median of this FeatureGroupJobNumericalFeatureStatistics.
        :type: float
        """
        self._median = median

    @property
    def mean(self):
        """
        Gets the mean of this FeatureGroupJobNumericalFeatureStatistics.
        The mean value of the feature


        :return: The mean of this FeatureGroupJobNumericalFeatureStatistics.
        :rtype: float
        """
        return self._mean

    @mean.setter
    def mean(self, mean):
        """
        Sets the mean of this FeatureGroupJobNumericalFeatureStatistics.
        The mean value of the feature


        :param mean: The mean of this FeatureGroupJobNumericalFeatureStatistics.
        :type: float
        """
        self._mean = mean

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
