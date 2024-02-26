# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.

from .feature_group_job_statistics_item import FeatureGroupJobStatisticsItem
from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class FeatureGroupJobCategoricalFeatureStatistics(FeatureGroupJobStatisticsItem):
    """
    Feature statistics for categorical type feature
    """

    def __init__(self, **kwargs):
        """
        Initializes a new FeatureGroupJobCategoricalFeatureStatistics object with values from keyword arguments. The default value of the :py:attr:`~oci.feature_store.models.FeatureGroupJobCategoricalFeatureStatistics.feature_type` attribute
        of this class is ``CATEGORICAL`` and it should not be changed.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param feature:
            The value to assign to the feature property of this FeatureGroupJobCategoricalFeatureStatistics.
        :type feature: str

        :param feature_type:
            The value to assign to the feature_type property of this FeatureGroupJobCategoricalFeatureStatistics.
            Allowed values for this property are: "NUMERICAL", "CATEGORICAL"
        :type feature_type: str

        :param distinct_count:
            The value to assign to the distinct_count property of this FeatureGroupJobCategoricalFeatureStatistics.
        :type distinct_count: int

        """
        self.swagger_types = {
            'feature': 'str',
            'feature_type': 'str',
            'distinct_count': 'int'
        }

        self.attribute_map = {
            'feature': 'feature',
            'feature_type': 'featureType',
            'distinct_count': 'distinctCount'
        }

        self._feature = None
        self._feature_type = None
        self._distinct_count = None
        self._feature_type = 'CATEGORICAL'

    @property
    def distinct_count(self):
        """
        Gets the distinct_count of this FeatureGroupJobCategoricalFeatureStatistics.
        Total distinct count of the feature


        :return: The distinct_count of this FeatureGroupJobCategoricalFeatureStatistics.
        :rtype: int
        """
        return self._distinct_count

    @distinct_count.setter
    def distinct_count(self, distinct_count):
        """
        Sets the distinct_count of this FeatureGroupJobCategoricalFeatureStatistics.
        Total distinct count of the feature


        :param distinct_count: The distinct_count of this FeatureGroupJobCategoricalFeatureStatistics.
        :type: int
        """
        self._distinct_count = distinct_count

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
