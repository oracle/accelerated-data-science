# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class DatasetJobStatisticsItem(object):
    """
    Statistics details item of the Job
    """

    #: A constant which can be used with the feature_type property of a DatasetJobStatisticsItem.
    #: This constant has a value of "NUMERICAL"
    FEATURE_TYPE_NUMERICAL = "NUMERICAL"

    #: A constant which can be used with the feature_type property of a DatasetJobStatisticsItem.
    #: This constant has a value of "CATEGORICAL"
    FEATURE_TYPE_CATEGORICAL = "CATEGORICAL"

    def __init__(self, **kwargs):
        """
        Initializes a new DatasetJobStatisticsItem object with values from keyword arguments. This class has the following subclasses and if you are using this class as input
        to a service operations then you should favor using a subclass over the base class:

        * :class:`~oci.feature_store.models.DatasetJobCategoricalFeatureStatistics`
        * :class:`~oci.feature_store.models.DatasetJobNumericalFeatureStatistics`

        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param feature:
            The value to assign to the feature property of this DatasetJobStatisticsItem.
        :type feature: str

        :param feature_type:
            The value to assign to the feature_type property of this DatasetJobStatisticsItem.
            Allowed values for this property are: "NUMERICAL", "CATEGORICAL", 'UNKNOWN_ENUM_VALUE'.
            Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.
        :type feature_type: str

        """
        self.swagger_types = {
            'feature': 'str',
            'feature_type': 'str'
        }

        self.attribute_map = {
            'feature': 'feature',
            'feature_type': 'featureType'
        }

        self._feature = None
        self._feature_type = None

    @staticmethod
    def get_subtype(object_dictionary):
        """
        Given the hash representation of a subtype of this class,
        use the info in the hash to return the class of the subtype.
        """
        type = object_dictionary['featureType']

        if type == 'CATEGORICAL':
            return 'DatasetJobCategoricalFeatureStatistics'

        if type == 'NUMERICAL':
            return 'DatasetJobNumericalFeatureStatistics'
        else:
            return 'DatasetJobStatisticsItem'

    @property
    def feature(self):
        """
        Gets the feature of this DatasetJobStatisticsItem.
        The feature for which statistics is calculated


        :return: The feature of this DatasetJobStatisticsItem.
        :rtype: str
        """
        return self._feature

    @feature.setter
    def feature(self, feature):
        """
        Sets the feature of this DatasetJobStatisticsItem.
        The feature for which statistics is calculated


        :param feature: The feature of this DatasetJobStatisticsItem.
        :type: str
        """
        self._feature = feature

    @property
    def feature_type(self):
        """
        Gets the feature_type of this DatasetJobStatisticsItem.
        The feature  type for which statistics is calculated. Either numerical or categorical

        Allowed values for this property are: "NUMERICAL", "CATEGORICAL", 'UNKNOWN_ENUM_VALUE'.
        Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.


        :return: The feature_type of this DatasetJobStatisticsItem.
        :rtype: str
        """
        return self._feature_type

    @feature_type.setter
    def feature_type(self, feature_type):
        """
        Sets the feature_type of this DatasetJobStatisticsItem.
        The feature  type for which statistics is calculated. Either numerical or categorical


        :param feature_type: The feature_type of this DatasetJobStatisticsItem.
        :type: str
        """
        allowed_values = ["NUMERICAL", "CATEGORICAL"]
        if not value_allowed_none_or_none_sentinel(feature_type, allowed_values):
            feature_type = 'UNKNOWN_ENUM_VALUE'
        self._feature_type = feature_type

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
