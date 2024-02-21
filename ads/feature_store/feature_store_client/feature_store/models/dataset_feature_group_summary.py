# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class DatasetFeatureGroupSummary(object):
    """
    feature group details for dataset mapping.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new DatasetFeatureGroupSummary object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param feature_group_id:
            The value to assign to the feature_group_id property of this DatasetFeatureGroupSummary.
        :type feature_group_id: str

        """
        self.swagger_types = {
            'feature_group_id': 'str'
        }

        self.attribute_map = {
            'feature_group_id': 'featureGroupId'
        }

        self._feature_group_id = None

    @property
    def feature_group_id(self):
        """
        **[Required]** Gets the feature_group_id of this DatasetFeatureGroupSummary.
        ID of the feature group


        :return: The feature_group_id of this DatasetFeatureGroupSummary.
        :rtype: str
        """
        return self._feature_group_id

    @feature_group_id.setter
    def feature_group_id(self, feature_group_id):
        """
        Sets the feature_group_id of this DatasetFeatureGroupSummary.
        ID of the feature group


        :param feature_group_id: The feature_group_id of this DatasetFeatureGroupSummary.
        :type: str
        """
        self._feature_group_id = feature_group_id

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
