# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class FeatureOptionDetails(object):
    """
    feature options to be used for read / write access
    """

    def __init__(self, **kwargs):
        """
        Initializes a new FeatureOptionDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param feature_option_write_config_details:
            The value to assign to the feature_option_write_config_details property of this FeatureOptionDetails.
        :type feature_option_write_config_details: oci.feature_store.models.FeatureOptionWriteConfigDetails

        :param feature_option_read_config_details:
            The value to assign to the feature_option_read_config_details property of this FeatureOptionDetails.
        :type feature_option_read_config_details: oci.feature_store.models.FeatureOptionReadConfigDetails

        """
        self.swagger_types = {
            'feature_option_write_config_details': 'FeatureOptionWriteConfigDetails',
            'feature_option_read_config_details': 'FeatureOptionReadConfigDetails'
        }

        self.attribute_map = {
            'feature_option_write_config_details': 'featureOptionWriteConfigDetails',
            'feature_option_read_config_details': 'featureOptionReadConfigDetails'
        }

        self._feature_option_write_config_details = None
        self._feature_option_read_config_details = None

    @property
    def feature_option_write_config_details(self):
        """
        Gets the feature_option_write_config_details of this FeatureOptionDetails.

        :return: The feature_option_write_config_details of this FeatureOptionDetails.
        :rtype: oci.feature_store.models.FeatureOptionWriteConfigDetails
        """
        return self._feature_option_write_config_details

    @feature_option_write_config_details.setter
    def feature_option_write_config_details(self, feature_option_write_config_details):
        """
        Sets the feature_option_write_config_details of this FeatureOptionDetails.

        :param feature_option_write_config_details: The feature_option_write_config_details of this FeatureOptionDetails.
        :type: oci.feature_store.models.FeatureOptionWriteConfigDetails
        """
        self._feature_option_write_config_details = feature_option_write_config_details

    @property
    def feature_option_read_config_details(self):
        """
        Gets the feature_option_read_config_details of this FeatureOptionDetails.

        :return: The feature_option_read_config_details of this FeatureOptionDetails.
        :rtype: oci.feature_store.models.FeatureOptionReadConfigDetails
        """
        return self._feature_option_read_config_details

    @feature_option_read_config_details.setter
    def feature_option_read_config_details(self, feature_option_read_config_details):
        """
        Sets the feature_option_read_config_details of this FeatureOptionDetails.

        :param feature_option_read_config_details: The feature_option_read_config_details of this FeatureOptionDetails.
        :type: oci.feature_store.models.FeatureOptionReadConfigDetails
        """
        self._feature_option_read_config_details = feature_option_read_config_details

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
