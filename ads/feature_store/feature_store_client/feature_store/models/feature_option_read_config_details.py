# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class FeatureOptionReadConfigDetails(object):
    """
    feature options to be used for read / write access
    """

    def __init__(self, **kwargs):
        """
        Initializes a new FeatureOptionReadConfigDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param version_as_of:
            The value to assign to the version_as_of property of this FeatureOptionReadConfigDetails.
        :type version_as_of: int

        :param timestamp_as_of:
            The value to assign to the timestamp_as_of property of this FeatureOptionReadConfigDetails.
        :type timestamp_as_of: datetime

        """
        self.swagger_types = {
            'version_as_of': 'int',
            'timestamp_as_of': 'datetime'
        }

        self.attribute_map = {
            'version_as_of': 'versionAsOf',
            'timestamp_as_of': 'timestampAsOf'
        }

        self._version_as_of = None
        self._timestamp_as_of = None

    @property
    def version_as_of(self):
        """
        Gets the version_as_of of this FeatureOptionReadConfigDetails.
        specify the version number required for feature materialization job access


        :return: The version_as_of of this FeatureOptionReadConfigDetails.
        :rtype: int
        """
        return self._version_as_of

    @version_as_of.setter
    def version_as_of(self, version_as_of):
        """
        Sets the version_as_of of this FeatureOptionReadConfigDetails.
        specify the version number required for feature materialization job access


        :param version_as_of: The version_as_of of this FeatureOptionReadConfigDetails.
        :type: int
        """
        self._version_as_of = version_as_of

    @property
    def timestamp_as_of(self):
        """
        Gets the timestamp_as_of of this FeatureOptionReadConfigDetails.
        The date and time ater which feature data is required in the timestamp format defined by `RFC3339`__.

        __ https://tools.ietf.org/html/rfc3339


        :return: The timestamp_as_of of this FeatureOptionReadConfigDetails.
        :rtype: datetime
        """
        return self._timestamp_as_of

    @timestamp_as_of.setter
    def timestamp_as_of(self, timestamp_as_of):
        """
        Sets the timestamp_as_of of this FeatureOptionReadConfigDetails.
        The date and time ater which feature data is required in the timestamp format defined by `RFC3339`__.

        __ https://tools.ietf.org/html/rfc3339


        :param timestamp_as_of: The timestamp_as_of of this FeatureOptionReadConfigDetails.
        :type: datetime
        """
        self._timestamp_as_of = timestamp_as_of

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
