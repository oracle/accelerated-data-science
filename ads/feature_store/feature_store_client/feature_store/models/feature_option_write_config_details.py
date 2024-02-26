# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class FeatureOptionWriteConfigDetails(object):
    """
    feature options to be used for read / write access
    """

    def __init__(self, **kwargs):
        """
        Initializes a new FeatureOptionWriteConfigDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param merge_schema:
            The value to assign to the merge_schema property of this FeatureOptionWriteConfigDetails.
        :type merge_schema: bool

        :param overwrite_schema:
            The value to assign to the overwrite_schema property of this FeatureOptionWriteConfigDetails.
        :type overwrite_schema: bool

        """
        self.swagger_types = {
            'merge_schema': 'bool',
            'overwrite_schema': 'bool'
        }

        self.attribute_map = {
            'merge_schema': 'mergeSchema',
            'overwrite_schema': 'overwriteSchema'
        }

        self._merge_schema = None
        self._overwrite_schema = None

    @property
    def merge_schema(self):
        """
        Gets the merge_schema of this FeatureOptionWriteConfigDetails.
        enable this to support schema evolution


        :return: The merge_schema of this FeatureOptionWriteConfigDetails.
        :rtype: bool
        """
        return self._merge_schema

    @merge_schema.setter
    def merge_schema(self, merge_schema):
        """
        Sets the merge_schema of this FeatureOptionWriteConfigDetails.
        enable this to support schema evolution


        :param merge_schema: The merge_schema of this FeatureOptionWriteConfigDetails.
        :type: bool
        """
        self._merge_schema = merge_schema

    @property
    def overwrite_schema(self):
        """
        Gets the overwrite_schema of this FeatureOptionWriteConfigDetails.
        enable this to support schema evolution


        :return: The overwrite_schema of this FeatureOptionWriteConfigDetails.
        :rtype: bool
        """
        return self._overwrite_schema

    @overwrite_schema.setter
    def overwrite_schema(self, overwrite_schema):
        """
        Sets the overwrite_schema of this FeatureOptionWriteConfigDetails.
        enable this to support schema evolution


        :param overwrite_schema: The overwrite_schema of this FeatureOptionWriteConfigDetails.
        :type: bool
        """
        self._overwrite_schema = overwrite_schema

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
