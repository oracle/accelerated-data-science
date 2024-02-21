# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class Lineage(object):
    """
    Job of feature store construct
    """

    #: A constant which can be used with the lineage_type property of a Lineage.
    #: This constant has a value of "FEATURE_GROUP"
    LINEAGE_TYPE_FEATURE_GROUP = "FEATURE_GROUP"

    #: A constant which can be used with the lineage_type property of a Lineage.
    #: This constant has a value of "DATASET"
    LINEAGE_TYPE_DATASET = "DATASET"

    def __init__(self, **kwargs):
        """
        Initializes a new Lineage object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param lineage:
            The value to assign to the lineage property of this Lineage.
        :type lineage: oci.feature_store.models.LineageSummaryCollection

        :param lineage_type:
            The value to assign to the lineage_type property of this Lineage.
            Allowed values for this property are: "FEATURE_GROUP", "DATASET", 'UNKNOWN_ENUM_VALUE'.
            Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.
        :type lineage_type: str

        """
        self.swagger_types = {
            'lineage': 'LineageSummaryCollection',
            'lineage_type': 'str'
        }

        self.attribute_map = {
            'lineage': 'lineage',
            'lineage_type': 'lineageType'
        }

        self._lineage = None
        self._lineage_type = None

    @property
    def lineage(self):
        """
        **[Required]** Gets the lineage of this Lineage.

        :return: The lineage of this Lineage.
        :rtype: oci.feature_store.models.LineageSummaryCollection
        """
        return self._lineage

    @lineage.setter
    def lineage(self, lineage):
        """
        Sets the lineage of this Lineage.

        :param lineage: The lineage of this Lineage.
        :type: oci.feature_store.models.LineageSummaryCollection
        """
        self._lineage = lineage

    @property
    def lineage_type(self):
        """
        Gets the lineage_type of this Lineage.
        Type of lineage construct for which the lineage is required

        Allowed values for this property are: "FEATURE_GROUP", "DATASET", 'UNKNOWN_ENUM_VALUE'.
        Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.


        :return: The lineage_type of this Lineage.
        :rtype: str
        """
        return self._lineage_type

    @lineage_type.setter
    def lineage_type(self, lineage_type):
        """
        Sets the lineage_type of this Lineage.
        Type of lineage construct for which the lineage is required


        :param lineage_type: The lineage_type of this Lineage.
        :type: str
        """
        allowed_values = ["FEATURE_GROUP", "DATASET"]
        if not value_allowed_none_or_none_sentinel(lineage_type, allowed_values):
            lineage_type = 'UNKNOWN_ENUM_VALUE'
        self._lineage_type = lineage_type

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
