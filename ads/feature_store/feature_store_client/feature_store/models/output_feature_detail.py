# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class OutputFeatureDetail(object):
    """
    Feature detail for a particular feature in feature group or dataset
    """

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "STRING"
    FEATURE_TYPE_STRING = "STRING"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "SHORT"
    FEATURE_TYPE_SHORT = "SHORT"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "INTEGER"
    FEATURE_TYPE_INTEGER = "INTEGER"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "LONG"
    FEATURE_TYPE_LONG = "LONG"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "FLOAT"
    FEATURE_TYPE_FLOAT = "FLOAT"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "DOUBLE"
    FEATURE_TYPE_DOUBLE = "DOUBLE"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "BOOLEAN"
    FEATURE_TYPE_BOOLEAN = "BOOLEAN"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "DATE"
    FEATURE_TYPE_DATE = "DATE"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "TIMESTAMP"
    FEATURE_TYPE_TIMESTAMP = "TIMESTAMP"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "DECIMAL"
    FEATURE_TYPE_DECIMAL = "DECIMAL"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "BINARY"
    FEATURE_TYPE_BINARY = "BINARY"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "BYTE"
    FEATURE_TYPE_BYTE = "BYTE"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "STRING_ARRAY"
    FEATURE_TYPE_STRING_ARRAY = "STRING_ARRAY"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "SHORT_ARRAY"
    FEATURE_TYPE_SHORT_ARRAY = "SHORT_ARRAY"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "INTEGER_ARRAY"
    FEATURE_TYPE_INTEGER_ARRAY = "INTEGER_ARRAY"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "LONG_ARRAY"
    FEATURE_TYPE_LONG_ARRAY = "LONG_ARRAY"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "FLOAT_ARRAY"
    FEATURE_TYPE_FLOAT_ARRAY = "FLOAT_ARRAY"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "DOUBLE_ARRAY"
    FEATURE_TYPE_DOUBLE_ARRAY = "DOUBLE_ARRAY"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "BINARY_ARRAY"
    FEATURE_TYPE_BINARY_ARRAY = "BINARY_ARRAY"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "DATE_ARRAY"
    FEATURE_TYPE_DATE_ARRAY = "DATE_ARRAY"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "TIMESTAMP_ARRAY"
    FEATURE_TYPE_TIMESTAMP_ARRAY = "TIMESTAMP_ARRAY"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "BYTE_ARRAY"
    FEATURE_TYPE_BYTE_ARRAY = "BYTE_ARRAY"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "BOOLEAN_ARRAY"
    FEATURE_TYPE_BOOLEAN_ARRAY = "BOOLEAN_ARRAY"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "STRING_STRING_MAP"
    FEATURE_TYPE_STRING_STRING_MAP = "STRING_STRING_MAP"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "STRING_INTEGER_MAP"
    FEATURE_TYPE_STRING_INTEGER_MAP = "STRING_INTEGER_MAP"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "STRING_SHORT_MAP"
    FEATURE_TYPE_STRING_SHORT_MAP = "STRING_SHORT_MAP"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "STRING_LONG_MAP"
    FEATURE_TYPE_STRING_LONG_MAP = "STRING_LONG_MAP"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "STRING_FLOAT_MAP"
    FEATURE_TYPE_STRING_FLOAT_MAP = "STRING_FLOAT_MAP"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "STRING_DOUBLE_MAP"
    FEATURE_TYPE_STRING_DOUBLE_MAP = "STRING_DOUBLE_MAP"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "STRING_TIMESTAMP_MAP"
    FEATURE_TYPE_STRING_TIMESTAMP_MAP = "STRING_TIMESTAMP_MAP"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "STRING_DATE_MAP"
    FEATURE_TYPE_STRING_DATE_MAP = "STRING_DATE_MAP"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "STRING_BINARY_MAP"
    FEATURE_TYPE_STRING_BINARY_MAP = "STRING_BINARY_MAP"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "STRING_BYTE_MAP"
    FEATURE_TYPE_STRING_BYTE_MAP = "STRING_BYTE_MAP"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "STRING_BOOLEAN_MAP"
    FEATURE_TYPE_STRING_BOOLEAN_MAP = "STRING_BOOLEAN_MAP"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "STRUCT"
    FEATURE_TYPE_STRUCT = "STRUCT"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "UNKNOWN"
    FEATURE_TYPE_UNKNOWN = "UNKNOWN"

    #: A constant which can be used with the feature_type property of a OutputFeatureDetail.
    #: This constant has a value of "COMPLEX"
    FEATURE_TYPE_COMPLEX = "COMPLEX"

    def __init__(self, **kwargs):
        """
        Initializes a new OutputFeatureDetail object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param name:
            The value to assign to the name property of this OutputFeatureDetail.
        :type name: str

        :param feature_type:
            The value to assign to the feature_type property of this OutputFeatureDetail.
            Allowed values for this property are: "STRING", "SHORT", "INTEGER", "LONG", "FLOAT", "DOUBLE", "BOOLEAN", "DATE", "TIMESTAMP", "DECIMAL", "BINARY", "BYTE", "STRING_ARRAY", "SHORT_ARRAY", "INTEGER_ARRAY", "LONG_ARRAY", "FLOAT_ARRAY", "DOUBLE_ARRAY", "BINARY_ARRAY", "DATE_ARRAY", "TIMESTAMP_ARRAY", "BYTE_ARRAY", "BOOLEAN_ARRAY", "STRING_STRING_MAP", "STRING_INTEGER_MAP", "STRING_SHORT_MAP", "STRING_LONG_MAP", "STRING_FLOAT_MAP", "STRING_DOUBLE_MAP", "STRING_TIMESTAMP_MAP", "STRING_DATE_MAP", "STRING_BINARY_MAP", "STRING_BYTE_MAP", "STRING_BOOLEAN_MAP", "STRUCT", "UNKNOWN", "COMPLEX", 'UNKNOWN_ENUM_VALUE'.
            Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.
        :type feature_type: str

        """
        self.swagger_types = {
            'name': 'str',
            'feature_type': 'str'
        }

        self.attribute_map = {
            'name': 'name',
            'feature_type': 'featureType'
        }

        self._name = None
        self._feature_type = None

    @property
    def name(self):
        """
        **[Required]** Gets the name of this OutputFeatureDetail.
        feature group name.


        :return: The name of this OutputFeatureDetail.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Sets the name of this OutputFeatureDetail.
        feature group name.


        :param name: The name of this OutputFeatureDetail.
        :type: str
        """
        self._name = name

    @property
    def feature_type(self):
        """
        **[Required]** Gets the feature_type of this OutputFeatureDetail.
        The data-type of the feature

        Allowed values for this property are: "STRING", "SHORT", "INTEGER", "LONG", "FLOAT", "DOUBLE", "BOOLEAN", "DATE", "TIMESTAMP", "DECIMAL", "BINARY", "BYTE", "STRING_ARRAY", "SHORT_ARRAY", "INTEGER_ARRAY", "LONG_ARRAY", "FLOAT_ARRAY", "DOUBLE_ARRAY", "BINARY_ARRAY", "DATE_ARRAY", "TIMESTAMP_ARRAY", "BYTE_ARRAY", "BOOLEAN_ARRAY", "STRING_STRING_MAP", "STRING_INTEGER_MAP", "STRING_SHORT_MAP", "STRING_LONG_MAP", "STRING_FLOAT_MAP", "STRING_DOUBLE_MAP", "STRING_TIMESTAMP_MAP", "STRING_DATE_MAP", "STRING_BINARY_MAP", "STRING_BYTE_MAP", "STRING_BOOLEAN_MAP", "STRUCT", "UNKNOWN", "COMPLEX", 'UNKNOWN_ENUM_VALUE'.
        Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.


        :return: The feature_type of this OutputFeatureDetail.
        :rtype: str
        """
        return self._feature_type

    @feature_type.setter
    def feature_type(self, feature_type):
        """
        Sets the feature_type of this OutputFeatureDetail.
        The data-type of the feature


        :param feature_type: The feature_type of this OutputFeatureDetail.
        :type: str
        """
        allowed_values = ["STRING", "SHORT", "INTEGER", "LONG", "FLOAT", "DOUBLE", "BOOLEAN", "DATE", "TIMESTAMP", "DECIMAL", "BINARY", "BYTE", "STRING_ARRAY", "SHORT_ARRAY", "INTEGER_ARRAY", "LONG_ARRAY", "FLOAT_ARRAY", "DOUBLE_ARRAY", "BINARY_ARRAY", "DATE_ARRAY", "TIMESTAMP_ARRAY", "BYTE_ARRAY", "BOOLEAN_ARRAY", "STRING_STRING_MAP", "STRING_INTEGER_MAP", "STRING_SHORT_MAP", "STRING_LONG_MAP", "STRING_FLOAT_MAP", "STRING_DOUBLE_MAP", "STRING_TIMESTAMP_MAP", "STRING_DATE_MAP", "STRING_BINARY_MAP", "STRING_BYTE_MAP", "STRING_BOOLEAN_MAP", "STRUCT", "UNKNOWN", "COMPLEX"]
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
