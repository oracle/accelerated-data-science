# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class ExpectationDetails(object):
    """
    Set of expectations for validation of features
    """

    #: A constant which can be used with the expectation_type property of a ExpectationDetails.
    #: This constant has a value of "STRICT"
    EXPECTATION_TYPE_STRICT = "STRICT"

    #: A constant which can be used with the expectation_type property of a ExpectationDetails.
    #: This constant has a value of "LENIENT"
    EXPECTATION_TYPE_LENIENT = "LENIENT"

    #: A constant which can be used with the expectation_type property of a ExpectationDetails.
    #: This constant has a value of "NO_EXPECTATION"
    EXPECTATION_TYPE_NO_EXPECTATION = "NO_EXPECTATION"

    #: A constant which can be used with the validation_engine_type property of a ExpectationDetails.
    #: This constant has a value of "GREAT_EXPECTATIONS"
    VALIDATION_ENGINE_TYPE_GREAT_EXPECTATIONS = "GREAT_EXPECTATIONS"

    def __init__(self, **kwargs):
        """
        Initializes a new ExpectationDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param name:
            The value to assign to the name property of this ExpectationDetails.
        :type name: str

        :param description:
            The value to assign to the description property of this ExpectationDetails.
        :type description: str

        :param expectation_type:
            The value to assign to the expectation_type property of this ExpectationDetails.
            Allowed values for this property are: "STRICT", "LENIENT", "NO_EXPECTATION", 'UNKNOWN_ENUM_VALUE'.
            Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.
        :type expectation_type: str

        :param validation_engine_type:
            The value to assign to the validation_engine_type property of this ExpectationDetails.
            Allowed values for this property are: "GREAT_EXPECTATIONS", 'UNKNOWN_ENUM_VALUE'.
            Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.
        :type validation_engine_type: str

        :param create_rule_details:
            The value to assign to the create_rule_details property of this ExpectationDetails.
        :type create_rule_details: list[oci.feature_store.models.CreateRuleDetail]

        """
        self.swagger_types = {
            'name': 'str',
            'description': 'str',
            'expectation_type': 'str',
            'validation_engine_type': 'str',
            'create_rule_details': 'list[CreateRuleDetail]'
        }

        self.attribute_map = {
            'name': 'name',
            'description': 'description',
            'expectation_type': 'expectationType',
            'validation_engine_type': 'validationEngineType',
            'create_rule_details': 'createRuleDetails'
        }

        self._name = None
        self._description = None
        self._expectation_type = None
        self._validation_engine_type = None
        self._create_rule_details = None

    @property
    def name(self):
        """
        **[Required]** Gets the name of this ExpectationDetails.
        Name of expectation


        :return: The name of this ExpectationDetails.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Sets the name of this ExpectationDetails.
        Name of expectation


        :param name: The name of this ExpectationDetails.
        :type: str
        """
        self._name = name

    @property
    def description(self):
        """
        **[Required]** Gets the description of this ExpectationDetails.
        A short description of the expectation


        :return: The description of this ExpectationDetails.
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """
        Sets the description of this ExpectationDetails.
        A short description of the expectation


        :param description: The description of this ExpectationDetails.
        :type: str
        """
        self._description = description

    @property
    def expectation_type(self):
        """
        **[Required]** Gets the expectation_type of this ExpectationDetails.
        Type of expectation

        Allowed values for this property are: "STRICT", "LENIENT", "NO_EXPECTATION", 'UNKNOWN_ENUM_VALUE'.
        Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.


        :return: The expectation_type of this ExpectationDetails.
        :rtype: str
        """
        return self._expectation_type

    @expectation_type.setter
    def expectation_type(self, expectation_type):
        """
        Sets the expectation_type of this ExpectationDetails.
        Type of expectation


        :param expectation_type: The expectation_type of this ExpectationDetails.
        :type: str
        """
        allowed_values = ["STRICT", "LENIENT", "NO_EXPECTATION"]
        if not value_allowed_none_or_none_sentinel(expectation_type, allowed_values):
            expectation_type = 'UNKNOWN_ENUM_VALUE'
        self._expectation_type = expectation_type

    @property
    def validation_engine_type(self):
        """
        Gets the validation_engine_type of this ExpectationDetails.
        Type of validation engine

        Allowed values for this property are: "GREAT_EXPECTATIONS", 'UNKNOWN_ENUM_VALUE'.
        Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.


        :return: The validation_engine_type of this ExpectationDetails.
        :rtype: str
        """
        return self._validation_engine_type

    @validation_engine_type.setter
    def validation_engine_type(self, validation_engine_type):
        """
        Sets the validation_engine_type of this ExpectationDetails.
        Type of validation engine


        :param validation_engine_type: The validation_engine_type of this ExpectationDetails.
        :type: str
        """
        allowed_values = ["GREAT_EXPECTATIONS"]
        if not value_allowed_none_or_none_sentinel(validation_engine_type, allowed_values):
            validation_engine_type = 'UNKNOWN_ENUM_VALUE'
        self._validation_engine_type = validation_engine_type

    @property
    def create_rule_details(self):
        """
        **[Required]** Gets the create_rule_details of this ExpectationDetails.
        feature rules


        :return: The create_rule_details of this ExpectationDetails.
        :rtype: list[oci.feature_store.models.CreateRuleDetail]
        """
        return self._create_rule_details

    @create_rule_details.setter
    def create_rule_details(self, create_rule_details):
        """
        Sets the create_rule_details of this ExpectationDetails.
        feature rules


        :param create_rule_details: The create_rule_details of this ExpectationDetails.
        :type: list[oci.feature_store.models.CreateRuleDetail]
        """
        self._create_rule_details = create_rule_details

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
