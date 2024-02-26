# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class CreateRuleDetail(object):
    """
    Set of rules for validation of features
    """

    #: A constant which can be used with the level_type property of a CreateRuleDetail.
    #: This constant has a value of "ERROR"
    LEVEL_TYPE_ERROR = "ERROR"

    #: A constant which can be used with the level_type property of a CreateRuleDetail.
    #: This constant has a value of "WARNING"
    LEVEL_TYPE_WARNING = "WARNING"

    def __init__(self, **kwargs):
        """
        Initializes a new CreateRuleDetail object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param name:
            The value to assign to the name property of this CreateRuleDetail.
        :type name: str

        :param rule_type:
            The value to assign to the rule_type property of this CreateRuleDetail.
        :type rule_type: str

        :param level_type:
            The value to assign to the level_type property of this CreateRuleDetail.
            Allowed values for this property are: "ERROR", "WARNING", 'UNKNOWN_ENUM_VALUE'.
            Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.
        :type level_type: str

        :param arguments:
            The value to assign to the arguments property of this CreateRuleDetail.
        :type arguments: dict(str, object)

        """
        self.swagger_types = {
            'name': 'str',
            'rule_type': 'str',
            'level_type': 'str',
            'arguments': 'dict(str, object)'
        }

        self.attribute_map = {
            'name': 'name',
            'rule_type': 'ruleType',
            'level_type': 'levelType',
            'arguments': 'arguments'
        }

        self._name = None
        self._rule_type = None
        self._level_type = None
        self._arguments = None

    @property
    def name(self):
        """
        **[Required]** Gets the name of this CreateRuleDetail.
        Name of rule


        :return: The name of this CreateRuleDetail.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Sets the name of this CreateRuleDetail.
        Name of rule


        :param name: The name of this CreateRuleDetail.
        :type: str
        """
        self._name = name

    @property
    def rule_type(self):
        """
        **[Required]** Gets the rule_type of this CreateRuleDetail.
        Type of rule to be applied on a feature


        :return: The rule_type of this CreateRuleDetail.
        :rtype: str
        """
        return self._rule_type

    @rule_type.setter
    def rule_type(self, rule_type):
        """
        Sets the rule_type of this CreateRuleDetail.
        Type of rule to be applied on a feature


        :param rule_type: The rule_type of this CreateRuleDetail.
        :type: str
        """
        self._rule_type = rule_type

    @property
    def level_type(self):
        """
        **[Required]** Gets the level_type of this CreateRuleDetail.
        Severity level of a rule

        Allowed values for this property are: "ERROR", "WARNING", 'UNKNOWN_ENUM_VALUE'.
        Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.


        :return: The level_type of this CreateRuleDetail.
        :rtype: str
        """
        return self._level_type

    @level_type.setter
    def level_type(self, level_type):
        """
        Sets the level_type of this CreateRuleDetail.
        Severity level of a rule


        :param level_type: The level_type of this CreateRuleDetail.
        :type: str
        """
        allowed_values = ["ERROR", "WARNING"]
        if not value_allowed_none_or_none_sentinel(level_type, allowed_values):
            level_type = 'UNKNOWN_ENUM_VALUE'
        self._level_type = level_type

    @property
    def arguments(self):
        """
        **[Required]** Gets the arguments of this CreateRuleDetail.
        kwargs that will be passed to the great expectation.
        Example: `{\"column\": \"column name\"}`


        :return: The arguments of this CreateRuleDetail.
        :rtype: dict(str, object)
        """
        return self._arguments

    @arguments.setter
    def arguments(self, arguments):
        """
        Sets the arguments of this CreateRuleDetail.
        kwargs that will be passed to the great expectation.
        Example: `{\"column\": \"column name\"}`


        :param arguments: The arguments of this CreateRuleDetail.
        :type: dict(str, object)
        """
        self._arguments = arguments

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
