# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class DatasetJobValidationOutputDetails(object):
    """
    Validation Details of the Job
    """

    def __init__(self, **kwargs):
        """
        Initializes a new DatasetJobValidationOutputDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param is_successful:
            The value to assign to the is_successful property of this DatasetJobValidationOutputDetails.
        :type is_successful: bool

        :param evaluated_expectations:
            The value to assign to the evaluated_expectations property of this DatasetJobValidationOutputDetails.
        :type evaluated_expectations: int

        :param successful_expectations:
            The value to assign to the successful_expectations property of this DatasetJobValidationOutputDetails.
        :type successful_expectations: int

        :param unsuccessful_expectations:
            The value to assign to the unsuccessful_expectations property of this DatasetJobValidationOutputDetails.
        :type unsuccessful_expectations: int

        :param success_percent:
            The value to assign to the success_percent property of this DatasetJobValidationOutputDetails.
        :type success_percent: str

        """
        self.swagger_types = {
            'is_successful': 'bool',
            'evaluated_expectations': 'int',
            'successful_expectations': 'int',
            'unsuccessful_expectations': 'int',
            'success_percent': 'str'
        }

        self.attribute_map = {
            'is_successful': 'isSuccessful',
            'evaluated_expectations': 'evaluatedExpectations',
            'successful_expectations': 'successfulExpectations',
            'unsuccessful_expectations': 'unsuccessfulExpectations',
            'success_percent': 'successPercent'
        }

        self._is_successful = None
        self._evaluated_expectations = None
        self._successful_expectations = None
        self._unsuccessful_expectations = None
        self._success_percent = None

    @property
    def is_successful(self):
        """
        Gets the is_successful of this DatasetJobValidationOutputDetails.
        Validation Success Result


        :return: The is_successful of this DatasetJobValidationOutputDetails.
        :rtype: bool
        """
        return self._is_successful

    @is_successful.setter
    def is_successful(self, is_successful):
        """
        Sets the is_successful of this DatasetJobValidationOutputDetails.
        Validation Success Result


        :param is_successful: The is_successful of this DatasetJobValidationOutputDetails.
        :type: bool
        """
        self._is_successful = is_successful

    @property
    def evaluated_expectations(self):
        """
        Gets the evaluated_expectations of this DatasetJobValidationOutputDetails.
        Total number of Evaluated Expectations


        :return: The evaluated_expectations of this DatasetJobValidationOutputDetails.
        :rtype: int
        """
        return self._evaluated_expectations

    @evaluated_expectations.setter
    def evaluated_expectations(self, evaluated_expectations):
        """
        Sets the evaluated_expectations of this DatasetJobValidationOutputDetails.
        Total number of Evaluated Expectations


        :param evaluated_expectations: The evaluated_expectations of this DatasetJobValidationOutputDetails.
        :type: int
        """
        self._evaluated_expectations = evaluated_expectations

    @property
    def successful_expectations(self):
        """
        Gets the successful_expectations of this DatasetJobValidationOutputDetails.
        Total number of Successful Expectations


        :return: The successful_expectations of this DatasetJobValidationOutputDetails.
        :rtype: int
        """
        return self._successful_expectations

    @successful_expectations.setter
    def successful_expectations(self, successful_expectations):
        """
        Sets the successful_expectations of this DatasetJobValidationOutputDetails.
        Total number of Successful Expectations


        :param successful_expectations: The successful_expectations of this DatasetJobValidationOutputDetails.
        :type: int
        """
        self._successful_expectations = successful_expectations

    @property
    def unsuccessful_expectations(self):
        """
        Gets the unsuccessful_expectations of this DatasetJobValidationOutputDetails.
        Total number of Unsuccessful Expectations


        :return: The unsuccessful_expectations of this DatasetJobValidationOutputDetails.
        :rtype: int
        """
        return self._unsuccessful_expectations

    @unsuccessful_expectations.setter
    def unsuccessful_expectations(self, unsuccessful_expectations):
        """
        Sets the unsuccessful_expectations of this DatasetJobValidationOutputDetails.
        Total number of Unsuccessful Expectations


        :param unsuccessful_expectations: The unsuccessful_expectations of this DatasetJobValidationOutputDetails.
        :type: int
        """
        self._unsuccessful_expectations = unsuccessful_expectations

    @property
    def success_percent(self):
        """
        Gets the success_percent of this DatasetJobValidationOutputDetails.
        Success Percentage of the Validation Result


        :return: The success_percent of this DatasetJobValidationOutputDetails.
        :rtype: str
        """
        return self._success_percent

    @success_percent.setter
    def success_percent(self, success_percent):
        """
        Sets the success_percent of this DatasetJobValidationOutputDetails.
        Success Percentage of the Validation Result


        :param success_percent: The success_percent of this DatasetJobValidationOutputDetails.
        :type: str
        """
        self._success_percent = success_percent

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
