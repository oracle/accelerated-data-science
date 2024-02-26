# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class PatchFeatureGroupDetails(object):
    """
    Patch details for updating feature group
    """

    def __init__(self, **kwargs):
        """
        Initializes a new PatchFeatureGroupDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param expectation_details:
            The value to assign to the expectation_details property of this PatchFeatureGroupDetails.
        :type expectation_details: oci.feature_store.models.ExpectationDetails

        :param input_feature_details:
            The value to assign to the input_feature_details property of this PatchFeatureGroupDetails.
        :type input_feature_details: list[oci.feature_store.models.RawFeatureDetail]

        """
        self.swagger_types = {
            'expectation_details': 'ExpectationDetails',
            'input_feature_details': 'list[RawFeatureDetail]'
        }

        self.attribute_map = {
            'expectation_details': 'expectationDetails',
            'input_feature_details': 'inputFeatureDetails'
        }

        self._expectation_details = None
        self._input_feature_details = None

    @property
    def expectation_details(self):
        """
        Gets the expectation_details of this PatchFeatureGroupDetails.

        :return: The expectation_details of this PatchFeatureGroupDetails.
        :rtype: oci.feature_store.models.ExpectationDetails
        """
        return self._expectation_details

    @expectation_details.setter
    def expectation_details(self, expectation_details):
        """
        Sets the expectation_details of this PatchFeatureGroupDetails.

        :param expectation_details: The expectation_details of this PatchFeatureGroupDetails.
        :type: oci.feature_store.models.ExpectationDetails
        """
        self._expectation_details = expectation_details

    @property
    def input_feature_details(self):
        """
        Gets the input_feature_details of this PatchFeatureGroupDetails.
        input feature group schema details


        :return: The input_feature_details of this PatchFeatureGroupDetails.
        :rtype: list[oci.feature_store.models.RawFeatureDetail]
        """
        return self._input_feature_details

    @input_feature_details.setter
    def input_feature_details(self, input_feature_details):
        """
        Sets the input_feature_details of this PatchFeatureGroupDetails.
        input feature group schema details


        :param input_feature_details: The input_feature_details of this PatchFeatureGroupDetails.
        :type: list[oci.feature_store.models.RawFeatureDetail]
        """
        self._input_feature_details = input_feature_details

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
