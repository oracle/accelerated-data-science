# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class UpdateDatasetDetails(object):
    """
    Update details for dataset
    """

    def __init__(self, **kwargs):
        """
        Initializes a new UpdateDatasetDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param output_feature_details:
            The value to assign to the output_feature_details property of this UpdateDatasetDetails.
        :type output_feature_details: oci.feature_store.models.OutputFeatureDetailCollection

        :param expectation_details:
            The value to assign to the expectation_details property of this UpdateDatasetDetails.
        :type expectation_details: oci.feature_store.models.ExpectationDetails

        :param statistics_config:
            The value to assign to the statistics_config property of this UpdateDatasetDetails.
        :type statistics_config: oci.feature_store.models.StatisticsConfig

        :param model_details:
            The value to assign to the model_details property of this UpdateDatasetDetails.
        :type model_details: oci.feature_store.models.ModelCollection

        :param partition_keys:
            The value to assign to the partition_keys property of this UpdateDatasetDetails.
        :type partition_keys: oci.feature_store.models.PartitionKeyCollection

        """
        self.swagger_types = {
            'output_feature_details': 'OutputFeatureDetailCollection',
            'expectation_details': 'ExpectationDetails',
            'statistics_config': 'StatisticsConfig',
            'model_details': 'ModelCollection',
            'partition_keys': 'PartitionKeyCollection'
        }

        self.attribute_map = {
            'output_feature_details': 'outputFeatureDetails',
            'expectation_details': 'expectationDetails',
            'statistics_config': 'statisticsConfig',
            'model_details': 'modelDetails',
            'partition_keys': 'partitionKeys'
        }

        self._output_feature_details = None
        self._expectation_details = None
        self._statistics_config = None
        self._model_details = None
        self._partition_keys = None

    @property
    def output_feature_details(self):
        """
        Gets the output_feature_details of this UpdateDatasetDetails.

        :return: The output_feature_details of this UpdateDatasetDetails.
        :rtype: oci.feature_store.models.OutputFeatureDetailCollection
        """
        return self._output_feature_details

    @output_feature_details.setter
    def output_feature_details(self, output_feature_details):
        """
        Sets the output_feature_details of this UpdateDatasetDetails.

        :param output_feature_details: The output_feature_details of this UpdateDatasetDetails.
        :type: oci.feature_store.models.OutputFeatureDetailCollection
        """
        self._output_feature_details = output_feature_details

    @property
    def expectation_details(self):
        """
        Gets the expectation_details of this UpdateDatasetDetails.

        :return: The expectation_details of this UpdateDatasetDetails.
        :rtype: oci.feature_store.models.ExpectationDetails
        """
        return self._expectation_details

    @expectation_details.setter
    def expectation_details(self, expectation_details):
        """
        Sets the expectation_details of this UpdateDatasetDetails.

        :param expectation_details: The expectation_details of this UpdateDatasetDetails.
        :type: oci.feature_store.models.ExpectationDetails
        """
        self._expectation_details = expectation_details

    @property
    def statistics_config(self):
        """
        Gets the statistics_config of this UpdateDatasetDetails.

        :return: The statistics_config of this UpdateDatasetDetails.
        :rtype: oci.feature_store.models.StatisticsConfig
        """
        return self._statistics_config

    @statistics_config.setter
    def statistics_config(self, statistics_config):
        """
        Sets the statistics_config of this UpdateDatasetDetails.

        :param statistics_config: The statistics_config of this UpdateDatasetDetails.
        :type: oci.feature_store.models.StatisticsConfig
        """
        self._statistics_config = statistics_config

    @property
    def model_details(self):
        """
        Gets the model_details of this UpdateDatasetDetails.

        :return: The model_details of this UpdateDatasetDetails.
        :rtype: oci.feature_store.models.ModelCollection
        """
        return self._model_details

    @model_details.setter
    def model_details(self, model_details):
        """
        Sets the model_details of this UpdateDatasetDetails.

        :param model_details: The model_details of this UpdateDatasetDetails.
        :type: oci.feature_store.models.ModelCollection
        """
        self._model_details = model_details

    @property
    def partition_keys(self):
        """
        Gets the partition_keys of this UpdateDatasetDetails.

        :return: The partition_keys of this UpdateDatasetDetails.
        :rtype: oci.feature_store.models.PartitionKeyCollection
        """
        return self._partition_keys

    @partition_keys.setter
    def partition_keys(self, partition_keys):
        """
        Sets the partition_keys of this UpdateDatasetDetails.

        :param partition_keys: The partition_keys of this UpdateDatasetDetails.
        :type: oci.feature_store.models.PartitionKeyCollection
        """
        self._partition_keys = partition_keys

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
