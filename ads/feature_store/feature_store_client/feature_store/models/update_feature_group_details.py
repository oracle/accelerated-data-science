# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class UpdateFeatureGroupDetails(object):
    """
    Parameters needed to update an existing feature group.
    """

    #: A constant which can be used with the lifecycle_state property of a UpdateFeatureGroupDetails.
    #: This constant has a value of "CREATING"
    LIFECYCLE_STATE_CREATING = "CREATING"

    #: A constant which can be used with the lifecycle_state property of a UpdateFeatureGroupDetails.
    #: This constant has a value of "UPDATING"
    LIFECYCLE_STATE_UPDATING = "UPDATING"

    #: A constant which can be used with the lifecycle_state property of a UpdateFeatureGroupDetails.
    #: This constant has a value of "ACTIVE"
    LIFECYCLE_STATE_ACTIVE = "ACTIVE"

    #: A constant which can be used with the lifecycle_state property of a UpdateFeatureGroupDetails.
    #: This constant has a value of "DELETING"
    LIFECYCLE_STATE_DELETING = "DELETING"

    #: A constant which can be used with the lifecycle_state property of a UpdateFeatureGroupDetails.
    #: This constant has a value of "DELETED"
    LIFECYCLE_STATE_DELETED = "DELETED"

    #: A constant which can be used with the lifecycle_state property of a UpdateFeatureGroupDetails.
    #: This constant has a value of "FAILED"
    LIFECYCLE_STATE_FAILED = "FAILED"

    #: A constant which can be used with the lifecycle_state property of a UpdateFeatureGroupDetails.
    #: This constant has a value of "NEEDS_ATTENTION"
    LIFECYCLE_STATE_NEEDS_ATTENTION = "NEEDS_ATTENTION"

    def __init__(self, **kwargs):
        """
        Initializes a new UpdateFeatureGroupDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param transformation_id:
            The value to assign to the transformation_id property of this UpdateFeatureGroupDetails.
        :type transformation_id: str

        :param input_feature_details:
            The value to assign to the input_feature_details property of this UpdateFeatureGroupDetails.
        :type input_feature_details: list[oci.feature_store.models.RawFeatureDetail]

        :param output_feature_details:
            The value to assign to the output_feature_details property of this UpdateFeatureGroupDetails.
        :type output_feature_details: oci.feature_store.models.OutputFeatureDetailCollection

        :param lifecycle_state:
            The value to assign to the lifecycle_state property of this UpdateFeatureGroupDetails.
            Allowed values for this property are: "CREATING", "UPDATING", "ACTIVE", "DELETING", "DELETED", "FAILED", "NEEDS_ATTENTION"
        :type lifecycle_state: str

        :param expectation_details:
            The value to assign to the expectation_details property of this UpdateFeatureGroupDetails.
        :type expectation_details: oci.feature_store.models.ExpectationDetails

        :param primary_keys:
            The value to assign to the primary_keys property of this UpdateFeatureGroupDetails.
        :type primary_keys: oci.feature_store.models.PrimaryKeyCollection

        :param partition_keys:
            The value to assign to the partition_keys property of this UpdateFeatureGroupDetails.
        :type partition_keys: oci.feature_store.models.PartitionKeyCollection

        :param statistics_config:
            The value to assign to the statistics_config property of this UpdateFeatureGroupDetails.
        :type statistics_config: oci.feature_store.models.StatisticsConfig

        :param is_infer_schema:
            The value to assign to the is_infer_schema property of this UpdateFeatureGroupDetails.
        :type is_infer_schema: bool

        """
        self.swagger_types = {
            'transformation_id': 'str',
            'input_feature_details': 'list[RawFeatureDetail]',
            'output_feature_details': 'OutputFeatureDetailCollection',
            'lifecycle_state': 'str',
            'expectation_details': 'ExpectationDetails',
            'primary_keys': 'PrimaryKeyCollection',
            'partition_keys': 'PartitionKeyCollection',
            'statistics_config': 'StatisticsConfig',
            'is_infer_schema': 'bool'
        }

        self.attribute_map = {
            'transformation_id': 'transformationId',
            'input_feature_details': 'inputFeatureDetails',
            'output_feature_details': 'outputFeatureDetails',
            'lifecycle_state': 'lifecycleState',
            'expectation_details': 'expectationDetails',
            'primary_keys': 'primaryKeys',
            'partition_keys': 'partitionKeys',
            'statistics_config': 'statisticsConfig',
            'is_infer_schema': 'isInferSchema'
        }

        self._transformation_id = None
        self._input_feature_details = None
        self._output_feature_details = None
        self._lifecycle_state = None
        self._expectation_details = None
        self._primary_keys = None
        self._partition_keys = None
        self._statistics_config = None
        self._is_infer_schema = None

    @property
    def transformation_id(self):
        """
        Gets the transformation_id of this UpdateFeatureGroupDetails.
        The ID for the transformation.


        :return: The transformation_id of this UpdateFeatureGroupDetails.
        :rtype: str
        """
        return self._transformation_id

    @transformation_id.setter
    def transformation_id(self, transformation_id):
        """
        Sets the transformation_id of this UpdateFeatureGroupDetails.
        The ID for the transformation.


        :param transformation_id: The transformation_id of this UpdateFeatureGroupDetails.
        :type: str
        """
        self._transformation_id = transformation_id

    @property
    def input_feature_details(self):
        """
        Gets the input_feature_details of this UpdateFeatureGroupDetails.
        input feature group schema details


        :return: The input_feature_details of this UpdateFeatureGroupDetails.
        :rtype: list[oci.feature_store.models.RawFeatureDetail]
        """
        return self._input_feature_details

    @input_feature_details.setter
    def input_feature_details(self, input_feature_details):
        """
        Sets the input_feature_details of this UpdateFeatureGroupDetails.
        input feature group schema details


        :param input_feature_details: The input_feature_details of this UpdateFeatureGroupDetails.
        :type: list[oci.feature_store.models.RawFeatureDetail]
        """
        self._input_feature_details = input_feature_details

    @property
    def output_feature_details(self):
        """
        Gets the output_feature_details of this UpdateFeatureGroupDetails.

        :return: The output_feature_details of this UpdateFeatureGroupDetails.
        :rtype: oci.feature_store.models.OutputFeatureDetailCollection
        """
        return self._output_feature_details

    @output_feature_details.setter
    def output_feature_details(self, output_feature_details):
        """
        Sets the output_feature_details of this UpdateFeatureGroupDetails.

        :param output_feature_details: The output_feature_details of this UpdateFeatureGroupDetails.
        :type: oci.feature_store.models.OutputFeatureDetailCollection
        """
        self._output_feature_details = output_feature_details

    @property
    def lifecycle_state(self):
        """
        Gets the lifecycle_state of this UpdateFeatureGroupDetails.
        The current state of the feature group.

        Allowed values for this property are: "CREATING", "UPDATING", "ACTIVE", "DELETING", "DELETED", "FAILED", "NEEDS_ATTENTION"


        :return: The lifecycle_state of this UpdateFeatureGroupDetails.
        :rtype: str
        """
        return self._lifecycle_state

    @lifecycle_state.setter
    def lifecycle_state(self, lifecycle_state):
        """
        Sets the lifecycle_state of this UpdateFeatureGroupDetails.
        The current state of the feature group.


        :param lifecycle_state: The lifecycle_state of this UpdateFeatureGroupDetails.
        :type: str
        """
        allowed_values = ["CREATING", "UPDATING", "ACTIVE", "DELETING", "DELETED", "FAILED", "NEEDS_ATTENTION"]
        if not value_allowed_none_or_none_sentinel(lifecycle_state, allowed_values):
            raise ValueError(
                "Invalid value for `lifecycle_state`, must be None or one of {0}"
                .format(allowed_values)
            )
        self._lifecycle_state = lifecycle_state

    @property
    def expectation_details(self):
        """
        Gets the expectation_details of this UpdateFeatureGroupDetails.

        :return: The expectation_details of this UpdateFeatureGroupDetails.
        :rtype: oci.feature_store.models.ExpectationDetails
        """
        return self._expectation_details

    @expectation_details.setter
    def expectation_details(self, expectation_details):
        """
        Sets the expectation_details of this UpdateFeatureGroupDetails.

        :param expectation_details: The expectation_details of this UpdateFeatureGroupDetails.
        :type: oci.feature_store.models.ExpectationDetails
        """
        self._expectation_details = expectation_details

    @property
    def primary_keys(self):
        """
        Gets the primary_keys of this UpdateFeatureGroupDetails.

        :return: The primary_keys of this UpdateFeatureGroupDetails.
        :rtype: oci.feature_store.models.PrimaryKeyCollection
        """
        return self._primary_keys

    @primary_keys.setter
    def primary_keys(self, primary_keys):
        """
        Sets the primary_keys of this UpdateFeatureGroupDetails.

        :param primary_keys: The primary_keys of this UpdateFeatureGroupDetails.
        :type: oci.feature_store.models.PrimaryKeyCollection
        """
        self._primary_keys = primary_keys

    @property
    def partition_keys(self):
        """
        Gets the partition_keys of this UpdateFeatureGroupDetails.

        :return: The partition_keys of this UpdateFeatureGroupDetails.
        :rtype: oci.feature_store.models.PartitionKeyCollection
        """
        return self._partition_keys

    @partition_keys.setter
    def partition_keys(self, partition_keys):
        """
        Sets the partition_keys of this UpdateFeatureGroupDetails.

        :param partition_keys: The partition_keys of this UpdateFeatureGroupDetails.
        :type: oci.feature_store.models.PartitionKeyCollection
        """
        self._partition_keys = partition_keys

    @property
    def statistics_config(self):
        """
        Gets the statistics_config of this UpdateFeatureGroupDetails.

        :return: The statistics_config of this UpdateFeatureGroupDetails.
        :rtype: oci.feature_store.models.StatisticsConfig
        """
        return self._statistics_config

    @statistics_config.setter
    def statistics_config(self, statistics_config):
        """
        Sets the statistics_config of this UpdateFeatureGroupDetails.

        :param statistics_config: The statistics_config of this UpdateFeatureGroupDetails.
        :type: oci.feature_store.models.StatisticsConfig
        """
        self._statistics_config = statistics_config

    @property
    def is_infer_schema(self):
        """
        Gets the is_infer_schema of this UpdateFeatureGroupDetails.
        infer the schema or not


        :return: The is_infer_schema of this UpdateFeatureGroupDetails.
        :rtype: bool
        """
        return self._is_infer_schema

    @is_infer_schema.setter
    def is_infer_schema(self, is_infer_schema):
        """
        Sets the is_infer_schema of this UpdateFeatureGroupDetails.
        infer the schema or not


        :param is_infer_schema: The is_infer_schema of this UpdateFeatureGroupDetails.
        :type: bool
        """
        self._is_infer_schema = is_infer_schema

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
