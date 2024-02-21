# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class CreateFeatureGroupDetails(object):
    """
    Parameters needed to create a new feature group.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new CreateFeatureGroupDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param name:
            The value to assign to the name property of this CreateFeatureGroupDetails.
        :type name: str

        :param compartment_id:
            The value to assign to the compartment_id property of this CreateFeatureGroupDetails.
        :type compartment_id: str

        :param description:
            The value to assign to the description property of this CreateFeatureGroupDetails.
        :type description: str

        :param on_demand_transformation_id:
            The value to assign to the on_demand_transformation_id property of this CreateFeatureGroupDetails.
        :type on_demand_transformation_id: str

        :param transformation_id:
            The value to assign to the transformation_id property of this CreateFeatureGroupDetails.
        :type transformation_id: str

        :param entity_id:
            The value to assign to the entity_id property of this CreateFeatureGroupDetails.
        :type entity_id: str

        :param feature_store_id:
            The value to assign to the feature_store_id property of this CreateFeatureGroupDetails.
        :type feature_store_id: str

        :param input_feature_details:
            The value to assign to the input_feature_details property of this CreateFeatureGroupDetails.
        :type input_feature_details: list[oci.feature_store.models.RawFeatureDetail]

        :param transformation_parameters:
            The value to assign to the transformation_parameters property of this CreateFeatureGroupDetails.
        :type transformation_parameters: str

        :param expectation_details:
            The value to assign to the expectation_details property of this CreateFeatureGroupDetails.
        :type expectation_details: oci.feature_store.models.ExpectationDetails

        :param statistics_config:
            The value to assign to the statistics_config property of this CreateFeatureGroupDetails.
        :type statistics_config: oci.feature_store.models.StatisticsConfig

        :param primary_keys:
            The value to assign to the primary_keys property of this CreateFeatureGroupDetails.
        :type primary_keys: oci.feature_store.models.PrimaryKeyCollection

        :param partition_keys:
            The value to assign to the partition_keys property of this CreateFeatureGroupDetails.
        :type partition_keys: oci.feature_store.models.PartitionKeyCollection

        :param is_infer_schema:
            The value to assign to the is_infer_schema property of this CreateFeatureGroupDetails.
        :type is_infer_schema: bool

        :param is_online_enabled:
            The value to assign to the is_online_enabled property of this CreateFeatureGroupDetails.
        :type is_online_enabled: bool

        :param is_offline_enabled:
            The value to assign to the is_offline_enabled property of this CreateFeatureGroupDetails.
        :type is_offline_enabled: bool

        """
        self.swagger_types = {
            'name': 'str',
            'compartment_id': 'str',
            'description': 'str',
            'on_demand_transformation_id': 'str',
            'transformation_id': 'str',
            'entity_id': 'str',
            'feature_store_id': 'str',
            'input_feature_details': 'list[RawFeatureDetail]',
            'transformation_parameters': 'str',
            'expectation_details': 'ExpectationDetails',
            'statistics_config': 'StatisticsConfig',
            'primary_keys': 'PrimaryKeyCollection',
            'partition_keys': 'PartitionKeyCollection',
            'is_infer_schema': 'bool',
            'is_online_enabled': 'bool',
            'is_offline_enabled': 'bool'
        }

        self.attribute_map = {
            'name': 'name',
            'compartment_id': 'compartmentId',
            'description': 'description',
            'on_demand_transformation_id': 'onDemandTransformationId',
            'transformation_id': 'transformationId',
            'entity_id': 'entityId',
            'feature_store_id': 'featureStoreId',
            'input_feature_details': 'inputFeatureDetails',
            'transformation_parameters': 'transformationParameters',
            'expectation_details': 'expectationDetails',
            'statistics_config': 'statisticsConfig',
            'primary_keys': 'primaryKeys',
            'partition_keys': 'partitionKeys',
            'is_infer_schema': 'isInferSchema',
            'is_online_enabled': 'isOnlineEnabled',
            'is_offline_enabled': 'isOfflineEnabled'
        }

        self._name = None
        self._compartment_id = None
        self._description = None
        self._on_demand_transformation_id = None
        self._transformation_id = None
        self._entity_id = None
        self._feature_store_id = None
        self._input_feature_details = None
        self._transformation_parameters = None
        self._expectation_details = None
        self._statistics_config = None
        self._primary_keys = None
        self._partition_keys = None
        self._is_infer_schema = None
        self._is_online_enabled = None
        self._is_offline_enabled = None

    @property
    def name(self):
        """
        **[Required]** Gets the name of this CreateFeatureGroupDetails.
        A user-friendly unique name for the resource.


        :return: The name of this CreateFeatureGroupDetails.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Sets the name of this CreateFeatureGroupDetails.
        A user-friendly unique name for the resource.


        :param name: The name of this CreateFeatureGroupDetails.
        :type: str
        """
        self._name = name

    @property
    def compartment_id(self):
        """
        **[Required]** Gets the compartment_id of this CreateFeatureGroupDetails.
        The OCID for the data asset's compartment.


        :return: The compartment_id of this CreateFeatureGroupDetails.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this CreateFeatureGroupDetails.
        The OCID for the data asset's compartment.


        :param compartment_id: The compartment_id of this CreateFeatureGroupDetails.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def description(self):
        """
        Gets the description of this CreateFeatureGroupDetails.
        A short description of the Ai data asset


        :return: The description of this CreateFeatureGroupDetails.
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """
        Sets the description of this CreateFeatureGroupDetails.
        A short description of the Ai data asset


        :param description: The description of this CreateFeatureGroupDetails.
        :type: str
        """
        self._description = description

    @property
    def on_demand_transformation_id(self):
        """
        **[Required]** Gets the on_demand_transformation_id of this CreateFeatureGroupDetails.
        The OCID for the on demand transformation.


        :return: The on_demand_transformation_id of this CreateFeatureGroupDetails.
        :rtype: str
        """
        return self._on_demand_transformation_id

    @on_demand_transformation_id.setter
    def on_demand_transformation_id(self, on_demand_transformation_id):
        """
        Sets the on_demand_transformation_id of this CreateFeatureGroupDetails.
        The OCID for the on demand transformation.


        :param on_demand_transformation_id: The on_demand_transformation_id of this CreateFeatureGroupDetails.
        :type: str
        """
        self._on_demand_transformation_id = on_demand_transformation_id

    @property
    def transformation_id(self):
        """
        **[Required]** Gets the transformation_id of this CreateFeatureGroupDetails.
        The OCID for the transformation.


        :return: The transformation_id of this CreateFeatureGroupDetails.
        :rtype: str
        """
        return self._transformation_id

    @transformation_id.setter
    def transformation_id(self, transformation_id):
        """
        Sets the transformation_id of this CreateFeatureGroupDetails.
        The OCID for the transformation.


        :param transformation_id: The transformation_id of this CreateFeatureGroupDetails.
        :type: str
        """
        self._transformation_id = transformation_id

    @property
    def entity_id(self):
        """
        **[Required]** Gets the entity_id of this CreateFeatureGroupDetails.
        The OCID for the entity.


        :return: The entity_id of this CreateFeatureGroupDetails.
        :rtype: str
        """
        return self._entity_id

    @entity_id.setter
    def entity_id(self, entity_id):
        """
        Sets the entity_id of this CreateFeatureGroupDetails.
        The OCID for the entity.


        :param entity_id: The entity_id of this CreateFeatureGroupDetails.
        :type: str
        """
        self._entity_id = entity_id

    @property
    def feature_store_id(self):
        """
        **[Required]** Gets the feature_store_id of this CreateFeatureGroupDetails.
        The OCID for the feature store.


        :return: The feature_store_id of this CreateFeatureGroupDetails.
        :rtype: str
        """
        return self._feature_store_id

    @feature_store_id.setter
    def feature_store_id(self, feature_store_id):
        """
        Sets the feature_store_id of this CreateFeatureGroupDetails.
        The OCID for the feature store.


        :param feature_store_id: The feature_store_id of this CreateFeatureGroupDetails.
        :type: str
        """
        self._feature_store_id = feature_store_id

    @property
    def input_feature_details(self):
        """
        **[Required]** Gets the input_feature_details of this CreateFeatureGroupDetails.
        input feature group schema details


        :return: The input_feature_details of this CreateFeatureGroupDetails.
        :rtype: list[oci.feature_store.models.RawFeatureDetail]
        """
        return self._input_feature_details

    @input_feature_details.setter
    def input_feature_details(self, input_feature_details):
        """
        Sets the input_feature_details of this CreateFeatureGroupDetails.
        input feature group schema details


        :param input_feature_details: The input_feature_details of this CreateFeatureGroupDetails.
        :type: list[oci.feature_store.models.RawFeatureDetail]
        """
        self._input_feature_details = input_feature_details

    @property
    def transformation_parameters(self):
        """
        Gets the transformation_parameters of this CreateFeatureGroupDetails.
        Arguments for the transformation function.


        :return: The transformation_parameters of this CreateFeatureGroupDetails.
        :rtype: str
        """
        return self._transformation_parameters

    @transformation_parameters.setter
    def transformation_parameters(self, transformation_parameters):
        """
        Sets the transformation_parameters of this CreateFeatureGroupDetails.
        Arguments for the transformation function.


        :param transformation_parameters: The transformation_parameters of this CreateFeatureGroupDetails.
        :type: str
        """
        self._transformation_parameters = transformation_parameters

    @property
    def expectation_details(self):
        """
        Gets the expectation_details of this CreateFeatureGroupDetails.

        :return: The expectation_details of this CreateFeatureGroupDetails.
        :rtype: oci.feature_store.models.ExpectationDetails
        """
        return self._expectation_details

    @expectation_details.setter
    def expectation_details(self, expectation_details):
        """
        Sets the expectation_details of this CreateFeatureGroupDetails.

        :param expectation_details: The expectation_details of this CreateFeatureGroupDetails.
        :type: oci.feature_store.models.ExpectationDetails
        """
        self._expectation_details = expectation_details

    @property
    def statistics_config(self):
        """
        Gets the statistics_config of this CreateFeatureGroupDetails.

        :return: The statistics_config of this CreateFeatureGroupDetails.
        :rtype: oci.feature_store.models.StatisticsConfig
        """
        return self._statistics_config

    @statistics_config.setter
    def statistics_config(self, statistics_config):
        """
        Sets the statistics_config of this CreateFeatureGroupDetails.

        :param statistics_config: The statistics_config of this CreateFeatureGroupDetails.
        :type: oci.feature_store.models.StatisticsConfig
        """
        self._statistics_config = statistics_config

    @property
    def primary_keys(self):
        """
        **[Required]** Gets the primary_keys of this CreateFeatureGroupDetails.

        :return: The primary_keys of this CreateFeatureGroupDetails.
        :rtype: oci.feature_store.models.PrimaryKeyCollection
        """
        return self._primary_keys

    @primary_keys.setter
    def primary_keys(self, primary_keys):
        """
        Sets the primary_keys of this CreateFeatureGroupDetails.

        :param primary_keys: The primary_keys of this CreateFeatureGroupDetails.
        :type: oci.feature_store.models.PrimaryKeyCollection
        """
        self._primary_keys = primary_keys

    @property
    def partition_keys(self):
        """
        **[Required]** Gets the partition_keys of this CreateFeatureGroupDetails.

        :return: The partition_keys of this CreateFeatureGroupDetails.
        :rtype: oci.feature_store.models.PartitionKeyCollection
        """
        return self._partition_keys

    @partition_keys.setter
    def partition_keys(self, partition_keys):
        """
        Sets the partition_keys of this CreateFeatureGroupDetails.

        :param partition_keys: The partition_keys of this CreateFeatureGroupDetails.
        :type: oci.feature_store.models.PartitionKeyCollection
        """
        self._partition_keys = partition_keys

    @property
    def is_infer_schema(self):
        """
        Gets the is_infer_schema of this CreateFeatureGroupDetails.
        infer the schema or not


        :return: The is_infer_schema of this CreateFeatureGroupDetails.
        :rtype: bool
        """
        return self._is_infer_schema

    @is_infer_schema.setter
    def is_infer_schema(self, is_infer_schema):
        """
        Sets the is_infer_schema of this CreateFeatureGroupDetails.
        infer the schema or not


        :param is_infer_schema: The is_infer_schema of this CreateFeatureGroupDetails.
        :type: bool
        """
        self._is_infer_schema = is_infer_schema

    @property
    def is_online_enabled(self):
        """
        Gets the is_online_enabled of this CreateFeatureGroupDetails.
        online feature store enabled or not


        :return: The is_online_enabled of this CreateFeatureGroupDetails.
        :rtype: bool
        """
        return self._is_online_enabled

    @is_online_enabled.setter
    def is_online_enabled(self, is_online_enabled):
        """
        Sets the is_online_enabled of this CreateFeatureGroupDetails.
        online feature store enabled or not


        :param is_online_enabled: The is_online_enabled of this CreateFeatureGroupDetails.
        :type: bool
        """
        self._is_online_enabled = is_online_enabled

    @property
    def is_offline_enabled(self):
        """
        Gets the is_offline_enabled of this CreateFeatureGroupDetails.
        offline feature store enabled or not


        :return: The is_offline_enabled of this CreateFeatureGroupDetails.
        :rtype: bool
        """
        return self._is_offline_enabled

    @is_offline_enabled.setter
    def is_offline_enabled(self, is_offline_enabled):
        """
        Sets the is_offline_enabled of this CreateFeatureGroupDetails.
        offline feature store enabled or not


        :param is_offline_enabled: The is_offline_enabled of this CreateFeatureGroupDetails.
        :type: bool
        """
        self._is_offline_enabled = is_offline_enabled

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
