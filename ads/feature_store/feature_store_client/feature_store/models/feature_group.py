# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class FeatureGroup(object):
    """
    FeatureGroup Description.
    """

    #: A constant which can be used with the lifecycle_state property of a FeatureGroup.
    #: This constant has a value of "ACTIVE"
    LIFECYCLE_STATE_ACTIVE = "ACTIVE"

    def __init__(self, **kwargs):
        """
        Initializes a new FeatureGroup object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param id:
            The value to assign to the id property of this FeatureGroup.
        :type id: str

        :param compartment_id:
            The value to assign to the compartment_id property of this FeatureGroup.
        :type compartment_id: str

        :param name:
            The value to assign to the name property of this FeatureGroup.
        :type name: str

        :param description:
            The value to assign to the description property of this FeatureGroup.
        :type description: str

        :param transformation_id:
            The value to assign to the transformation_id property of this FeatureGroup.
        :type transformation_id: str

        :param on_demand_transformation_id:
            The value to assign to the on_demand_transformation_id property of this FeatureGroup.
        :type on_demand_transformation_id: str

        :param entity_id:
            The value to assign to the entity_id property of this FeatureGroup.
        :type entity_id: str

        :param feature_store_id:
            The value to assign to the feature_store_id property of this FeatureGroup.
        :type feature_store_id: str

        :param primary_keys:
            The value to assign to the primary_keys property of this FeatureGroup.
        :type primary_keys: oci.feature_store.models.PrimaryKeyCollection

        :param partition_keys:
            The value to assign to the partition_keys property of this FeatureGroup.
        :type partition_keys: oci.feature_store.models.PartitionKeyCollection

        :param transformation_parameters:
            The value to assign to the transformation_parameters property of this FeatureGroup.
        :type transformation_parameters: str

        :param output_feature_details:
            The value to assign to the output_feature_details property of this FeatureGroup.
        :type output_feature_details: oci.feature_store.models.OutputFeatureDetailCollection

        :param datasets:
            The value to assign to the datasets property of this FeatureGroup.
        :type datasets: list[str]

        :param input_feature_details:
            The value to assign to the input_feature_details property of this FeatureGroup.
        :type input_feature_details: list[oci.feature_store.models.RawFeatureDetail]

        :param expectation_details:
            The value to assign to the expectation_details property of this FeatureGroup.
        :type expectation_details: oci.feature_store.models.ExpectationDetails

        :param statistics_config:
            The value to assign to the statistics_config property of this FeatureGroup.
        :type statistics_config: oci.feature_store.models.StatisticsConfig

        :param is_infer_schema:
            The value to assign to the is_infer_schema property of this FeatureGroup.
        :type is_infer_schema: bool

        :param time_created:
            The value to assign to the time_created property of this FeatureGroup.
        :type time_created: str

        :param time_updated:
            The value to assign to the time_updated property of this FeatureGroup.
        :type time_updated: str

        :param created_by:
            The value to assign to the created_by property of this FeatureGroup.
        :type created_by: str

        :param is_online_enabled:
            The value to assign to the is_online_enabled property of this FeatureGroup.
        :type is_online_enabled: bool

        :param is_offline_enabled:
            The value to assign to the is_offline_enabled property of this FeatureGroup.
        :type is_offline_enabled: bool

        :param updated_by:
            The value to assign to the updated_by property of this FeatureGroup.
        :type updated_by: str

        :param lifecycle_state:
            The value to assign to the lifecycle_state property of this FeatureGroup.
            Allowed values for this property are: "ACTIVE", 'UNKNOWN_ENUM_VALUE'.
            Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.
        :type lifecycle_state: str

        """
        self.swagger_types = {
            'id': 'str',
            'compartment_id': 'str',
            'name': 'str',
            'description': 'str',
            'transformation_id': 'str',
            'on_demand_transformation_id': 'str',
            'entity_id': 'str',
            'feature_store_id': 'str',
            'primary_keys': 'PrimaryKeyCollection',
            'partition_keys': 'PartitionKeyCollection',
            'transformation_parameters': 'str',
            'output_feature_details': 'OutputFeatureDetailCollection',
            'datasets': 'list[str]',
            'input_feature_details': 'list[RawFeatureDetail]',
            'expectation_details': 'ExpectationDetails',
            'statistics_config': 'StatisticsConfig',
            'is_infer_schema': 'bool',
            'time_created': 'str',
            'time_updated': 'str',
            'created_by': 'str',
            'is_online_enabled': 'bool',
            'is_offline_enabled': 'bool',
            'updated_by': 'str',
            'lifecycle_state': 'str'
        }

        self.attribute_map = {
            'id': 'id',
            'compartment_id': 'compartmentId',
            'name': 'name',
            'description': 'description',
            'transformation_id': 'transformationId',
            'on_demand_transformation_id': 'onDemandTransformationId',
            'entity_id': 'entityId',
            'feature_store_id': 'featureStoreId',
            'primary_keys': 'primaryKeys',
            'partition_keys': 'partitionKeys',
            'transformation_parameters': 'transformationParameters',
            'output_feature_details': 'outputFeatureDetails',
            'datasets': 'datasets',
            'input_feature_details': 'inputFeatureDetails',
            'expectation_details': 'expectationDetails',
            'statistics_config': 'statisticsConfig',
            'is_infer_schema': 'isInferSchema',
            'time_created': 'timeCreated',
            'time_updated': 'timeUpdated',
            'created_by': 'createdBy',
            'is_online_enabled': 'isOnlineEnabled',
            'is_offline_enabled': 'isOfflineEnabled',
            'updated_by': 'updatedBy',
            'lifecycle_state': 'lifecycleState'
        }

        self._id = None
        self._compartment_id = None
        self._name = None
        self._description = None
        self._transformation_id = None
        self._on_demand_transformation_id = None
        self._entity_id = None
        self._feature_store_id = None
        self._primary_keys = None
        self._partition_keys = None
        self._transformation_parameters = None
        self._output_feature_details = None
        self._datasets = None
        self._input_feature_details = None
        self._expectation_details = None
        self._statistics_config = None
        self._is_infer_schema = None
        self._time_created = None
        self._time_updated = None
        self._created_by = None
        self._is_online_enabled = None
        self._is_offline_enabled = None
        self._updated_by = None
        self._lifecycle_state = None

    @property
    def id(self):
        """
        **[Required]** Gets the id of this FeatureGroup.
        The Unique Oracle ID (OCID) that is immutable on creation.


        :return: The id of this FeatureGroup.
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """
        Sets the id of this FeatureGroup.
        The Unique Oracle ID (OCID) that is immutable on creation.


        :param id: The id of this FeatureGroup.
        :type: str
        """
        self._id = id

    @property
    def compartment_id(self):
        """
        **[Required]** Gets the compartment_id of this FeatureGroup.
        The OCID of the compartment containing the DataAsset.


        :return: The compartment_id of this FeatureGroup.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this FeatureGroup.
        The OCID of the compartment containing the DataAsset.


        :param compartment_id: The compartment_id of this FeatureGroup.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def name(self):
        """
        **[Required]** Gets the name of this FeatureGroup.
        A user-friendly unique name


        :return: The name of this FeatureGroup.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Sets the name of this FeatureGroup.
        A user-friendly unique name


        :param name: The name of this FeatureGroup.
        :type: str
        """
        self._name = name

    @property
    def description(self):
        """
        Gets the description of this FeatureGroup.
        A short description of the data asset.


        :return: The description of this FeatureGroup.
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """
        Sets the description of this FeatureGroup.
        A short description of the data asset.


        :param description: The description of this FeatureGroup.
        :type: str
        """
        self._description = description

    @property
    def transformation_id(self):
        """
        **[Required]** Gets the transformation_id of this FeatureGroup.
        The OCID for the transformation.


        :return: The transformation_id of this FeatureGroup.
        :rtype: str
        """
        return self._transformation_id

    @transformation_id.setter
    def transformation_id(self, transformation_id):
        """
        Sets the transformation_id of this FeatureGroup.
        The OCID for the transformation.


        :param transformation_id: The transformation_id of this FeatureGroup.
        :type: str
        """
        self._transformation_id = transformation_id

    @property
    def on_demand_transformation_id(self):
        """
        **[Required]** Gets the on_demand_transformation_id of this FeatureGroup.
        The OCID for the transformation.


        :return: The on_demand_transformation_id of this FeatureGroup.
        :rtype: str
        """
        return self._on_demand_transformation_id

    @on_demand_transformation_id.setter
    def on_demand_transformation_id(self, on_demand_transformation_id):
        """
        Sets the on_demand_transformation_id of this FeatureGroup.
        The OCID for the transformation.


        :param on_demand_transformation_id: The on_demand_transformation_id of this FeatureGroup.
        :type: str
        """
        self._on_demand_transformation_id = on_demand_transformation_id

    @property
    def entity_id(self):
        """
        **[Required]** Gets the entity_id of this FeatureGroup.
        The OCID for the entity.


        :return: The entity_id of this FeatureGroup.
        :rtype: str
        """
        return self._entity_id

    @entity_id.setter
    def entity_id(self, entity_id):
        """
        Sets the entity_id of this FeatureGroup.
        The OCID for the entity.


        :param entity_id: The entity_id of this FeatureGroup.
        :type: str
        """
        self._entity_id = entity_id

    @property
    def feature_store_id(self):
        """
        **[Required]** Gets the feature_store_id of this FeatureGroup.
        The OCID for the feature store.


        :return: The feature_store_id of this FeatureGroup.
        :rtype: str
        """
        return self._feature_store_id

    @feature_store_id.setter
    def feature_store_id(self, feature_store_id):
        """
        Sets the feature_store_id of this FeatureGroup.
        The OCID for the feature store.


        :param feature_store_id: The feature_store_id of this FeatureGroup.
        :type: str
        """
        self._feature_store_id = feature_store_id

    @property
    def primary_keys(self):
        """
        **[Required]** Gets the primary_keys of this FeatureGroup.

        :return: The primary_keys of this FeatureGroup.
        :rtype: oci.feature_store.models.PrimaryKeyCollection
        """
        return self._primary_keys

    @primary_keys.setter
    def primary_keys(self, primary_keys):
        """
        Sets the primary_keys of this FeatureGroup.

        :param primary_keys: The primary_keys of this FeatureGroup.
        :type: oci.feature_store.models.PrimaryKeyCollection
        """
        self._primary_keys = primary_keys

    @property
    def partition_keys(self):
        """
        **[Required]** Gets the partition_keys of this FeatureGroup.

        :return: The partition_keys of this FeatureGroup.
        :rtype: oci.feature_store.models.PartitionKeyCollection
        """
        return self._partition_keys

    @partition_keys.setter
    def partition_keys(self, partition_keys):
        """
        Sets the partition_keys of this FeatureGroup.

        :param partition_keys: The partition_keys of this FeatureGroup.
        :type: oci.feature_store.models.PartitionKeyCollection
        """
        self._partition_keys = partition_keys

    @property
    def transformation_parameters(self):
        """
        Gets the transformation_parameters of this FeatureGroup.
        Arguments for the transformation function.


        :return: The transformation_parameters of this FeatureGroup.
        :rtype: str
        """
        return self._transformation_parameters

    @transformation_parameters.setter
    def transformation_parameters(self, transformation_parameters):
        """
        Sets the transformation_parameters of this FeatureGroup.
        Arguments for the transformation function.


        :param transformation_parameters: The transformation_parameters of this FeatureGroup.
        :type: str
        """
        self._transformation_parameters = transformation_parameters

    @property
    def output_feature_details(self):
        """
        Gets the output_feature_details of this FeatureGroup.

        :return: The output_feature_details of this FeatureGroup.
        :rtype: oci.feature_store.models.OutputFeatureDetailCollection
        """
        return self._output_feature_details

    @output_feature_details.setter
    def output_feature_details(self, output_feature_details):
        """
        Sets the output_feature_details of this FeatureGroup.

        :param output_feature_details: The output_feature_details of this FeatureGroup.
        :type: oci.feature_store.models.OutputFeatureDetailCollection
        """
        self._output_feature_details = output_feature_details

    @property
    def datasets(self):
        """
        **[Required]** Gets the datasets of this FeatureGroup.
        datasets linked to a particular feature group


        :return: The datasets of this FeatureGroup.
        :rtype: list[str]
        """
        return self._datasets

    @datasets.setter
    def datasets(self, datasets):
        """
        Sets the datasets of this FeatureGroup.
        datasets linked to a particular feature group


        :param datasets: The datasets of this FeatureGroup.
        :type: list[str]
        """
        self._datasets = datasets

    @property
    def input_feature_details(self):
        """
        **[Required]** Gets the input_feature_details of this FeatureGroup.
        input feature group schema details


        :return: The input_feature_details of this FeatureGroup.
        :rtype: list[oci.feature_store.models.RawFeatureDetail]
        """
        return self._input_feature_details

    @input_feature_details.setter
    def input_feature_details(self, input_feature_details):
        """
        Sets the input_feature_details of this FeatureGroup.
        input feature group schema details


        :param input_feature_details: The input_feature_details of this FeatureGroup.
        :type: list[oci.feature_store.models.RawFeatureDetail]
        """
        self._input_feature_details = input_feature_details

    @property
    def expectation_details(self):
        """
        Gets the expectation_details of this FeatureGroup.

        :return: The expectation_details of this FeatureGroup.
        :rtype: oci.feature_store.models.ExpectationDetails
        """
        return self._expectation_details

    @expectation_details.setter
    def expectation_details(self, expectation_details):
        """
        Sets the expectation_details of this FeatureGroup.

        :param expectation_details: The expectation_details of this FeatureGroup.
        :type: oci.feature_store.models.ExpectationDetails
        """
        self._expectation_details = expectation_details

    @property
    def statistics_config(self):
        """
        Gets the statistics_config of this FeatureGroup.

        :return: The statistics_config of this FeatureGroup.
        :rtype: oci.feature_store.models.StatisticsConfig
        """
        return self._statistics_config

    @statistics_config.setter
    def statistics_config(self, statistics_config):
        """
        Sets the statistics_config of this FeatureGroup.

        :param statistics_config: The statistics_config of this FeatureGroup.
        :type: oci.feature_store.models.StatisticsConfig
        """
        self._statistics_config = statistics_config

    @property
    def is_infer_schema(self):
        """
        Gets the is_infer_schema of this FeatureGroup.
        infer the schema or not


        :return: The is_infer_schema of this FeatureGroup.
        :rtype: bool
        """
        return self._is_infer_schema

    @is_infer_schema.setter
    def is_infer_schema(self, is_infer_schema):
        """
        Sets the is_infer_schema of this FeatureGroup.
        infer the schema or not


        :param is_infer_schema: The is_infer_schema of this FeatureGroup.
        :type: bool
        """
        self._is_infer_schema = is_infer_schema

    @property
    def time_created(self):
        """
        **[Required]** Gets the time_created of this FeatureGroup.

        :return: The time_created of this FeatureGroup.
        :rtype: str
        """
        return self._time_created

    @time_created.setter
    def time_created(self, time_created):
        """
        Sets the time_created of this FeatureGroup.

        :param time_created: The time_created of this FeatureGroup.
        :type: str
        """
        self._time_created = time_created

    @property
    def time_updated(self):
        """
        **[Required]** Gets the time_updated of this FeatureGroup.

        :return: The time_updated of this FeatureGroup.
        :rtype: str
        """
        return self._time_updated

    @time_updated.setter
    def time_updated(self, time_updated):
        """
        Sets the time_updated of this FeatureGroup.

        :param time_updated: The time_updated of this FeatureGroup.
        :type: str
        """
        self._time_updated = time_updated

    @property
    def created_by(self):
        """
        **[Required]** Gets the created_by of this FeatureGroup.
        Feature defintion created by details


        :return: The created_by of this FeatureGroup.
        :rtype: str
        """
        return self._created_by

    @created_by.setter
    def created_by(self, created_by):
        """
        Sets the created_by of this FeatureGroup.
        Feature defintion created by details


        :param created_by: The created_by of this FeatureGroup.
        :type: str
        """
        self._created_by = created_by

    @property
    def is_online_enabled(self):
        """
        Gets the is_online_enabled of this FeatureGroup.
        online feature store enabled or not


        :return: The is_online_enabled of this FeatureGroup.
        :rtype: bool
        """
        return self._is_online_enabled

    @is_online_enabled.setter
    def is_online_enabled(self, is_online_enabled):
        """
        Sets the is_online_enabled of this FeatureGroup.
        online feature store enabled or not


        :param is_online_enabled: The is_online_enabled of this FeatureGroup.
        :type: bool
        """
        self._is_online_enabled = is_online_enabled

    @property
    def is_offline_enabled(self):
        """
        Gets the is_offline_enabled of this FeatureGroup.
        offline feature store enabled or not


        :return: The is_offline_enabled of this FeatureGroup.
        :rtype: bool
        """
        return self._is_offline_enabled

    @is_offline_enabled.setter
    def is_offline_enabled(self, is_offline_enabled):
        """
        Sets the is_offline_enabled of this FeatureGroup.
        offline feature store enabled or not


        :param is_offline_enabled: The is_offline_enabled of this FeatureGroup.
        :type: bool
        """
        self._is_offline_enabled = is_offline_enabled

    @property
    def updated_by(self):
        """
        **[Required]** Gets the updated_by of this FeatureGroup.
        Feature defintion updated by details


        :return: The updated_by of this FeatureGroup.
        :rtype: str
        """
        return self._updated_by

    @updated_by.setter
    def updated_by(self, updated_by):
        """
        Sets the updated_by of this FeatureGroup.
        Feature defintion updated by details


        :param updated_by: The updated_by of this FeatureGroup.
        :type: str
        """
        self._updated_by = updated_by

    @property
    def lifecycle_state(self):
        """
        **[Required]** Gets the lifecycle_state of this FeatureGroup.
        The current state of the feature group.

        Allowed values for this property are: "ACTIVE", 'UNKNOWN_ENUM_VALUE'.
        Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.


        :return: The lifecycle_state of this FeatureGroup.
        :rtype: str
        """
        return self._lifecycle_state

    @lifecycle_state.setter
    def lifecycle_state(self, lifecycle_state):
        """
        Sets the lifecycle_state of this FeatureGroup.
        The current state of the feature group.


        :param lifecycle_state: The lifecycle_state of this FeatureGroup.
        :type: str
        """
        allowed_values = ["ACTIVE"]
        if not value_allowed_none_or_none_sentinel(lifecycle_state, allowed_values):
            lifecycle_state = 'UNKNOWN_ENUM_VALUE'
        self._lifecycle_state = lifecycle_state

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
