# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class Dataset(object):
    """
    Description of FeatureStore Dataset.
    """

    #: A constant which can be used with the lifecycle_state property of a Dataset.
    #: This constant has a value of "ACTIVE"
    LIFECYCLE_STATE_ACTIVE = "ACTIVE"

    def __init__(self, **kwargs):
        """
        Initializes a new Dataset object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param id:
            The value to assign to the id property of this Dataset.
        :type id: str

        :param name:
            The value to assign to the name property of this Dataset.
        :type name: str

        :param compartment_id:
            The value to assign to the compartment_id property of this Dataset.
        :type compartment_id: str

        :param entity_id:
            The value to assign to the entity_id property of this Dataset.
        :type entity_id: str

        :param feature_store_id:
            The value to assign to the feature_store_id property of this Dataset.
        :type feature_store_id: str

        :param lifecycle_state:
            The value to assign to the lifecycle_state property of this Dataset.
            Allowed values for this property are: "ACTIVE", 'UNKNOWN_ENUM_VALUE'.
            Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.
        :type lifecycle_state: str

        :param time_created:
            The value to assign to the time_created property of this Dataset.
        :type time_created: str

        :param time_updated:
            The value to assign to the time_updated property of this Dataset.
        :type time_updated: str

        :param description:
            The value to assign to the description property of this Dataset.
        :type description: str

        :param created_by:
            The value to assign to the created_by property of this Dataset.
        :type created_by: str

        :param updated_by:
            The value to assign to the updated_by property of this Dataset.
        :type updated_by: str

        :param is_online_enabled:
            The value to assign to the is_online_enabled property of this Dataset.
        :type is_online_enabled: bool

        :param is_offline_enabled:
            The value to assign to the is_offline_enabled property of this Dataset.
        :type is_offline_enabled: bool

        :param query:
            The value to assign to the query property of this Dataset.
        :type query: str

        :param dataset_feature_groups:
            The value to assign to the dataset_feature_groups property of this Dataset.
        :type dataset_feature_groups: oci.feature_store.models.DatasetFeatureGroupCollection

        :param partition_keys:
            The value to assign to the partition_keys property of this Dataset.
        :type partition_keys: oci.feature_store.models.PartitionKeyCollection

        :param primary_keys:
            The value to assign to the primary_keys property of this Dataset.
        :type primary_keys: oci.feature_store.models.PrimaryKeyCollection

        :param expectation_details:
            The value to assign to the expectation_details property of this Dataset.
        :type expectation_details: oci.feature_store.models.ExpectationDetails

        :param statistics_config:
            The value to assign to the statistics_config property of this Dataset.
        :type statistics_config: oci.feature_store.models.StatisticsConfig

        :param output_feature_details:
            The value to assign to the output_feature_details property of this Dataset.
        :type output_feature_details: oci.feature_store.models.OutputFeatureDetailCollection

        :param model_details:
            The value to assign to the model_details property of this Dataset.
        :type model_details: oci.feature_store.models.ModelCollection

        :param freeform_tags:
            The value to assign to the freeform_tags property of this Dataset.
        :type freeform_tags: dict(str, str)

        :param defined_tags:
            The value to assign to the defined_tags property of this Dataset.
        :type defined_tags: dict(str, dict(str, object))

        :param system_tags:
            The value to assign to the system_tags property of this Dataset.
        :type system_tags: dict(str, dict(str, object))

        """
        self.swagger_types = {
            'id': 'str',
            'name': 'str',
            'compartment_id': 'str',
            'entity_id': 'str',
            'feature_store_id': 'str',
            'lifecycle_state': 'str',
            'time_created': 'str',
            'time_updated': 'str',
            'description': 'str',
            'created_by': 'str',
            'updated_by': 'str',
            'is_online_enabled': 'bool',
            'is_offline_enabled': 'bool',
            'query': 'str',
            'dataset_feature_groups': 'DatasetFeatureGroupCollection',
            'partition_keys': 'PartitionKeyCollection',
            'primary_keys': 'PrimaryKeyCollection',
            'expectation_details': 'ExpectationDetails',
            'statistics_config': 'StatisticsConfig',
            'output_feature_details': 'OutputFeatureDetailCollection',
            'model_details': 'ModelCollection',
            'freeform_tags': 'dict(str, str)',
            'defined_tags': 'dict(str, dict(str, object))',
            'system_tags': 'dict(str, dict(str, object))'
        }

        self.attribute_map = {
            'id': 'id',
            'name': 'name',
            'compartment_id': 'compartmentId',
            'entity_id': 'entityId',
            'feature_store_id': 'featureStoreId',
            'lifecycle_state': 'lifecycleState',
            'time_created': 'timeCreated',
            'time_updated': 'timeUpdated',
            'description': 'description',
            'created_by': 'createdBy',
            'updated_by': 'updatedBy',
            'is_online_enabled': 'isOnlineEnabled',
            'is_offline_enabled': 'isOfflineEnabled',
            'query': 'query',
            'dataset_feature_groups': 'datasetFeatureGroups',
            'partition_keys': 'partitionKeys',
            'primary_keys': 'primaryKeys',
            'expectation_details': 'expectationDetails',
            'statistics_config': 'statisticsConfig',
            'output_feature_details': 'outputFeatureDetails',
            'model_details': 'modelDetails',
            'freeform_tags': 'freeformTags',
            'defined_tags': 'definedTags',
            'system_tags': 'systemTags'
        }

        self._id = None
        self._name = None
        self._compartment_id = None
        self._entity_id = None
        self._feature_store_id = None
        self._lifecycle_state = None
        self._time_created = None
        self._time_updated = None
        self._description = None
        self._created_by = None
        self._updated_by = None
        self._is_online_enabled = None
        self._is_offline_enabled = None
        self._query = None
        self._dataset_feature_groups = None
        self._partition_keys = None
        self._primary_keys = None
        self._expectation_details = None
        self._statistics_config = None
        self._output_feature_details = None
        self._model_details = None
        self._freeform_tags = None
        self._defined_tags = None
        self._system_tags = None

    @property
    def id(self):
        """
        **[Required]** Gets the id of this Dataset.
        The Unique Oracle ID (OCID) that is immutable on creation.


        :return: The id of this Dataset.
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """
        Sets the id of this Dataset.
        The Unique Oracle ID (OCID) that is immutable on creation.


        :param id: The id of this Dataset.
        :type: str
        """
        self._id = id

    @property
    def name(self):
        """
        **[Required]** Gets the name of this Dataset.
        A user-friendly unique name


        :return: The name of this Dataset.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Sets the name of this Dataset.
        A user-friendly unique name


        :param name: The name of this Dataset.
        :type: str
        """
        self._name = name

    @property
    def compartment_id(self):
        """
        **[Required]** Gets the compartment_id of this Dataset.
        The OCID of the compartment containing the FeatureStore datset.


        :return: The compartment_id of this Dataset.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this Dataset.
        The OCID of the compartment containing the FeatureStore datset.


        :param compartment_id: The compartment_id of this Dataset.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def entity_id(self):
        """
        **[Required]** Gets the entity_id of this Dataset.
        The OCID of the Data Science entity linked with dataset.


        :return: The entity_id of this Dataset.
        :rtype: str
        """
        return self._entity_id

    @entity_id.setter
    def entity_id(self, entity_id):
        """
        Sets the entity_id of this Dataset.
        The OCID of the Data Science entity linked with dataset.


        :param entity_id: The entity_id of this Dataset.
        :type: str
        """
        self._entity_id = entity_id

    @property
    def feature_store_id(self):
        """
        **[Required]** Gets the feature_store_id of this Dataset.
        The OCID of the Data Science Project containing the FeatureStore dataset.


        :return: The feature_store_id of this Dataset.
        :rtype: str
        """
        return self._feature_store_id

    @feature_store_id.setter
    def feature_store_id(self, feature_store_id):
        """
        Sets the feature_store_id of this Dataset.
        The OCID of the Data Science Project containing the FeatureStore dataset.


        :param feature_store_id: The feature_store_id of this Dataset.
        :type: str
        """
        self._feature_store_id = feature_store_id

    @property
    def lifecycle_state(self):
        """
        **[Required]** Gets the lifecycle_state of this Dataset.
        The current state of the dataset

        Allowed values for this property are: "ACTIVE", 'UNKNOWN_ENUM_VALUE'.
        Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.


        :return: The lifecycle_state of this Dataset.
        :rtype: str
        """
        return self._lifecycle_state

    @lifecycle_state.setter
    def lifecycle_state(self, lifecycle_state):
        """
        Sets the lifecycle_state of this Dataset.
        The current state of the dataset


        :param lifecycle_state: The lifecycle_state of this Dataset.
        :type: str
        """
        allowed_values = ["ACTIVE"]
        if not value_allowed_none_or_none_sentinel(lifecycle_state, allowed_values):
            lifecycle_state = 'UNKNOWN_ENUM_VALUE'
        self._lifecycle_state = lifecycle_state

    @property
    def time_created(self):
        """
        **[Required]** Gets the time_created of this Dataset.

        :return: The time_created of this Dataset.
        :rtype: str
        """
        return self._time_created

    @time_created.setter
    def time_created(self, time_created):
        """
        Sets the time_created of this Dataset.

        :param time_created: The time_created of this Dataset.
        :type: str
        """
        self._time_created = time_created

    @property
    def time_updated(self):
        """
        **[Required]** Gets the time_updated of this Dataset.

        :return: The time_updated of this Dataset.
        :rtype: str
        """
        return self._time_updated

    @time_updated.setter
    def time_updated(self, time_updated):
        """
        Sets the time_updated of this Dataset.

        :param time_updated: The time_updated of this Dataset.
        :type: str
        """
        self._time_updated = time_updated

    @property
    def description(self):
        """
        Gets the description of this Dataset.
        feature store entity description


        :return: The description of this Dataset.
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """
        Sets the description of this Dataset.
        feature store entity description


        :param description: The description of this Dataset.
        :type: str
        """
        self._description = description

    @property
    def created_by(self):
        """
        **[Required]** Gets the created_by of this Dataset.
        feature store dataset created by details


        :return: The created_by of this Dataset.
        :rtype: str
        """
        return self._created_by

    @created_by.setter
    def created_by(self, created_by):
        """
        Sets the created_by of this Dataset.
        feature store dataset created by details


        :param created_by: The created_by of this Dataset.
        :type: str
        """
        self._created_by = created_by

    @property
    def updated_by(self):
        """
        **[Required]** Gets the updated_by of this Dataset.
        feature store dataset updated by details


        :return: The updated_by of this Dataset.
        :rtype: str
        """
        return self._updated_by

    @updated_by.setter
    def updated_by(self, updated_by):
        """
        Sets the updated_by of this Dataset.
        feature store dataset updated by details


        :param updated_by: The updated_by of this Dataset.
        :type: str
        """
        self._updated_by = updated_by

    @property
    def is_online_enabled(self):
        """
        Gets the is_online_enabled of this Dataset.
        online feature store enabled or not


        :return: The is_online_enabled of this Dataset.
        :rtype: bool
        """
        return self._is_online_enabled

    @is_online_enabled.setter
    def is_online_enabled(self, is_online_enabled):
        """
        Sets the is_online_enabled of this Dataset.
        online feature store enabled or not


        :param is_online_enabled: The is_online_enabled of this Dataset.
        :type: bool
        """
        self._is_online_enabled = is_online_enabled

    @property
    def is_offline_enabled(self):
        """
        Gets the is_offline_enabled of this Dataset.
        offline feature store enabled or not


        :return: The is_offline_enabled of this Dataset.
        :rtype: bool
        """
        return self._is_offline_enabled

    @is_offline_enabled.setter
    def is_offline_enabled(self, is_offline_enabled):
        """
        Sets the is_offline_enabled of this Dataset.
        offline feature store enabled or not


        :param is_offline_enabled: The is_offline_enabled of this Dataset.
        :type: bool
        """
        self._is_offline_enabled = is_offline_enabled

    @property
    def query(self):
        """
        Gets the query of this Dataset.
        SQL query which will be executed to create the dataset


        :return: The query of this Dataset.
        :rtype: str
        """
        return self._query

    @query.setter
    def query(self, query):
        """
        Sets the query of this Dataset.
        SQL query which will be executed to create the dataset


        :param query: The query of this Dataset.
        :type: str
        """
        self._query = query

    @property
    def dataset_feature_groups(self):
        """
        Gets the dataset_feature_groups of this Dataset.

        :return: The dataset_feature_groups of this Dataset.
        :rtype: oci.feature_store.models.DatasetFeatureGroupCollection
        """
        return self._dataset_feature_groups

    @dataset_feature_groups.setter
    def dataset_feature_groups(self, dataset_feature_groups):
        """
        Sets the dataset_feature_groups of this Dataset.

        :param dataset_feature_groups: The dataset_feature_groups of this Dataset.
        :type: oci.feature_store.models.DatasetFeatureGroupCollection
        """
        self._dataset_feature_groups = dataset_feature_groups

    @property
    def partition_keys(self):
        """
        **[Required]** Gets the partition_keys of this Dataset.

        :return: The partition_keys of this Dataset.
        :rtype: oci.feature_store.models.PartitionKeyCollection
        """
        return self._partition_keys

    @partition_keys.setter
    def partition_keys(self, partition_keys):
        """
        Sets the partition_keys of this Dataset.

        :param partition_keys: The partition_keys of this Dataset.
        :type: oci.feature_store.models.PartitionKeyCollection
        """
        self._partition_keys = partition_keys

    @property
    def primary_keys(self):
        """
        Gets the primary_keys of this Dataset.

        :return: The primary_keys of this Dataset.
        :rtype: oci.feature_store.models.PrimaryKeyCollection
        """
        return self._primary_keys

    @primary_keys.setter
    def primary_keys(self, primary_keys):
        """
        Sets the primary_keys of this Dataset.

        :param primary_keys: The primary_keys of this Dataset.
        :type: oci.feature_store.models.PrimaryKeyCollection
        """
        self._primary_keys = primary_keys

    @property
    def expectation_details(self):
        """
        Gets the expectation_details of this Dataset.

        :return: The expectation_details of this Dataset.
        :rtype: oci.feature_store.models.ExpectationDetails
        """
        return self._expectation_details

    @expectation_details.setter
    def expectation_details(self, expectation_details):
        """
        Sets the expectation_details of this Dataset.

        :param expectation_details: The expectation_details of this Dataset.
        :type: oci.feature_store.models.ExpectationDetails
        """
        self._expectation_details = expectation_details

    @property
    def statistics_config(self):
        """
        Gets the statistics_config of this Dataset.

        :return: The statistics_config of this Dataset.
        :rtype: oci.feature_store.models.StatisticsConfig
        """
        return self._statistics_config

    @statistics_config.setter
    def statistics_config(self, statistics_config):
        """
        Sets the statistics_config of this Dataset.

        :param statistics_config: The statistics_config of this Dataset.
        :type: oci.feature_store.models.StatisticsConfig
        """
        self._statistics_config = statistics_config

    @property
    def output_feature_details(self):
        """
        Gets the output_feature_details of this Dataset.

        :return: The output_feature_details of this Dataset.
        :rtype: oci.feature_store.models.OutputFeatureDetailCollection
        """
        return self._output_feature_details

    @output_feature_details.setter
    def output_feature_details(self, output_feature_details):
        """
        Sets the output_feature_details of this Dataset.

        :param output_feature_details: The output_feature_details of this Dataset.
        :type: oci.feature_store.models.OutputFeatureDetailCollection
        """
        self._output_feature_details = output_feature_details

    @property
    def model_details(self):
        """
        Gets the model_details of this Dataset.

        :return: The model_details of this Dataset.
        :rtype: oci.feature_store.models.ModelCollection
        """
        return self._model_details

    @model_details.setter
    def model_details(self, model_details):
        """
        Sets the model_details of this Dataset.

        :param model_details: The model_details of this Dataset.
        :type: oci.feature_store.models.ModelCollection
        """
        self._model_details = model_details

    @property
    def freeform_tags(self):
        """
        Gets the freeform_tags of this Dataset.
        Simple key-value pair that is applied without any predefined name, type or scope. Exists for cross-compatibility only.
        Example: `{\"bar-key\": \"value\"}`


        :return: The freeform_tags of this Dataset.
        :rtype: dict(str, str)
        """
        return self._freeform_tags

    @freeform_tags.setter
    def freeform_tags(self, freeform_tags):
        """
        Sets the freeform_tags of this Dataset.
        Simple key-value pair that is applied without any predefined name, type or scope. Exists for cross-compatibility only.
        Example: `{\"bar-key\": \"value\"}`


        :param freeform_tags: The freeform_tags of this Dataset.
        :type: dict(str, str)
        """
        self._freeform_tags = freeform_tags

    @property
    def defined_tags(self):
        """
        Gets the defined_tags of this Dataset.
        Defined tags for this resource. Each key is predefined and scoped to a namespace.
        Example: `{\"foo-namespace\": {\"bar-key\": \"value\"}}`


        :return: The defined_tags of this Dataset.
        :rtype: dict(str, dict(str, object))
        """
        return self._defined_tags

    @defined_tags.setter
    def defined_tags(self, defined_tags):
        """
        Sets the defined_tags of this Dataset.
        Defined tags for this resource. Each key is predefined and scoped to a namespace.
        Example: `{\"foo-namespace\": {\"bar-key\": \"value\"}}`


        :param defined_tags: The defined_tags of this Dataset.
        :type: dict(str, dict(str, object))
        """
        self._defined_tags = defined_tags

    @property
    def system_tags(self):
        """
        Gets the system_tags of this Dataset.
        System tags for this resource. Each key is predefined and scoped to a namespace.
        Example: `{\"orcl-cloud\": {\"free-tier-retained\": \"true\"}}`


        :return: The system_tags of this Dataset.
        :rtype: dict(str, dict(str, object))
        """
        return self._system_tags

    @system_tags.setter
    def system_tags(self, system_tags):
        """
        Sets the system_tags of this Dataset.
        System tags for this resource. Each key is predefined and scoped to a namespace.
        Example: `{\"orcl-cloud\": {\"free-tier-retained\": \"true\"}}`


        :param system_tags: The system_tags of this Dataset.
        :type: dict(str, dict(str, object))
        """
        self._system_tags = system_tags

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
