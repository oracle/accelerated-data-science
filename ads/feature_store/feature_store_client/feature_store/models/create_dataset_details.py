# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class CreateDatasetDetails(object):
    """
    The information about new FeatureStore entity.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new CreateDatasetDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param name:
            The value to assign to the name property of this CreateDatasetDetails.
        :type name: str

        :param compartment_id:
            The value to assign to the compartment_id property of this CreateDatasetDetails.
        :type compartment_id: str

        :param feature_store_id:
            The value to assign to the feature_store_id property of this CreateDatasetDetails.
        :type feature_store_id: str

        :param entity_id:
            The value to assign to the entity_id property of this CreateDatasetDetails.
        :type entity_id: str

        :param description:
            The value to assign to the description property of this CreateDatasetDetails.
        :type description: str

        :param query:
            The value to assign to the query property of this CreateDatasetDetails.
        :type query: str

        :param is_online_enabled:
            The value to assign to the is_online_enabled property of this CreateDatasetDetails.
        :type is_online_enabled: bool

        :param is_offline_enabled:
            The value to assign to the is_offline_enabled property of this CreateDatasetDetails.
        :type is_offline_enabled: bool

        :param dataset_feature_groups:
            The value to assign to the dataset_feature_groups property of this CreateDatasetDetails.
        :type dataset_feature_groups: oci.feature_store.models.CreateDatasetFeatureGroupCollectionDetails

        :param partition_keys:
            The value to assign to the partition_keys property of this CreateDatasetDetails.
        :type partition_keys: oci.feature_store.models.PartitionKeyCollection

        :param primary_keys:
            The value to assign to the primary_keys property of this CreateDatasetDetails.
        :type primary_keys: oci.feature_store.models.PrimaryKeyCollection

        :param expectation_details:
            The value to assign to the expectation_details property of this CreateDatasetDetails.
        :type expectation_details: oci.feature_store.models.ExpectationDetails

        :param statistics_config:
            The value to assign to the statistics_config property of this CreateDatasetDetails.
        :type statistics_config: oci.feature_store.models.StatisticsConfig

        :param model_details:
            The value to assign to the model_details property of this CreateDatasetDetails.
        :type model_details: oci.feature_store.models.ModelCollection

        :param freeform_tags:
            The value to assign to the freeform_tags property of this CreateDatasetDetails.
        :type freeform_tags: dict(str, str)

        :param defined_tags:
            The value to assign to the defined_tags property of this CreateDatasetDetails.
        :type defined_tags: dict(str, dict(str, object))

        """
        self.swagger_types = {
            'name': 'str',
            'compartment_id': 'str',
            'feature_store_id': 'str',
            'entity_id': 'str',
            'description': 'str',
            'query': 'str',
            'is_online_enabled': 'bool',
            'is_offline_enabled': 'bool',
            'dataset_feature_groups': 'CreateDatasetFeatureGroupCollectionDetails',
            'partition_keys': 'PartitionKeyCollection',
            'primary_keys': 'PrimaryKeyCollection',
            'expectation_details': 'ExpectationDetails',
            'statistics_config': 'StatisticsConfig',
            'model_details': 'ModelCollection',
            'freeform_tags': 'dict(str, str)',
            'defined_tags': 'dict(str, dict(str, object))'
        }

        self.attribute_map = {
            'name': 'name',
            'compartment_id': 'compartmentId',
            'feature_store_id': 'featureStoreId',
            'entity_id': 'entityId',
            'description': 'description',
            'query': 'query',
            'is_online_enabled': 'isOnlineEnabled',
            'is_offline_enabled': 'isOfflineEnabled',
            'dataset_feature_groups': 'datasetFeatureGroups',
            'partition_keys': 'partitionKeys',
            'primary_keys': 'primaryKeys',
            'expectation_details': 'expectationDetails',
            'statistics_config': 'statisticsConfig',
            'model_details': 'modelDetails',
            'freeform_tags': 'freeformTags',
            'defined_tags': 'definedTags'
        }

        self._name = None
        self._compartment_id = None
        self._feature_store_id = None
        self._entity_id = None
        self._description = None
        self._query = None
        self._is_online_enabled = None
        self._is_offline_enabled = None
        self._dataset_feature_groups = None
        self._partition_keys = None
        self._primary_keys = None
        self._expectation_details = None
        self._statistics_config = None
        self._model_details = None
        self._freeform_tags = None
        self._defined_tags = None

    @property
    def name(self):
        """
        **[Required]** Gets the name of this CreateDatasetDetails.
        A user-friendly unique name for the resource.


        :return: The name of this CreateDatasetDetails.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Sets the name of this CreateDatasetDetails.
        A user-friendly unique name for the resource.


        :param name: The name of this CreateDatasetDetails.
        :type: str
        """
        self._name = name

    @property
    def compartment_id(self):
        """
        **[Required]** Gets the compartment_id of this CreateDatasetDetails.
        Compartment Identifier


        :return: The compartment_id of this CreateDatasetDetails.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this CreateDatasetDetails.
        Compartment Identifier


        :param compartment_id: The compartment_id of this CreateDatasetDetails.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def feature_store_id(self):
        """
        **[Required]** Gets the feature_store_id of this CreateDatasetDetails.
        Id of feature store


        :return: The feature_store_id of this CreateDatasetDetails.
        :rtype: str
        """
        return self._feature_store_id

    @feature_store_id.setter
    def feature_store_id(self, feature_store_id):
        """
        Sets the feature_store_id of this CreateDatasetDetails.
        Id of feature store


        :param feature_store_id: The feature_store_id of this CreateDatasetDetails.
        :type: str
        """
        self._feature_store_id = feature_store_id

    @property
    def entity_id(self):
        """
        **[Required]** Gets the entity_id of this CreateDatasetDetails.
        feature store entity id


        :return: The entity_id of this CreateDatasetDetails.
        :rtype: str
        """
        return self._entity_id

    @entity_id.setter
    def entity_id(self, entity_id):
        """
        Sets the entity_id of this CreateDatasetDetails.
        feature store entity id


        :param entity_id: The entity_id of this CreateDatasetDetails.
        :type: str
        """
        self._entity_id = entity_id

    @property
    def description(self):
        """
        Gets the description of this CreateDatasetDetails.
        feature store entity description


        :return: The description of this CreateDatasetDetails.
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """
        Sets the description of this CreateDatasetDetails.
        feature store entity description


        :param description: The description of this CreateDatasetDetails.
        :type: str
        """
        self._description = description

    @property
    def query(self):
        """
        **[Required]** Gets the query of this CreateDatasetDetails.
        SQL query which will be executed to create the dataset


        :return: The query of this CreateDatasetDetails.
        :rtype: str
        """
        return self._query

    @query.setter
    def query(self, query):
        """
        Sets the query of this CreateDatasetDetails.
        SQL query which will be executed to create the dataset


        :param query: The query of this CreateDatasetDetails.
        :type: str
        """
        self._query = query

    @property
    def is_online_enabled(self):
        """
        Gets the is_online_enabled of this CreateDatasetDetails.
        online feature store enabled or not


        :return: The is_online_enabled of this CreateDatasetDetails.
        :rtype: bool
        """
        return self._is_online_enabled

    @is_online_enabled.setter
    def is_online_enabled(self, is_online_enabled):
        """
        Sets the is_online_enabled of this CreateDatasetDetails.
        online feature store enabled or not


        :param is_online_enabled: The is_online_enabled of this CreateDatasetDetails.
        :type: bool
        """
        self._is_online_enabled = is_online_enabled

    @property
    def is_offline_enabled(self):
        """
        Gets the is_offline_enabled of this CreateDatasetDetails.
        offline feature store enabled or not


        :return: The is_offline_enabled of this CreateDatasetDetails.
        :rtype: bool
        """
        return self._is_offline_enabled

    @is_offline_enabled.setter
    def is_offline_enabled(self, is_offline_enabled):
        """
        Sets the is_offline_enabled of this CreateDatasetDetails.
        offline feature store enabled or not


        :param is_offline_enabled: The is_offline_enabled of this CreateDatasetDetails.
        :type: bool
        """
        self._is_offline_enabled = is_offline_enabled

    @property
    def dataset_feature_groups(self):
        """
        Gets the dataset_feature_groups of this CreateDatasetDetails.

        :return: The dataset_feature_groups of this CreateDatasetDetails.
        :rtype: oci.feature_store.models.CreateDatasetFeatureGroupCollectionDetails
        """
        return self._dataset_feature_groups

    @dataset_feature_groups.setter
    def dataset_feature_groups(self, dataset_feature_groups):
        """
        Sets the dataset_feature_groups of this CreateDatasetDetails.

        :param dataset_feature_groups: The dataset_feature_groups of this CreateDatasetDetails.
        :type: oci.feature_store.models.CreateDatasetFeatureGroupCollectionDetails
        """
        self._dataset_feature_groups = dataset_feature_groups

    @property
    def partition_keys(self):
        """
        **[Required]** Gets the partition_keys of this CreateDatasetDetails.

        :return: The partition_keys of this CreateDatasetDetails.
        :rtype: oci.feature_store.models.PartitionKeyCollection
        """
        return self._partition_keys

    @partition_keys.setter
    def partition_keys(self, partition_keys):
        """
        Sets the partition_keys of this CreateDatasetDetails.

        :param partition_keys: The partition_keys of this CreateDatasetDetails.
        :type: oci.feature_store.models.PartitionKeyCollection
        """
        self._partition_keys = partition_keys

    @property
    def primary_keys(self):
        """
        Gets the primary_keys of this CreateDatasetDetails.

        :return: The primary_keys of this CreateDatasetDetails.
        :rtype: oci.feature_store.models.PrimaryKeyCollection
        """
        return self._primary_keys

    @primary_keys.setter
    def primary_keys(self, primary_keys):
        """
        Sets the primary_keys of this CreateDatasetDetails.

        :param primary_keys: The primary_keys of this CreateDatasetDetails.
        :type: oci.feature_store.models.PrimaryKeyCollection
        """
        self._primary_keys = primary_keys

    @property
    def expectation_details(self):
        """
        Gets the expectation_details of this CreateDatasetDetails.

        :return: The expectation_details of this CreateDatasetDetails.
        :rtype: oci.feature_store.models.ExpectationDetails
        """
        return self._expectation_details

    @expectation_details.setter
    def expectation_details(self, expectation_details):
        """
        Sets the expectation_details of this CreateDatasetDetails.

        :param expectation_details: The expectation_details of this CreateDatasetDetails.
        :type: oci.feature_store.models.ExpectationDetails
        """
        self._expectation_details = expectation_details

    @property
    def statistics_config(self):
        """
        Gets the statistics_config of this CreateDatasetDetails.

        :return: The statistics_config of this CreateDatasetDetails.
        :rtype: oci.feature_store.models.StatisticsConfig
        """
        return self._statistics_config

    @statistics_config.setter
    def statistics_config(self, statistics_config):
        """
        Sets the statistics_config of this CreateDatasetDetails.

        :param statistics_config: The statistics_config of this CreateDatasetDetails.
        :type: oci.feature_store.models.StatisticsConfig
        """
        self._statistics_config = statistics_config

    @property
    def model_details(self):
        """
        Gets the model_details of this CreateDatasetDetails.

        :return: The model_details of this CreateDatasetDetails.
        :rtype: oci.feature_store.models.ModelCollection
        """
        return self._model_details

    @model_details.setter
    def model_details(self, model_details):
        """
        Sets the model_details of this CreateDatasetDetails.

        :param model_details: The model_details of this CreateDatasetDetails.
        :type: oci.feature_store.models.ModelCollection
        """
        self._model_details = model_details

    @property
    def freeform_tags(self):
        """
        Gets the freeform_tags of this CreateDatasetDetails.
        Simple key-value pair that is applied without any predefined name, type or scope. Exists for cross-compatibility only.
        Example: `{\"bar-key\": \"value\"}`


        :return: The freeform_tags of this CreateDatasetDetails.
        :rtype: dict(str, str)
        """
        return self._freeform_tags

    @freeform_tags.setter
    def freeform_tags(self, freeform_tags):
        """
        Sets the freeform_tags of this CreateDatasetDetails.
        Simple key-value pair that is applied without any predefined name, type or scope. Exists for cross-compatibility only.
        Example: `{\"bar-key\": \"value\"}`


        :param freeform_tags: The freeform_tags of this CreateDatasetDetails.
        :type: dict(str, str)
        """
        self._freeform_tags = freeform_tags

    @property
    def defined_tags(self):
        """
        Gets the defined_tags of this CreateDatasetDetails.
        Defined tags for this resource. Each key is predefined and scoped to a namespace.
        Example: `{\"foo-namespace\": {\"bar-key\": \"value\"}}`


        :return: The defined_tags of this CreateDatasetDetails.
        :rtype: dict(str, dict(str, object))
        """
        return self._defined_tags

    @defined_tags.setter
    def defined_tags(self, defined_tags):
        """
        Sets the defined_tags of this CreateDatasetDetails.
        Defined tags for this resource. Each key is predefined and scoped to a namespace.
        Example: `{\"foo-namespace\": {\"bar-key\": \"value\"}}`


        :param defined_tags: The defined_tags of this CreateDatasetDetails.
        :type: dict(str, dict(str, object))
        """
        self._defined_tags = defined_tags

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
