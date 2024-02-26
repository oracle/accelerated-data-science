# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class SearchSummary(object):
    """
    Search result item.
    """

    #: A constant which can be used with the resource_type property of a SearchSummary.
    #: This constant has a value of "FEATURE_STORE"
    RESOURCE_TYPE_FEATURE_STORE = "FEATURE_STORE"

    #: A constant which can be used with the resource_type property of a SearchSummary.
    #: This constant has a value of "FEATURE_STORE_DATA_SOURCE"
    RESOURCE_TYPE_FEATURE_STORE_DATA_SOURCE = "FEATURE_STORE_DATA_SOURCE"

    #: A constant which can be used with the resource_type property of a SearchSummary.
    #: This constant has a value of "FEATURE_STORE_ENTITY"
    RESOURCE_TYPE_FEATURE_STORE_ENTITY = "FEATURE_STORE_ENTITY"

    #: A constant which can be used with the resource_type property of a SearchSummary.
    #: This constant has a value of "FEATURE_STORE_FEATURE_GROUP"
    RESOURCE_TYPE_FEATURE_STORE_FEATURE_GROUP = "FEATURE_STORE_FEATURE_GROUP"

    #: A constant which can be used with the resource_type property of a SearchSummary.
    #: This constant has a value of "FEATURE_STORE_TRANSFORMATION"
    RESOURCE_TYPE_FEATURE_STORE_TRANSFORMATION = "FEATURE_STORE_TRANSFORMATION"

    #: A constant which can be used with the resource_type property of a SearchSummary.
    #: This constant has a value of "FEATURE_STORE_DATASET"
    RESOURCE_TYPE_FEATURE_STORE_DATASET = "FEATURE_STORE_DATASET"

    #: A constant which can be used with the resource_type property of a SearchSummary.
    #: This constant has a value of "FEATURE_STORE_FEATURE"
    RESOURCE_TYPE_FEATURE_STORE_FEATURE = "FEATURE_STORE_FEATURE"

    #: A constant which can be used with the resource_type property of a SearchSummary.
    #: This constant has a value of "FEATURE_STORE_DATA_ASSET"
    RESOURCE_TYPE_FEATURE_STORE_DATA_ASSET = "FEATURE_STORE_DATA_ASSET"

    def __init__(self, **kwargs):
        """
        Initializes a new SearchSummary object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param identifier:
            The value to assign to the identifier property of this SearchSummary.
        :type identifier: str

        :param compartment_id:
            The value to assign to the compartment_id property of this SearchSummary.
        :type compartment_id: str

        :param display_name:
            The value to assign to the display_name property of this SearchSummary.
        :type display_name: str

        :param time_created:
            The value to assign to the time_created property of this SearchSummary.
        :type time_created: str

        :param lifecycle_state:
            The value to assign to the lifecycle_state property of this SearchSummary.
        :type lifecycle_state: str

        :param resource_type:
            The value to assign to the resource_type property of this SearchSummary.
            Allowed values for this property are: "FEATURE_STORE", "FEATURE_STORE_DATA_SOURCE", "FEATURE_STORE_ENTITY", "FEATURE_STORE_FEATURE_GROUP", "FEATURE_STORE_TRANSFORMATION", "FEATURE_STORE_DATASET", "FEATURE_STORE_FEATURE", "FEATURE_STORE_DATA_ASSET", 'UNKNOWN_ENUM_VALUE'.
            Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.
        :type resource_type: str

        :param freeform_tags:
            The value to assign to the freeform_tags property of this SearchSummary.
        :type freeform_tags: dict(str, str)

        :param defined_tags:
            The value to assign to the defined_tags property of this SearchSummary.
        :type defined_tags: dict(str, dict(str, object))

        :param system_tags:
            The value to assign to the system_tags property of this SearchSummary.
        :type system_tags: dict(str, dict(str, object))

        :param search_context:
            The value to assign to the search_context property of this SearchSummary.
        :type search_context: oci.feature_store.models.SearchContext

        :param name:
            The value to assign to the name property of this SearchSummary.
        :type name: str

        :param description:
            The value to assign to the description property of this SearchSummary.
        :type description: str

        """
        self.swagger_types = {
            'identifier': 'str',
            'compartment_id': 'str',
            'display_name': 'str',
            'time_created': 'str',
            'lifecycle_state': 'str',
            'resource_type': 'str',
            'freeform_tags': 'dict(str, str)',
            'defined_tags': 'dict(str, dict(str, object))',
            'system_tags': 'dict(str, dict(str, object))',
            'search_context': 'SearchContext',
            'name': 'str',
            'description': 'str'
        }

        self.attribute_map = {
            'identifier': 'identifier',
            'compartment_id': 'compartmentId',
            'display_name': 'displayName',
            'time_created': 'timeCreated',
            'lifecycle_state': 'lifecycleState',
            'resource_type': 'resourceType',
            'freeform_tags': 'freeformTags',
            'defined_tags': 'definedTags',
            'system_tags': 'systemTags',
            'search_context': 'searchContext',
            'name': 'name',
            'description': 'description'
        }

        self._identifier = None
        self._compartment_id = None
        self._display_name = None
        self._time_created = None
        self._lifecycle_state = None
        self._resource_type = None
        self._freeform_tags = None
        self._defined_tags = None
        self._system_tags = None
        self._search_context = None
        self._name = None
        self._description = None

    @property
    def identifier(self):
        """
        **[Required]** Gets the identifier of this SearchSummary.
        The `OCID`__ of the entity

        __ https://docs.cloud.oracle.com/iaas/Content/General/Concepts/identifiers.htm


        :return: The identifier of this SearchSummary.
        :rtype: str
        """
        return self._identifier

    @identifier.setter
    def identifier(self, identifier):
        """
        Sets the identifier of this SearchSummary.
        The `OCID`__ of the entity

        __ https://docs.cloud.oracle.com/iaas/Content/General/Concepts/identifiers.htm


        :param identifier: The identifier of this SearchSummary.
        :type: str
        """
        self._identifier = identifier

    @property
    def compartment_id(self):
        """
        **[Required]** Gets the compartment_id of this SearchSummary.
        The OCID of the compartment containing the FeatureStore.


        :return: The compartment_id of this SearchSummary.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this SearchSummary.
        The OCID of the compartment containing the FeatureStore.


        :param compartment_id: The compartment_id of this SearchSummary.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def display_name(self):
        """
        **[Required]** Gets the display_name of this SearchSummary.
        A user-friendly display name for the resource.


        :return: The display_name of this SearchSummary.
        :rtype: str
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name):
        """
        Sets the display_name of this SearchSummary.
        A user-friendly display name for the resource.


        :param display_name: The display_name of this SearchSummary.
        :type: str
        """
        self._display_name = display_name

    @property
    def time_created(self):
        """
        Gets the time_created of this SearchSummary.
        The time that this resource was created.


        :return: The time_created of this SearchSummary.
        :rtype: str
        """
        return self._time_created

    @time_created.setter
    def time_created(self, time_created):
        """
        Sets the time_created of this SearchSummary.
        The time that this resource was created.


        :param time_created: The time_created of this SearchSummary.
        :type: str
        """
        self._time_created = time_created

    @property
    def lifecycle_state(self):
        """
        Gets the lifecycle_state of this SearchSummary.
        The current state of the resource.


        :return: The lifecycle_state of this SearchSummary.
        :rtype: str
        """
        return self._lifecycle_state

    @lifecycle_state.setter
    def lifecycle_state(self, lifecycle_state):
        """
        Sets the lifecycle_state of this SearchSummary.
        The current state of the resource.


        :param lifecycle_state: The lifecycle_state of this SearchSummary.
        :type: str
        """
        self._lifecycle_state = lifecycle_state

    @property
    def resource_type(self):
        """
        **[Required]** Gets the resource_type of this SearchSummary.
        The type of the entity

        Allowed values for this property are: "FEATURE_STORE", "FEATURE_STORE_DATA_SOURCE", "FEATURE_STORE_ENTITY", "FEATURE_STORE_FEATURE_GROUP", "FEATURE_STORE_TRANSFORMATION", "FEATURE_STORE_DATASET", "FEATURE_STORE_FEATURE", "FEATURE_STORE_DATA_ASSET", 'UNKNOWN_ENUM_VALUE'.
        Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.


        :return: The resource_type of this SearchSummary.
        :rtype: str
        """
        return self._resource_type

    @resource_type.setter
    def resource_type(self, resource_type):
        """
        Sets the resource_type of this SearchSummary.
        The type of the entity


        :param resource_type: The resource_type of this SearchSummary.
        :type: str
        """
        allowed_values = ["FEATURE_STORE", "FEATURE_STORE_DATA_SOURCE", "FEATURE_STORE_ENTITY", "FEATURE_STORE_FEATURE_GROUP", "FEATURE_STORE_TRANSFORMATION", "FEATURE_STORE_DATASET", "FEATURE_STORE_FEATURE", "FEATURE_STORE_DATA_ASSET"]
        if not value_allowed_none_or_none_sentinel(resource_type, allowed_values):
            resource_type = 'UNKNOWN_ENUM_VALUE'
        self._resource_type = resource_type

    @property
    def freeform_tags(self):
        """
        Gets the freeform_tags of this SearchSummary.
        Free-form tags for this resource. Each tag is a simple key-value pair with no predefined name, type, or namespace.
        For more information, see `Resource Tags`__.
        Example: `{\"Department\": \"Finance\"}`

        __ https://docs.cloud.oracle.com/Content/General/Concepts/resourcetags.htm


        :return: The freeform_tags of this SearchSummary.
        :rtype: dict(str, str)
        """
        return self._freeform_tags

    @freeform_tags.setter
    def freeform_tags(self, freeform_tags):
        """
        Sets the freeform_tags of this SearchSummary.
        Free-form tags for this resource. Each tag is a simple key-value pair with no predefined name, type, or namespace.
        For more information, see `Resource Tags`__.
        Example: `{\"Department\": \"Finance\"}`

        __ https://docs.cloud.oracle.com/Content/General/Concepts/resourcetags.htm


        :param freeform_tags: The freeform_tags of this SearchSummary.
        :type: dict(str, str)
        """
        self._freeform_tags = freeform_tags

    @property
    def defined_tags(self):
        """
        Gets the defined_tags of this SearchSummary.
        Defined tags for this resource. Each key is predefined and scoped to a namespace.
        For more information, see `Resource Tags`__.
        Example: `{\"Operations\": {\"CostCenter\": \"42\"}}`

        __ https://docs.cloud.oracle.com/Content/General/Concepts/resourcetags.htm


        :return: The defined_tags of this SearchSummary.
        :rtype: dict(str, dict(str, object))
        """
        return self._defined_tags

    @defined_tags.setter
    def defined_tags(self, defined_tags):
        """
        Sets the defined_tags of this SearchSummary.
        Defined tags for this resource. Each key is predefined and scoped to a namespace.
        For more information, see `Resource Tags`__.
        Example: `{\"Operations\": {\"CostCenter\": \"42\"}}`

        __ https://docs.cloud.oracle.com/Content/General/Concepts/resourcetags.htm


        :param defined_tags: The defined_tags of this SearchSummary.
        :type: dict(str, dict(str, object))
        """
        self._defined_tags = defined_tags

    @property
    def system_tags(self):
        """
        Gets the system_tags of this SearchSummary.
        System tags associated with this resource, if any. System tags are set by Oracle Cloud Infrastructure services. Each key is predefined and scoped to namespaces.
        For more information, see `Resource Tags`__.
        Example: `{orcl-cloud: {free-tier-retain: true}}`

        __ https://docs.oracle.com/iaas/Content/General/Concepts/resourcetags.htm


        :return: The system_tags of this SearchSummary.
        :rtype: dict(str, dict(str, object))
        """
        return self._system_tags

    @system_tags.setter
    def system_tags(self, system_tags):
        """
        Sets the system_tags of this SearchSummary.
        System tags associated with this resource, if any. System tags are set by Oracle Cloud Infrastructure services. Each key is predefined and scoped to namespaces.
        For more information, see `Resource Tags`__.
        Example: `{orcl-cloud: {free-tier-retain: true}}`

        __ https://docs.oracle.com/iaas/Content/General/Concepts/resourcetags.htm


        :param system_tags: The system_tags of this SearchSummary.
        :type: dict(str, dict(str, object))
        """
        self._system_tags = system_tags

    @property
    def search_context(self):
        """
        Gets the search_context of this SearchSummary.

        :return: The search_context of this SearchSummary.
        :rtype: oci.feature_store.models.SearchContext
        """
        return self._search_context

    @search_context.setter
    def search_context(self, search_context):
        """
        Sets the search_context of this SearchSummary.

        :param search_context: The search_context of this SearchSummary.
        :type: oci.feature_store.models.SearchContext
        """
        self._search_context = search_context

    @property
    def name(self):
        """
        Gets the name of this SearchSummary.
        The name of the item that was found.


        :return: The name of this SearchSummary.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Sets the name of this SearchSummary.
        The name of the item that was found.


        :param name: The name of this SearchSummary.
        :type: str
        """
        self._name = name

    @property
    def description(self):
        """
        Gets the description of this SearchSummary.
        The description of the item that was found.


        :return: The description of this SearchSummary.
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """
        Sets the description of this SearchSummary.
        The description of the item that was found.


        :param description: The description of this SearchSummary.
        :type: str
        """
        self._description = description

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
