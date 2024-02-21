# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class Entity(object):
    """
    Description of FeatureStore Entity.
    """

    #: A constant which can be used with the lifecycle_state property of a Entity.
    #: This constant has a value of "ACTIVE"
    LIFECYCLE_STATE_ACTIVE = "ACTIVE"

    #: A constant which can be used with the lifecycle_state property of a Entity.
    #: This constant has a value of "DELETED"
    LIFECYCLE_STATE_DELETED = "DELETED"

    #: A constant which can be used with the lifecycle_state property of a Entity.
    #: This constant has a value of "FAILED"
    LIFECYCLE_STATE_FAILED = "FAILED"

    def __init__(self, **kwargs):
        """
        Initializes a new Entity object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param id:
            The value to assign to the id property of this Entity.
        :type id: str

        :param compartment_id:
            The value to assign to the compartment_id property of this Entity.
        :type compartment_id: str

        :param feature_store_id:
            The value to assign to the feature_store_id property of this Entity.
        :type feature_store_id: str

        :param lifecycle_state:
            The value to assign to the lifecycle_state property of this Entity.
            Allowed values for this property are: "ACTIVE", "DELETED", "FAILED", 'UNKNOWN_ENUM_VALUE'.
            Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.
        :type lifecycle_state: str

        :param name:
            The value to assign to the name property of this Entity.
        :type name: str

        :param time_created:
            The value to assign to the time_created property of this Entity.
        :type time_created: str

        :param time_updated:
            The value to assign to the time_updated property of this Entity.
        :type time_updated: str

        :param description:
            The value to assign to the description property of this Entity.
        :type description: str

        :param created_by:
            The value to assign to the created_by property of this Entity.
        :type created_by: str

        :param updated_by:
            The value to assign to the updated_by property of this Entity.
        :type updated_by: str

        :param freeform_tags:
            The value to assign to the freeform_tags property of this Entity.
        :type freeform_tags: dict(str, str)

        :param defined_tags:
            The value to assign to the defined_tags property of this Entity.
        :type defined_tags: dict(str, dict(str, object))

        :param system_tags:
            The value to assign to the system_tags property of this Entity.
        :type system_tags: dict(str, dict(str, object))

        """
        self.swagger_types = {
            'id': 'str',
            'compartment_id': 'str',
            'feature_store_id': 'str',
            'lifecycle_state': 'str',
            'name': 'str',
            'time_created': 'str',
            'time_updated': 'str',
            'description': 'str',
            'created_by': 'str',
            'updated_by': 'str',
            'freeform_tags': 'dict(str, str)',
            'defined_tags': 'dict(str, dict(str, object))',
            'system_tags': 'dict(str, dict(str, object))'
        }

        self.attribute_map = {
            'id': 'id',
            'compartment_id': 'compartmentId',
            'feature_store_id': 'featureStoreId',
            'lifecycle_state': 'lifecycleState',
            'name': 'name',
            'time_created': 'timeCreated',
            'time_updated': 'timeUpdated',
            'description': 'description',
            'created_by': 'createdBy',
            'updated_by': 'updatedBy',
            'freeform_tags': 'freeformTags',
            'defined_tags': 'definedTags',
            'system_tags': 'systemTags'
        }

        self._id = None
        self._compartment_id = None
        self._feature_store_id = None
        self._lifecycle_state = None
        self._name = None
        self._time_created = None
        self._time_updated = None
        self._description = None
        self._created_by = None
        self._updated_by = None
        self._freeform_tags = None
        self._defined_tags = None
        self._system_tags = None

    @property
    def id(self):
        """
        **[Required]** Gets the id of this Entity.
        The Unique Oracle ID (OCID) that is immutable on creation.


        :return: The id of this Entity.
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """
        Sets the id of this Entity.
        The Unique Oracle ID (OCID) that is immutable on creation.


        :param id: The id of this Entity.
        :type: str
        """
        self._id = id

    @property
    def compartment_id(self):
        """
        **[Required]** Gets the compartment_id of this Entity.
        The OCID of the compartment containing the feature store entity.


        :return: The compartment_id of this Entity.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this Entity.
        The OCID of the compartment containing the feature store entity.


        :param compartment_id: The compartment_id of this Entity.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def feature_store_id(self):
        """
        **[Required]** Gets the feature_store_id of this Entity.
        The OCID of the feature store resource containing the entity.


        :return: The feature_store_id of this Entity.
        :rtype: str
        """
        return self._feature_store_id

    @feature_store_id.setter
    def feature_store_id(self, feature_store_id):
        """
        Sets the feature_store_id of this Entity.
        The OCID of the feature store resource containing the entity.


        :param feature_store_id: The feature_store_id of this Entity.
        :type: str
        """
        self._feature_store_id = feature_store_id

    @property
    def lifecycle_state(self):
        """
        **[Required]** Gets the lifecycle_state of this Entity.
        The current state of the entity.

        Allowed values for this property are: "ACTIVE", "DELETED", "FAILED", 'UNKNOWN_ENUM_VALUE'.
        Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.


        :return: The lifecycle_state of this Entity.
        :rtype: str
        """
        return self._lifecycle_state

    @lifecycle_state.setter
    def lifecycle_state(self, lifecycle_state):
        """
        Sets the lifecycle_state of this Entity.
        The current state of the entity.


        :param lifecycle_state: The lifecycle_state of this Entity.
        :type: str
        """
        allowed_values = ["ACTIVE", "DELETED", "FAILED"]
        if not value_allowed_none_or_none_sentinel(lifecycle_state, allowed_values):
            lifecycle_state = 'UNKNOWN_ENUM_VALUE'
        self._lifecycle_state = lifecycle_state

    @property
    def name(self):
        """
        **[Required]** Gets the name of this Entity.
        A user-friendly unique name.


        :return: The name of this Entity.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Sets the name of this Entity.
        A user-friendly unique name.


        :param name: The name of this Entity.
        :type: str
        """
        self._name = name

    @property
    def time_created(self):
        """
        **[Required]** Gets the time_created of this Entity.

        :return: The time_created of this Entity.
        :rtype: str
        """
        return self._time_created

    @time_created.setter
    def time_created(self, time_created):
        """
        Sets the time_created of this Entity.

        :param time_created: The time_created of this Entity.
        :type: str
        """
        self._time_created = time_created

    @property
    def time_updated(self):
        """
        **[Required]** Gets the time_updated of this Entity.

        :return: The time_updated of this Entity.
        :rtype: str
        """
        return self._time_updated

    @time_updated.setter
    def time_updated(self, time_updated):
        """
        Sets the time_updated of this Entity.

        :param time_updated: The time_updated of this Entity.
        :type: str
        """
        self._time_updated = time_updated

    @property
    def description(self):
        """
        Gets the description of this Entity.
        feature store entity description


        :return: The description of this Entity.
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """
        Sets the description of this Entity.
        feature store entity description


        :param description: The description of this Entity.
        :type: str
        """
        self._description = description

    @property
    def created_by(self):
        """
        Gets the created_by of this Entity.
        feature store entity user creation details


        :return: The created_by of this Entity.
        :rtype: str
        """
        return self._created_by

    @created_by.setter
    def created_by(self, created_by):
        """
        Sets the created_by of this Entity.
        feature store entity user creation details


        :param created_by: The created_by of this Entity.
        :type: str
        """
        self._created_by = created_by

    @property
    def updated_by(self):
        """
        Gets the updated_by of this Entity.
        feature store entity updated by details


        :return: The updated_by of this Entity.
        :rtype: str
        """
        return self._updated_by

    @updated_by.setter
    def updated_by(self, updated_by):
        """
        Sets the updated_by of this Entity.
        feature store entity updated by details


        :param updated_by: The updated_by of this Entity.
        :type: str
        """
        self._updated_by = updated_by

    @property
    def freeform_tags(self):
        """
        Gets the freeform_tags of this Entity.
        Simple key-value pair that is applied without any predefined name, type or scope. Exists for cross-compatibility only.
        Example: `{\"bar-key\": \"value\"}`


        :return: The freeform_tags of this Entity.
        :rtype: dict(str, str)
        """
        return self._freeform_tags

    @freeform_tags.setter
    def freeform_tags(self, freeform_tags):
        """
        Sets the freeform_tags of this Entity.
        Simple key-value pair that is applied without any predefined name, type or scope. Exists for cross-compatibility only.
        Example: `{\"bar-key\": \"value\"}`


        :param freeform_tags: The freeform_tags of this Entity.
        :type: dict(str, str)
        """
        self._freeform_tags = freeform_tags

    @property
    def defined_tags(self):
        """
        Gets the defined_tags of this Entity.
        Defined tags for this resource. Each key is predefined and scoped to a namespace.
        Example: `{\"foo-namespace\": {\"bar-key\": \"value\"}}`


        :return: The defined_tags of this Entity.
        :rtype: dict(str, dict(str, object))
        """
        return self._defined_tags

    @defined_tags.setter
    def defined_tags(self, defined_tags):
        """
        Sets the defined_tags of this Entity.
        Defined tags for this resource. Each key is predefined and scoped to a namespace.
        Example: `{\"foo-namespace\": {\"bar-key\": \"value\"}}`


        :param defined_tags: The defined_tags of this Entity.
        :type: dict(str, dict(str, object))
        """
        self._defined_tags = defined_tags

    @property
    def system_tags(self):
        """
        Gets the system_tags of this Entity.
        System tags for this resource. Each key is predefined and scoped to a namespace.
        Example: `{\"orcl-cloud\": {\"free-tier-retained\": \"true\"}}`


        :return: The system_tags of this Entity.
        :rtype: dict(str, dict(str, object))
        """
        return self._system_tags

    @system_tags.setter
    def system_tags(self, system_tags):
        """
        Sets the system_tags of this Entity.
        System tags for this resource. Each key is predefined and scoped to a namespace.
        Example: `{\"orcl-cloud\": {\"free-tier-retained\": \"true\"}}`


        :param system_tags: The system_tags of this Entity.
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
