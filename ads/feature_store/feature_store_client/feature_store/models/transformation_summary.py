# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class TransformationSummary(object):
    """
    Summary of the FeatureStore transformation.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new TransformationSummary object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param id:
            The value to assign to the id property of this TransformationSummary.
        :type id: str

        :param compartment_id:
            The value to assign to the compartment_id property of this TransformationSummary.
        :type compartment_id: str

        :param name:
            The value to assign to the name property of this TransformationSummary.
        :type name: str

        :param time_created:
            The value to assign to the time_created property of this TransformationSummary.
        :type time_created: str

        :param time_updated:
            The value to assign to the time_updated property of this TransformationSummary.
        :type time_updated: str

        :param lifecycle_state:
            The value to assign to the lifecycle_state property of this TransformationSummary.
        :type lifecycle_state: str

        :param feature_store_id:
            The value to assign to the feature_store_id property of this TransformationSummary.
        :type feature_store_id: str

        :param created_by:
            The value to assign to the created_by property of this TransformationSummary.
        :type created_by: str

        :param updated_by:
            The value to assign to the updated_by property of this TransformationSummary.
        :type updated_by: str

        :param freeform_tags:
            The value to assign to the freeform_tags property of this TransformationSummary.
        :type freeform_tags: dict(str, str)

        :param defined_tags:
            The value to assign to the defined_tags property of this TransformationSummary.
        :type defined_tags: dict(str, dict(str, object))

        """
        self.swagger_types = {
            'id': 'str',
            'compartment_id': 'str',
            'name': 'str',
            'time_created': 'str',
            'time_updated': 'str',
            'lifecycle_state': 'str',
            'feature_store_id': 'str',
            'created_by': 'str',
            'updated_by': 'str',
            'freeform_tags': 'dict(str, str)',
            'defined_tags': 'dict(str, dict(str, object))'
        }

        self.attribute_map = {
            'id': 'id',
            'compartment_id': 'compartmentId',
            'name': 'name',
            'time_created': 'timeCreated',
            'time_updated': 'timeUpdated',
            'lifecycle_state': 'lifecycleState',
            'feature_store_id': 'featureStoreId',
            'created_by': 'createdBy',
            'updated_by': 'updatedBy',
            'freeform_tags': 'freeformTags',
            'defined_tags': 'definedTags'
        }

        self._id = None
        self._compartment_id = None
        self._name = None
        self._time_created = None
        self._time_updated = None
        self._lifecycle_state = None
        self._feature_store_id = None
        self._created_by = None
        self._updated_by = None
        self._freeform_tags = None
        self._defined_tags = None

    @property
    def id(self):
        """
        **[Required]** Gets the id of this TransformationSummary.
        Unique identifier that is immutable on creation


        :return: The id of this TransformationSummary.
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """
        Sets the id of this TransformationSummary.
        Unique identifier that is immutable on creation


        :param id: The id of this TransformationSummary.
        :type: str
        """
        self._id = id

    @property
    def compartment_id(self):
        """
        **[Required]** Gets the compartment_id of this TransformationSummary.
        Compartment Identifier


        :return: The compartment_id of this TransformationSummary.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this TransformationSummary.
        Compartment Identifier


        :param compartment_id: The compartment_id of this TransformationSummary.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def name(self):
        """
        **[Required]** Gets the name of this TransformationSummary.
        FeatureStore transformation name, can be renamed


        :return: The name of this TransformationSummary.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Sets the name of this TransformationSummary.
        FeatureStore transformation name, can be renamed


        :param name: The name of this TransformationSummary.
        :type: str
        """
        self._name = name

    @property
    def time_created(self):
        """
        **[Required]** Gets the time_created of this TransformationSummary.

        :return: The time_created of this TransformationSummary.
        :rtype: str
        """
        return self._time_created

    @time_created.setter
    def time_created(self, time_created):
        """
        Sets the time_created of this TransformationSummary.

        :param time_created: The time_created of this TransformationSummary.
        :type: str
        """
        self._time_created = time_created

    @property
    def time_updated(self):
        """
        **[Required]** Gets the time_updated of this TransformationSummary.

        :return: The time_updated of this TransformationSummary.
        :rtype: str
        """
        return self._time_updated

    @time_updated.setter
    def time_updated(self, time_updated):
        """
        Sets the time_updated of this TransformationSummary.

        :param time_updated: The time_updated of this TransformationSummary.
        :type: str
        """
        self._time_updated = time_updated

    @property
    def lifecycle_state(self):
        """
        Gets the lifecycle_state of this TransformationSummary.
        The current state of the FeatureStore transformation.


        :return: The lifecycle_state of this TransformationSummary.
        :rtype: str
        """
        return self._lifecycle_state

    @lifecycle_state.setter
    def lifecycle_state(self, lifecycle_state):
        """
        Sets the lifecycle_state of this TransformationSummary.
        The current state of the FeatureStore transformation.


        :param lifecycle_state: The lifecycle_state of this TransformationSummary.
        :type: str
        """
        self._lifecycle_state = lifecycle_state

    @property
    def feature_store_id(self):
        """
        Gets the feature_store_id of this TransformationSummary.
        The OCID of feature store


        :return: The feature_store_id of this TransformationSummary.
        :rtype: str
        """
        return self._feature_store_id

    @feature_store_id.setter
    def feature_store_id(self, feature_store_id):
        """
        Sets the feature_store_id of this TransformationSummary.
        The OCID of feature store


        :param feature_store_id: The feature_store_id of this TransformationSummary.
        :type: str
        """
        self._feature_store_id = feature_store_id

    @property
    def created_by(self):
        """
        Gets the created_by of this TransformationSummary.
        User creation details


        :return: The created_by of this TransformationSummary.
        :rtype: str
        """
        return self._created_by

    @created_by.setter
    def created_by(self, created_by):
        """
        Sets the created_by of this TransformationSummary.
        User creation details


        :param created_by: The created_by of this TransformationSummary.
        :type: str
        """
        self._created_by = created_by

    @property
    def updated_by(self):
        """
        Gets the updated_by of this TransformationSummary.
        feature store transformation updated by details


        :return: The updated_by of this TransformationSummary.
        :rtype: str
        """
        return self._updated_by

    @updated_by.setter
    def updated_by(self, updated_by):
        """
        Sets the updated_by of this TransformationSummary.
        feature store transformation updated by details


        :param updated_by: The updated_by of this TransformationSummary.
        :type: str
        """
        self._updated_by = updated_by

    @property
    def freeform_tags(self):
        """
        Gets the freeform_tags of this TransformationSummary.
        Simple key-value pair that is applied without any predefined name, type or scope. Exists for cross-compatibility only.
        Example: `{\"bar-key\": \"value\"}`


        :return: The freeform_tags of this TransformationSummary.
        :rtype: dict(str, str)
        """
        return self._freeform_tags

    @freeform_tags.setter
    def freeform_tags(self, freeform_tags):
        """
        Sets the freeform_tags of this TransformationSummary.
        Simple key-value pair that is applied without any predefined name, type or scope. Exists for cross-compatibility only.
        Example: `{\"bar-key\": \"value\"}`


        :param freeform_tags: The freeform_tags of this TransformationSummary.
        :type: dict(str, str)
        """
        self._freeform_tags = freeform_tags

    @property
    def defined_tags(self):
        """
        Gets the defined_tags of this TransformationSummary.
        Defined tags for this resource. Each key is predefined and scoped to a namespace.
        Example: `{\"foo-namespace\": {\"bar-key\": \"value\"}}`


        :return: The defined_tags of this TransformationSummary.
        :rtype: dict(str, dict(str, object))
        """
        return self._defined_tags

    @defined_tags.setter
    def defined_tags(self, defined_tags):
        """
        Sets the defined_tags of this TransformationSummary.
        Defined tags for this resource. Each key is predefined and scoped to a namespace.
        Example: `{\"foo-namespace\": {\"bar-key\": \"value\"}}`


        :param defined_tags: The defined_tags of this TransformationSummary.
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
