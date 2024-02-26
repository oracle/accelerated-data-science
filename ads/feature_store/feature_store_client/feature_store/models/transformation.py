# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class Transformation(object):
    """
    Transformation Description.
    """

    #: A constant which can be used with the lifecycle_state property of a Transformation.
    #: This constant has a value of "ACTIVE"
    LIFECYCLE_STATE_ACTIVE = "ACTIVE"

    #: A constant which can be used with the lifecycle_state property of a Transformation.
    #: This constant has a value of "DELETED"
    LIFECYCLE_STATE_DELETED = "DELETED"

    #: A constant which can be used with the lifecycle_state property of a Transformation.
    #: This constant has a value of "FAILED"
    LIFECYCLE_STATE_FAILED = "FAILED"

    def __init__(self, **kwargs):
        """
        Initializes a new Transformation object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param id:
            The value to assign to the id property of this Transformation.
        :type id: str

        :param compartment_id:
            The value to assign to the compartment_id property of this Transformation.
        :type compartment_id: str

        :param feature_store_id:
            The value to assign to the feature_store_id property of this Transformation.
        :type feature_store_id: str

        :param name:
            The value to assign to the name property of this Transformation.
        :type name: str

        :param description:
            The value to assign to the description property of this Transformation.
        :type description: str

        :param source_code:
            The value to assign to the source_code property of this Transformation.
        :type source_code: str

        :param transformation_mode:
            The value to assign to the transformation_mode property of this Transformation.
        :type transformation_mode: str

        :param time_created:
            The value to assign to the time_created property of this Transformation.
        :type time_created: str

        :param time_updated:
            The value to assign to the time_updated property of this Transformation.
        :type time_updated: str

        :param created_by:
            The value to assign to the created_by property of this Transformation.
        :type created_by: str

        :param updated_by:
            The value to assign to the updated_by property of this Transformation.
        :type updated_by: str

        :param lifecycle_state:
            The value to assign to the lifecycle_state property of this Transformation.
            Allowed values for this property are: "ACTIVE", "DELETED", "FAILED", 'UNKNOWN_ENUM_VALUE'.
            Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.
        :type lifecycle_state: str

        """
        self.swagger_types = {
            'id': 'str',
            'compartment_id': 'str',
            'feature_store_id': 'str',
            'name': 'str',
            'description': 'str',
            'source_code': 'str',
            'transformation_mode': 'str',
            'time_created': 'str',
            'time_updated': 'str',
            'created_by': 'str',
            'updated_by': 'str',
            'lifecycle_state': 'str'
        }

        self.attribute_map = {
            'id': 'id',
            'compartment_id': 'compartmentId',
            'feature_store_id': 'featureStoreId',
            'name': 'name',
            'description': 'description',
            'source_code': 'sourceCode',
            'transformation_mode': 'transformationMode',
            'time_created': 'timeCreated',
            'time_updated': 'timeUpdated',
            'created_by': 'createdBy',
            'updated_by': 'updatedBy',
            'lifecycle_state': 'lifecycleState'
        }

        self._id = None
        self._compartment_id = None
        self._feature_store_id = None
        self._name = None
        self._description = None
        self._source_code = None
        self._transformation_mode = None
        self._time_created = None
        self._time_updated = None
        self._created_by = None
        self._updated_by = None
        self._lifecycle_state = None

    @property
    def id(self):
        """
        **[Required]** Gets the id of this Transformation.
        The Unique Oracle ID (OCID) that is immutable on creation.


        :return: The id of this Transformation.
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """
        Sets the id of this Transformation.
        The Unique Oracle ID (OCID) that is immutable on creation.


        :param id: The id of this Transformation.
        :type: str
        """
        self._id = id

    @property
    def compartment_id(self):
        """
        **[Required]** Gets the compartment_id of this Transformation.
        The OCID of the compartment containing the DataAsset.


        :return: The compartment_id of this Transformation.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this Transformation.
        The OCID of the compartment containing the DataAsset.


        :param compartment_id: The compartment_id of this Transformation.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def feature_store_id(self):
        """
        **[Required]** Gets the feature_store_id of this Transformation.
        The OCID of feature store


        :return: The feature_store_id of this Transformation.
        :rtype: str
        """
        return self._feature_store_id

    @feature_store_id.setter
    def feature_store_id(self, feature_store_id):
        """
        Sets the feature_store_id of this Transformation.
        The OCID of feature store


        :param feature_store_id: The feature_store_id of this Transformation.
        :type: str
        """
        self._feature_store_id = feature_store_id

    @property
    def name(self):
        """
        **[Required]** Gets the name of this Transformation.
        A user-friendly name. Does not have to be unique, and it's changeable. Avoid entering confidential information.


        :return: The name of this Transformation.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Sets the name of this Transformation.
        A user-friendly name. Does not have to be unique, and it's changeable. Avoid entering confidential information.


        :param name: The name of this Transformation.
        :type: str
        """
        self._name = name

    @property
    def description(self):
        """
        Gets the description of this Transformation.
        A short description of the data asset.


        :return: The description of this Transformation.
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """
        Sets the description of this Transformation.
        A short description of the data asset.


        :param description: The description of this Transformation.
        :type: str
        """
        self._description = description

    @property
    def source_code(self):
        """
        **[Required]** Gets the source_code of this Transformation.
        Source code for the transformation


        :return: The source_code of this Transformation.
        :rtype: str
        """
        return self._source_code

    @source_code.setter
    def source_code(self, source_code):
        """
        Sets the source_code of this Transformation.
        Source code for the transformation


        :param source_code: The source_code of this Transformation.
        :type: str
        """
        self._source_code = source_code

    @property
    def transformation_mode(self):
        """
        **[Required]** Gets the transformation_mode of this Transformation.
        Mode of the transformation


        :return: The transformation_mode of this Transformation.
        :rtype: str
        """
        return self._transformation_mode

    @transformation_mode.setter
    def transformation_mode(self, transformation_mode):
        """
        Sets the transformation_mode of this Transformation.
        Mode of the transformation


        :param transformation_mode: The transformation_mode of this Transformation.
        :type: str
        """
        self._transformation_mode = transformation_mode

    @property
    def time_created(self):
        """
        **[Required]** Gets the time_created of this Transformation.

        :return: The time_created of this Transformation.
        :rtype: str
        """
        return self._time_created

    @time_created.setter
    def time_created(self, time_created):
        """
        Sets the time_created of this Transformation.

        :param time_created: The time_created of this Transformation.
        :type: str
        """
        self._time_created = time_created

    @property
    def time_updated(self):
        """
        Gets the time_updated of this Transformation.

        :return: The time_updated of this Transformation.
        :rtype: str
        """
        return self._time_updated

    @time_updated.setter
    def time_updated(self, time_updated):
        """
        Sets the time_updated of this Transformation.

        :param time_updated: The time_updated of this Transformation.
        :type: str
        """
        self._time_updated = time_updated

    @property
    def created_by(self):
        """
        Gets the created_by of this Transformation.
        User creation details


        :return: The created_by of this Transformation.
        :rtype: str
        """
        return self._created_by

    @created_by.setter
    def created_by(self, created_by):
        """
        Sets the created_by of this Transformation.
        User creation details


        :param created_by: The created_by of this Transformation.
        :type: str
        """
        self._created_by = created_by

    @property
    def updated_by(self):
        """
        Gets the updated_by of this Transformation.
        feature store transformation updated by details


        :return: The updated_by of this Transformation.
        :rtype: str
        """
        return self._updated_by

    @updated_by.setter
    def updated_by(self, updated_by):
        """
        Sets the updated_by of this Transformation.
        feature store transformation updated by details


        :param updated_by: The updated_by of this Transformation.
        :type: str
        """
        self._updated_by = updated_by

    @property
    def lifecycle_state(self):
        """
        **[Required]** Gets the lifecycle_state of this Transformation.
        The current state of the transformation

        Allowed values for this property are: "ACTIVE", "DELETED", "FAILED", 'UNKNOWN_ENUM_VALUE'.
        Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.


        :return: The lifecycle_state of this Transformation.
        :rtype: str
        """
        return self._lifecycle_state

    @lifecycle_state.setter
    def lifecycle_state(self, lifecycle_state):
        """
        Sets the lifecycle_state of this Transformation.
        The current state of the transformation


        :param lifecycle_state: The lifecycle_state of this Transformation.
        :type: str
        """
        allowed_values = ["ACTIVE", "DELETED", "FAILED"]
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
