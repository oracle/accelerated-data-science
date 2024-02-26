# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class DatasetJobSummary(object):
    """
    Job of feature store construct
    """

    def __init__(self, **kwargs):
        """
        Initializes a new DatasetJobSummary object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param id:
            The value to assign to the id property of this DatasetJobSummary.
        :type id: str

        :param compartment_id:
            The value to assign to the compartment_id property of this DatasetJobSummary.
        :type compartment_id: str

        :param display_name:
            The value to assign to the display_name property of this DatasetJobSummary.
        :type display_name: str

        :param time_created:
            The value to assign to the time_created property of this DatasetJobSummary.
        :type time_created: str

        :param time_started:
            The value to assign to the time_started property of this DatasetJobSummary.
        :type time_started: str

        :param time_finished:
            The value to assign to the time_finished property of this DatasetJobSummary.
        :type time_finished: str

        :param created_by:
            The value to assign to the created_by property of this DatasetJobSummary.
        :type created_by: str

        :param lifecycle_state:
            The value to assign to the lifecycle_state property of this DatasetJobSummary.
        :type lifecycle_state: str

        :param validation_details:
            The value to assign to the validation_details property of this DatasetJobSummary.
        :type validation_details: oci.feature_store.models.ValidationOutputDetails

        """
        self.swagger_types = {
            'id': 'str',
            'compartment_id': 'str',
            'display_name': 'str',
            'time_created': 'str',
            'time_started': 'str',
            'time_finished': 'str',
            'created_by': 'str',
            'lifecycle_state': 'str',
            'validation_details': 'ValidationOutputDetails'
        }

        self.attribute_map = {
            'id': 'id',
            'compartment_id': 'compartmentId',
            'display_name': 'displayName',
            'time_created': 'timeCreated',
            'time_started': 'timeStarted',
            'time_finished': 'timeFinished',
            'created_by': 'createdBy',
            'lifecycle_state': 'lifecycleState',
            'validation_details': 'validationDetails'
        }

        self._id = None
        self._compartment_id = None
        self._display_name = None
        self._time_created = None
        self._time_started = None
        self._time_finished = None
        self._created_by = None
        self._lifecycle_state = None
        self._validation_details = None

    @property
    def id(self):
        """
        **[Required]** Gets the id of this DatasetJobSummary.
        The GUID of the construct.


        :return: The id of this DatasetJobSummary.
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """
        Sets the id of this DatasetJobSummary.
        The GUID of the construct.


        :param id: The id of this DatasetJobSummary.
        :type: str
        """
        self._id = id

    @property
    def compartment_id(self):
        """
        **[Required]** Gets the compartment_id of this DatasetJobSummary.
        Compartment Identifier


        :return: The compartment_id of this DatasetJobSummary.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this DatasetJobSummary.
        Compartment Identifier


        :param compartment_id: The compartment_id of this DatasetJobSummary.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def display_name(self):
        """
        Gets the display_name of this DatasetJobSummary.
        A user-friendly display name for the resource.


        :return: The display_name of this DatasetJobSummary.
        :rtype: str
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name):
        """
        Sets the display_name of this DatasetJobSummary.
        A user-friendly display name for the resource.


        :param display_name: The display_name of this DatasetJobSummary.
        :type: str
        """
        self._display_name = display_name

    @property
    def time_created(self):
        """
        Gets the time_created of this DatasetJobSummary.

        :return: The time_created of this DatasetJobSummary.
        :rtype: str
        """
        return self._time_created

    @time_created.setter
    def time_created(self, time_created):
        """
        Sets the time_created of this DatasetJobSummary.

        :param time_created: The time_created of this DatasetJobSummary.
        :type: str
        """
        self._time_created = time_created

    @property
    def time_started(self):
        """
        **[Required]** Gets the time_started of this DatasetJobSummary.

        :return: The time_started of this DatasetJobSummary.
        :rtype: str
        """
        return self._time_started

    @time_started.setter
    def time_started(self, time_started):
        """
        Sets the time_started of this DatasetJobSummary.

        :param time_started: The time_started of this DatasetJobSummary.
        :type: str
        """
        self._time_started = time_started

    @property
    def time_finished(self):
        """
        Gets the time_finished of this DatasetJobSummary.

        :return: The time_finished of this DatasetJobSummary.
        :rtype: str
        """
        return self._time_finished

    @time_finished.setter
    def time_finished(self, time_finished):
        """
        Sets the time_finished of this DatasetJobSummary.

        :param time_finished: The time_finished of this DatasetJobSummary.
        :type: str
        """
        self._time_finished = time_finished

    @property
    def created_by(self):
        """
        **[Required]** Gets the created_by of this DatasetJobSummary.
        The `OCID`__ of the user who created the job.

        __ https://docs.cloud.oracle.com/iaas/Content/General/Concepts/identifiers.htm


        :return: The created_by of this DatasetJobSummary.
        :rtype: str
        """
        return self._created_by

    @created_by.setter
    def created_by(self, created_by):
        """
        Sets the created_by of this DatasetJobSummary.
        The `OCID`__ of the user who created the job.

        __ https://docs.cloud.oracle.com/iaas/Content/General/Concepts/identifiers.htm


        :param created_by: The created_by of this DatasetJobSummary.
        :type: str
        """
        self._created_by = created_by

    @property
    def lifecycle_state(self):
        """
        **[Required]** Gets the lifecycle_state of this DatasetJobSummary.
        The current state of the job.


        :return: The lifecycle_state of this DatasetJobSummary.
        :rtype: str
        """
        return self._lifecycle_state

    @lifecycle_state.setter
    def lifecycle_state(self, lifecycle_state):
        """
        Sets the lifecycle_state of this DatasetJobSummary.
        The current state of the job.


        :param lifecycle_state: The lifecycle_state of this DatasetJobSummary.
        :type: str
        """
        self._lifecycle_state = lifecycle_state

    @property
    def validation_details(self):
        """
        Gets the validation_details of this DatasetJobSummary.

        :return: The validation_details of this DatasetJobSummary.
        :rtype: oci.feature_store.models.ValidationOutputDetails
        """
        return self._validation_details

    @validation_details.setter
    def validation_details(self, validation_details):
        """
        Sets the validation_details of this DatasetJobSummary.

        :param validation_details: The validation_details of this DatasetJobSummary.
        :type: oci.feature_store.models.ValidationOutputDetails
        """
        self._validation_details = validation_details

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
