# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class FeatureGroupJob(object):
    """
    feature group job
    """

    #: A constant which can be used with the lifecycle_state property of a FeatureGroupJob.
    #: This constant has a value of "IN_PROGRESS"
    LIFECYCLE_STATE_IN_PROGRESS = "IN_PROGRESS"

    #: A constant which can be used with the lifecycle_state property of a FeatureGroupJob.
    #: This constant has a value of "FAILED"
    LIFECYCLE_STATE_FAILED = "FAILED"

    #: A constant which can be used with the lifecycle_state property of a FeatureGroupJob.
    #: This constant has a value of "SUCCEEDED"
    LIFECYCLE_STATE_SUCCEEDED = "SUCCEEDED"

    #: A constant which can be used with the ingestion_mode property of a FeatureGroupJob.
    #: This constant has a value of "APPEND"
    INGESTION_MODE_APPEND = "APPEND"

    #: A constant which can be used with the ingestion_mode property of a FeatureGroupJob.
    #: This constant has a value of "OVERWRITE"
    INGESTION_MODE_OVERWRITE = "OVERWRITE"

    #: A constant which can be used with the ingestion_mode property of a FeatureGroupJob.
    #: This constant has a value of "UPSERT"
    INGESTION_MODE_UPSERT = "UPSERT"

    #: A constant which can be used with the ingestion_mode property of a FeatureGroupJob.
    #: This constant has a value of "COMPLETE"
    INGESTION_MODE_COMPLETE = "COMPLETE"

    #: A constant which can be used with the ingestion_mode property of a FeatureGroupJob.
    #: This constant has a value of "UPDATE"
    INGESTION_MODE_UPDATE = "UPDATE"

    #: A constant which can be used with the ingestion_mode property of a FeatureGroupJob.
    #: This constant has a value of "DEFAULT"
    INGESTION_MODE_DEFAULT = "DEFAULT"

    def __init__(self, **kwargs):
        """
        Initializes a new FeatureGroupJob object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param id:
            The value to assign to the id property of this FeatureGroupJob.
        :type id: str

        :param compartment_id:
            The value to assign to the compartment_id property of this FeatureGroupJob.
        :type compartment_id: str

        :param display_name:
            The value to assign to the display_name property of this FeatureGroupJob.
        :type display_name: str

        :param time_created:
            The value to assign to the time_created property of this FeatureGroupJob.
        :type time_created: str

        :param time_started:
            The value to assign to the time_started property of this FeatureGroupJob.
        :type time_started: str

        :param time_finished:
            The value to assign to the time_finished property of this FeatureGroupJob.
        :type time_finished: str

        :param created_by:
            The value to assign to the created_by property of this FeatureGroupJob.
        :type created_by: str

        :param feature_group_id:
            The value to assign to the feature_group_id property of this FeatureGroupJob.
        :type feature_group_id: str

        :param lifecycle_state:
            The value to assign to the lifecycle_state property of this FeatureGroupJob.
            Allowed values for this property are: "IN_PROGRESS", "FAILED", "SUCCEEDED", 'UNKNOWN_ENUM_VALUE'.
            Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.
        :type lifecycle_state: str

        :param time_from:
            The value to assign to the time_from property of this FeatureGroupJob.
        :type time_from: datetime

        :param time_to:
            The value to assign to the time_to property of this FeatureGroupJob.
        :type time_to: datetime

        :param ingestion_mode:
            The value to assign to the ingestion_mode property of this FeatureGroupJob.
            Allowed values for this property are: "APPEND", "OVERWRITE", "UPSERT", "COMPLETE", "UPDATE", "DEFAULT", 'UNKNOWN_ENUM_VALUE'.
            Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.
        :type ingestion_mode: str

        :param feature_option_details:
            The value to assign to the feature_option_details property of this FeatureGroupJob.
        :type feature_option_details: oci.feature_store.models.FeatureOptionDetails

        :param job_output_details:
            The value to assign to the job_output_details property of this FeatureGroupJob.
        :type job_output_details: oci.feature_store.models.FeatureGroupJobOutputDetails

        """
        self.swagger_types = {
            'id': 'str',
            'compartment_id': 'str',
            'display_name': 'str',
            'time_created': 'str',
            'time_started': 'str',
            'time_finished': 'str',
            'created_by': 'str',
            'feature_group_id': 'str',
            'lifecycle_state': 'str',
            'time_from': 'datetime',
            'time_to': 'datetime',
            'ingestion_mode': 'str',
            'feature_option_details': 'FeatureOptionDetails',
            'job_output_details': 'FeatureGroupJobOutputDetails'
        }

        self.attribute_map = {
            'id': 'id',
            'compartment_id': 'compartmentId',
            'display_name': 'displayName',
            'time_created': 'timeCreated',
            'time_started': 'timeStarted',
            'time_finished': 'timeFinished',
            'created_by': 'createdBy',
            'feature_group_id': 'featureGroupId',
            'lifecycle_state': 'lifecycleState',
            'time_from': 'timeFrom',
            'time_to': 'timeTo',
            'ingestion_mode': 'ingestionMode',
            'feature_option_details': 'featureOptionDetails',
            'job_output_details': 'jobOutputDetails'
        }

        self._id = None
        self._compartment_id = None
        self._display_name = None
        self._time_created = None
        self._time_started = None
        self._time_finished = None
        self._created_by = None
        self._feature_group_id = None
        self._lifecycle_state = None
        self._time_from = None
        self._time_to = None
        self._ingestion_mode = None
        self._feature_option_details = None
        self._job_output_details = None

    @property
    def id(self):
        """
        Gets the id of this FeatureGroupJob.
        feature group job.


        :return: The id of this FeatureGroupJob.
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """
        Sets the id of this FeatureGroupJob.
        feature group job.


        :param id: The id of this FeatureGroupJob.
        :type: str
        """
        self._id = id

    @property
    def compartment_id(self):
        """
        Gets the compartment_id of this FeatureGroupJob.
        Compartment Identifier


        :return: The compartment_id of this FeatureGroupJob.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this FeatureGroupJob.
        Compartment Identifier


        :param compartment_id: The compartment_id of this FeatureGroupJob.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def display_name(self):
        """
        Gets the display_name of this FeatureGroupJob.
        A user-friendly display name for the resource.


        :return: The display_name of this FeatureGroupJob.
        :rtype: str
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name):
        """
        Sets the display_name of this FeatureGroupJob.
        A user-friendly display name for the resource.


        :param display_name: The display_name of this FeatureGroupJob.
        :type: str
        """
        self._display_name = display_name

    @property
    def time_created(self):
        """
        Gets the time_created of this FeatureGroupJob.

        :return: The time_created of this FeatureGroupJob.
        :rtype: str
        """
        return self._time_created

    @time_created.setter
    def time_created(self, time_created):
        """
        Sets the time_created of this FeatureGroupJob.

        :param time_created: The time_created of this FeatureGroupJob.
        :type: str
        """
        self._time_created = time_created

    @property
    def time_started(self):
        """
        Gets the time_started of this FeatureGroupJob.

        :return: The time_started of this FeatureGroupJob.
        :rtype: str
        """
        return self._time_started

    @time_started.setter
    def time_started(self, time_started):
        """
        Sets the time_started of this FeatureGroupJob.

        :param time_started: The time_started of this FeatureGroupJob.
        :type: str
        """
        self._time_started = time_started

    @property
    def time_finished(self):
        """
        Gets the time_finished of this FeatureGroupJob.

        :return: The time_finished of this FeatureGroupJob.
        :rtype: str
        """
        return self._time_finished

    @time_finished.setter
    def time_finished(self, time_finished):
        """
        Sets the time_finished of this FeatureGroupJob.

        :param time_finished: The time_finished of this FeatureGroupJob.
        :type: str
        """
        self._time_finished = time_finished

    @property
    def created_by(self):
        """
        Gets the created_by of this FeatureGroupJob.
        The `OCID`__ of the user who created the job.

        __ https://docs.cloud.oracle.com/iaas/Content/General/Concepts/identifiers.htm


        :return: The created_by of this FeatureGroupJob.
        :rtype: str
        """
        return self._created_by

    @created_by.setter
    def created_by(self, created_by):
        """
        Sets the created_by of this FeatureGroupJob.
        The `OCID`__ of the user who created the job.

        __ https://docs.cloud.oracle.com/iaas/Content/General/Concepts/identifiers.htm


        :param created_by: The created_by of this FeatureGroupJob.
        :type: str
        """
        self._created_by = created_by

    @property
    def feature_group_id(self):
        """
        Gets the feature_group_id of this FeatureGroupJob.
        Id of the associated feature group


        :return: The feature_group_id of this FeatureGroupJob.
        :rtype: str
        """
        return self._feature_group_id

    @feature_group_id.setter
    def feature_group_id(self, feature_group_id):
        """
        Sets the feature_group_id of this FeatureGroupJob.
        Id of the associated feature group


        :param feature_group_id: The feature_group_id of this FeatureGroupJob.
        :type: str
        """
        self._feature_group_id = feature_group_id

    @property
    def lifecycle_state(self):
        """
        Gets the lifecycle_state of this FeatureGroupJob.
        The current state of the job.

        Allowed values for this property are: "IN_PROGRESS", "FAILED", "SUCCEEDED", 'UNKNOWN_ENUM_VALUE'.
        Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.


        :return: The lifecycle_state of this FeatureGroupJob.
        :rtype: str
        """
        return self._lifecycle_state

    @lifecycle_state.setter
    def lifecycle_state(self, lifecycle_state):
        """
        Sets the lifecycle_state of this FeatureGroupJob.
        The current state of the job.


        :param lifecycle_state: The lifecycle_state of this FeatureGroupJob.
        :type: str
        """
        allowed_values = ["IN_PROGRESS", "FAILED", "SUCCEEDED"]
        if not value_allowed_none_or_none_sentinel(lifecycle_state, allowed_values):
            lifecycle_state = 'UNKNOWN_ENUM_VALUE'
        self._lifecycle_state = lifecycle_state

    @property
    def time_from(self):
        """
        Gets the time_from of this FeatureGroupJob.
        From timestamp for feature group job


        :return: The time_from of this FeatureGroupJob.
        :rtype: datetime
        """
        return self._time_from

    @time_from.setter
    def time_from(self, time_from):
        """
        Sets the time_from of this FeatureGroupJob.
        From timestamp for feature group job


        :param time_from: The time_from of this FeatureGroupJob.
        :type: datetime
        """
        self._time_from = time_from

    @property
    def time_to(self):
        """
        Gets the time_to of this FeatureGroupJob.
        To timestamp for feature group job


        :return: The time_to of this FeatureGroupJob.
        :rtype: datetime
        """
        return self._time_to

    @time_to.setter
    def time_to(self, time_to):
        """
        Sets the time_to of this FeatureGroupJob.
        To timestamp for feature group job


        :param time_to: The time_to of this FeatureGroupJob.
        :type: datetime
        """
        self._time_to = time_to

    @property
    def ingestion_mode(self):
        """
        Gets the ingestion_mode of this FeatureGroupJob.
        The type of the ingestion mode

        Allowed values for this property are: "APPEND", "OVERWRITE", "UPSERT", "COMPLETE", "UPDATE", "DEFAULT", 'UNKNOWN_ENUM_VALUE'.
        Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.


        :return: The ingestion_mode of this FeatureGroupJob.
        :rtype: str
        """
        return self._ingestion_mode

    @ingestion_mode.setter
    def ingestion_mode(self, ingestion_mode):
        """
        Sets the ingestion_mode of this FeatureGroupJob.
        The type of the ingestion mode


        :param ingestion_mode: The ingestion_mode of this FeatureGroupJob.
        :type: str
        """
        allowed_values = ["APPEND", "OVERWRITE", "UPSERT", "COMPLETE", "UPDATE", "DEFAULT"]
        if not value_allowed_none_or_none_sentinel(ingestion_mode, allowed_values):
            ingestion_mode = 'UNKNOWN_ENUM_VALUE'
        self._ingestion_mode = ingestion_mode

    @property
    def feature_option_details(self):
        """
        Gets the feature_option_details of this FeatureGroupJob.

        :return: The feature_option_details of this FeatureGroupJob.
        :rtype: oci.feature_store.models.FeatureOptionDetails
        """
        return self._feature_option_details

    @feature_option_details.setter
    def feature_option_details(self, feature_option_details):
        """
        Sets the feature_option_details of this FeatureGroupJob.

        :param feature_option_details: The feature_option_details of this FeatureGroupJob.
        :type: oci.feature_store.models.FeatureOptionDetails
        """
        self._feature_option_details = feature_option_details

    @property
    def job_output_details(self):
        """
        Gets the job_output_details of this FeatureGroupJob.

        :return: The job_output_details of this FeatureGroupJob.
        :rtype: oci.feature_store.models.FeatureGroupJobOutputDetails
        """
        return self._job_output_details

    @job_output_details.setter
    def job_output_details(self, job_output_details):
        """
        Sets the job_output_details of this FeatureGroupJob.

        :param job_output_details: The job_output_details of this FeatureGroupJob.
        :type: oci.feature_store.models.FeatureGroupJobOutputDetails
        """
        self._job_output_details = job_output_details

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
