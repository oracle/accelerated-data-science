# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class DatasetJob(object):
    """
    Job of feature store construct
    """

    #: A constant which can be used with the lifecycle_state property of a DatasetJob.
    #: This constant has a value of "IN_PROGRESS"
    LIFECYCLE_STATE_IN_PROGRESS = "IN_PROGRESS"

    #: A constant which can be used with the lifecycle_state property of a DatasetJob.
    #: This constant has a value of "FAILED"
    LIFECYCLE_STATE_FAILED = "FAILED"

    #: A constant which can be used with the lifecycle_state property of a DatasetJob.
    #: This constant has a value of "SUCCEEDED"
    LIFECYCLE_STATE_SUCCEEDED = "SUCCEEDED"

    #: A constant which can be used with the ingestion_mode property of a DatasetJob.
    #: This constant has a value of "APPEND"
    INGESTION_MODE_APPEND = "APPEND"

    #: A constant which can be used with the ingestion_mode property of a DatasetJob.
    #: This constant has a value of "OVERWRITE"
    INGESTION_MODE_OVERWRITE = "OVERWRITE"

    #: A constant which can be used with the ingestion_mode property of a DatasetJob.
    #: This constant has a value of "UPSERT"
    INGESTION_MODE_UPSERT = "UPSERT"

    #: A constant which can be used with the ingestion_mode property of a DatasetJob.
    #: This constant has a value of "COMPLETE"
    INGESTION_MODE_COMPLETE = "COMPLETE"

    #: A constant which can be used with the ingestion_mode property of a DatasetJob.
    #: This constant has a value of "UPDATE"
    INGESTION_MODE_UPDATE = "UPDATE"

    #: A constant which can be used with the ingestion_mode property of a DatasetJob.
    #: This constant has a value of "DEFAULT"
    INGESTION_MODE_DEFAULT = "DEFAULT"

    def __init__(self, **kwargs):
        """
        Initializes a new DatasetJob object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param id:
            The value to assign to the id property of this DatasetJob.
        :type id: str

        :param display_name:
            The value to assign to the display_name property of this DatasetJob.
        :type display_name: str

        :param compartment_id:
            The value to assign to the compartment_id property of this DatasetJob.
        :type compartment_id: str

        :param time_created:
            The value to assign to the time_created property of this DatasetJob.
        :type time_created: str

        :param time_started:
            The value to assign to the time_started property of this DatasetJob.
        :type time_started: str

        :param time_finished:
            The value to assign to the time_finished property of this DatasetJob.
        :type time_finished: str

        :param created_by:
            The value to assign to the created_by property of this DatasetJob.
        :type created_by: str

        :param dataset_id:
            The value to assign to the dataset_id property of this DatasetJob.
        :type dataset_id: str

        :param lifecycle_state:
            The value to assign to the lifecycle_state property of this DatasetJob.
            Allowed values for this property are: "IN_PROGRESS", "FAILED", "SUCCEEDED", 'UNKNOWN_ENUM_VALUE'.
            Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.
        :type lifecycle_state: str

        :param ingestion_mode:
            The value to assign to the ingestion_mode property of this DatasetJob.
            Allowed values for this property are: "APPEND", "OVERWRITE", "UPSERT", "COMPLETE", "UPDATE", "DEFAULT", 'UNKNOWN_ENUM_VALUE'.
            Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.
        :type ingestion_mode: str

        :param feature_option_details:
            The value to assign to the feature_option_details property of this DatasetJob.
        :type feature_option_details: oci.feature_store.models.FeatureOptionDetails

        :param job_output_details:
            The value to assign to the job_output_details property of this DatasetJob.
        :type job_output_details: oci.feature_store.models.DatasetJobOutputDetails

        """
        self.swagger_types = {
            'id': 'str',
            'display_name': 'str',
            'compartment_id': 'str',
            'time_created': 'str',
            'time_started': 'str',
            'time_finished': 'str',
            'created_by': 'str',
            'dataset_id': 'str',
            'lifecycle_state': 'str',
            'ingestion_mode': 'str',
            'feature_option_details': 'FeatureOptionDetails',
            'job_output_details': 'DatasetJobOutputDetails'
        }

        self.attribute_map = {
            'id': 'id',
            'display_name': 'displayName',
            'compartment_id': 'compartmentId',
            'time_created': 'timeCreated',
            'time_started': 'timeStarted',
            'time_finished': 'timeFinished',
            'created_by': 'createdBy',
            'dataset_id': 'datasetId',
            'lifecycle_state': 'lifecycleState',
            'ingestion_mode': 'ingestionMode',
            'feature_option_details': 'featureOptionDetails',
            'job_output_details': 'jobOutputDetails'
        }

        self._id = None
        self._display_name = None
        self._compartment_id = None
        self._time_created = None
        self._time_started = None
        self._time_finished = None
        self._created_by = None
        self._dataset_id = None
        self._lifecycle_state = None
        self._ingestion_mode = None
        self._feature_option_details = None
        self._job_output_details = None

    @property
    def id(self):
        """
        **[Required]** Gets the id of this DatasetJob.
        The GUID of the construct.


        :return: The id of this DatasetJob.
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """
        Sets the id of this DatasetJob.
        The GUID of the construct.


        :param id: The id of this DatasetJob.
        :type: str
        """
        self._id = id

    @property
    def display_name(self):
        """
        **[Required]** Gets the display_name of this DatasetJob.
        FeatureStore dataset Identifier, can be renamed


        :return: The display_name of this DatasetJob.
        :rtype: str
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name):
        """
        Sets the display_name of this DatasetJob.
        FeatureStore dataset Identifier, can be renamed


        :param display_name: The display_name of this DatasetJob.
        :type: str
        """
        self._display_name = display_name

    @property
    def compartment_id(self):
        """
        **[Required]** Gets the compartment_id of this DatasetJob.
        Compartment Identifier


        :return: The compartment_id of this DatasetJob.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this DatasetJob.
        Compartment Identifier


        :param compartment_id: The compartment_id of this DatasetJob.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def time_created(self):
        """
        Gets the time_created of this DatasetJob.

        :return: The time_created of this DatasetJob.
        :rtype: str
        """
        return self._time_created

    @time_created.setter
    def time_created(self, time_created):
        """
        Sets the time_created of this DatasetJob.

        :param time_created: The time_created of this DatasetJob.
        :type: str
        """
        self._time_created = time_created

    @property
    def time_started(self):
        """
        **[Required]** Gets the time_started of this DatasetJob.

        :return: The time_started of this DatasetJob.
        :rtype: str
        """
        return self._time_started

    @time_started.setter
    def time_started(self, time_started):
        """
        Sets the time_started of this DatasetJob.

        :param time_started: The time_started of this DatasetJob.
        :type: str
        """
        self._time_started = time_started

    @property
    def time_finished(self):
        """
        Gets the time_finished of this DatasetJob.

        :return: The time_finished of this DatasetJob.
        :rtype: str
        """
        return self._time_finished

    @time_finished.setter
    def time_finished(self, time_finished):
        """
        Sets the time_finished of this DatasetJob.

        :param time_finished: The time_finished of this DatasetJob.
        :type: str
        """
        self._time_finished = time_finished

    @property
    def created_by(self):
        """
        **[Required]** Gets the created_by of this DatasetJob.
        The `OCID`__ of the user who created the job.

        __ https://docs.cloud.oracle.com/iaas/Content/General/Concepts/identifiers.htm


        :return: The created_by of this DatasetJob.
        :rtype: str
        """
        return self._created_by

    @created_by.setter
    def created_by(self, created_by):
        """
        Sets the created_by of this DatasetJob.
        The `OCID`__ of the user who created the job.

        __ https://docs.cloud.oracle.com/iaas/Content/General/Concepts/identifiers.htm


        :param created_by: The created_by of this DatasetJob.
        :type: str
        """
        self._created_by = created_by

    @property
    def dataset_id(self):
        """
        Gets the dataset_id of this DatasetJob.
        Dataset Id


        :return: The dataset_id of this DatasetJob.
        :rtype: str
        """
        return self._dataset_id

    @dataset_id.setter
    def dataset_id(self, dataset_id):
        """
        Sets the dataset_id of this DatasetJob.
        Dataset Id


        :param dataset_id: The dataset_id of this DatasetJob.
        :type: str
        """
        self._dataset_id = dataset_id

    @property
    def lifecycle_state(self):
        """
        **[Required]** Gets the lifecycle_state of this DatasetJob.
        The current state of the job.

        Allowed values for this property are: "IN_PROGRESS", "FAILED", "SUCCEEDED", 'UNKNOWN_ENUM_VALUE'.
        Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.


        :return: The lifecycle_state of this DatasetJob.
        :rtype: str
        """
        return self._lifecycle_state

    @lifecycle_state.setter
    def lifecycle_state(self, lifecycle_state):
        """
        Sets the lifecycle_state of this DatasetJob.
        The current state of the job.


        :param lifecycle_state: The lifecycle_state of this DatasetJob.
        :type: str
        """
        allowed_values = ["IN_PROGRESS", "FAILED", "SUCCEEDED"]
        if not value_allowed_none_or_none_sentinel(lifecycle_state, allowed_values):
            lifecycle_state = 'UNKNOWN_ENUM_VALUE'
        self._lifecycle_state = lifecycle_state

    @property
    def ingestion_mode(self):
        """
        Gets the ingestion_mode of this DatasetJob.
        The type of the ingestion mode

        Allowed values for this property are: "APPEND", "OVERWRITE", "UPSERT", "COMPLETE", "UPDATE", "DEFAULT", 'UNKNOWN_ENUM_VALUE'.
        Any unrecognized values returned by a service will be mapped to 'UNKNOWN_ENUM_VALUE'.


        :return: The ingestion_mode of this DatasetJob.
        :rtype: str
        """
        return self._ingestion_mode

    @ingestion_mode.setter
    def ingestion_mode(self, ingestion_mode):
        """
        Sets the ingestion_mode of this DatasetJob.
        The type of the ingestion mode


        :param ingestion_mode: The ingestion_mode of this DatasetJob.
        :type: str
        """
        allowed_values = ["APPEND", "OVERWRITE", "UPSERT", "COMPLETE", "UPDATE", "DEFAULT"]
        if not value_allowed_none_or_none_sentinel(ingestion_mode, allowed_values):
            ingestion_mode = 'UNKNOWN_ENUM_VALUE'
        self._ingestion_mode = ingestion_mode

    @property
    def feature_option_details(self):
        """
        Gets the feature_option_details of this DatasetJob.

        :return: The feature_option_details of this DatasetJob.
        :rtype: oci.feature_store.models.FeatureOptionDetails
        """
        return self._feature_option_details

    @feature_option_details.setter
    def feature_option_details(self, feature_option_details):
        """
        Sets the feature_option_details of this DatasetJob.

        :param feature_option_details: The feature_option_details of this DatasetJob.
        :type: oci.feature_store.models.FeatureOptionDetails
        """
        self._feature_option_details = feature_option_details

    @property
    def job_output_details(self):
        """
        Gets the job_output_details of this DatasetJob.

        :return: The job_output_details of this DatasetJob.
        :rtype: oci.feature_store.models.DatasetJobOutputDetails
        """
        return self._job_output_details

    @job_output_details.setter
    def job_output_details(self, job_output_details):
        """
        Sets the job_output_details of this DatasetJob.

        :param job_output_details: The job_output_details of this DatasetJob.
        :type: oci.feature_store.models.DatasetJobOutputDetails
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
