# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class DatasetJobStatisticsSummary(object):
    """
    Dataset Job of feature store construct with statistics details
    """

    def __init__(self, **kwargs):
        """
        Initializes a new DatasetJobStatisticsSummary object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param id:
            The value to assign to the id property of this DatasetJobStatisticsSummary.
        :type id: str

        :param compartment_id:
            The value to assign to the compartment_id property of this DatasetJobStatisticsSummary.
        :type compartment_id: str

        :param display_name:
            The value to assign to the display_name property of this DatasetJobStatisticsSummary.
        :type display_name: str

        :param time_started:
            The value to assign to the time_started property of this DatasetJobStatisticsSummary.
        :type time_started: str

        :param time_created:
            The value to assign to the time_created property of this DatasetJobStatisticsSummary.
        :type time_created: str

        :param time_finished:
            The value to assign to the time_finished property of this DatasetJobStatisticsSummary.
        :type time_finished: str

        :param statistics_details:
            The value to assign to the statistics_details property of this DatasetJobStatisticsSummary.
        :type statistics_details: oci.feature_store.models.DatasetJobStatisticsDetails

        """
        self.swagger_types = {
            'id': 'str',
            'compartment_id': 'str',
            'display_name': 'str',
            'time_started': 'str',
            'time_created': 'str',
            'time_finished': 'str',
            'statistics_details': 'DatasetJobStatisticsDetails'
        }

        self.attribute_map = {
            'id': 'id',
            'compartment_id': 'compartmentId',
            'display_name': 'displayName',
            'time_started': 'timeStarted',
            'time_created': 'timeCreated',
            'time_finished': 'timeFinished',
            'statistics_details': 'statisticsDetails'
        }

        self._id = None
        self._compartment_id = None
        self._display_name = None
        self._time_started = None
        self._time_created = None
        self._time_finished = None
        self._statistics_details = None

    @property
    def id(self):
        """
        **[Required]** Gets the id of this DatasetJobStatisticsSummary.
        The GUID of the construct.


        :return: The id of this DatasetJobStatisticsSummary.
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """
        Sets the id of this DatasetJobStatisticsSummary.
        The GUID of the construct.


        :param id: The id of this DatasetJobStatisticsSummary.
        :type: str
        """
        self._id = id

    @property
    def compartment_id(self):
        """
        **[Required]** Gets the compartment_id of this DatasetJobStatisticsSummary.
        Compartment Identifier


        :return: The compartment_id of this DatasetJobStatisticsSummary.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this DatasetJobStatisticsSummary.
        Compartment Identifier


        :param compartment_id: The compartment_id of this DatasetJobStatisticsSummary.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def display_name(self):
        """
        Gets the display_name of this DatasetJobStatisticsSummary.
        A user-friendly display name for the resource.


        :return: The display_name of this DatasetJobStatisticsSummary.
        :rtype: str
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name):
        """
        Sets the display_name of this DatasetJobStatisticsSummary.
        A user-friendly display name for the resource.


        :param display_name: The display_name of this DatasetJobStatisticsSummary.
        :type: str
        """
        self._display_name = display_name

    @property
    def time_started(self):
        """
        **[Required]** Gets the time_started of this DatasetJobStatisticsSummary.
        The date and time the job request was started in the timestamp format defined by `RFC3339`__.

        __ https://tools.ietf.org/html/rfc3339


        :return: The time_started of this DatasetJobStatisticsSummary.
        :rtype: str
        """
        return self._time_started

    @time_started.setter
    def time_started(self, time_started):
        """
        Sets the time_started of this DatasetJobStatisticsSummary.
        The date and time the job request was started in the timestamp format defined by `RFC3339`__.

        __ https://tools.ietf.org/html/rfc3339


        :param time_started: The time_started of this DatasetJobStatisticsSummary.
        :type: str
        """
        self._time_started = time_started

    @property
    def time_created(self):
        """
        Gets the time_created of this DatasetJobStatisticsSummary.
        The date and time the job request was created in the timestamp format defined by `RFC3339`__.

        __ https://tools.ietf.org/html/rfc3339


        :return: The time_created of this DatasetJobStatisticsSummary.
        :rtype: str
        """
        return self._time_created

    @time_created.setter
    def time_created(self, time_created):
        """
        Sets the time_created of this DatasetJobStatisticsSummary.
        The date and time the job request was created in the timestamp format defined by `RFC3339`__.

        __ https://tools.ietf.org/html/rfc3339


        :param time_created: The time_created of this DatasetJobStatisticsSummary.
        :type: str
        """
        self._time_created = time_created

    @property
    def time_finished(self):
        """
        Gets the time_finished of this DatasetJobStatisticsSummary.
        The date and time the job request was finished in the timestamp format defined by `RFC3339`__.

        __ https://tools.ietf.org/html/rfc3339


        :return: The time_finished of this DatasetJobStatisticsSummary.
        :rtype: str
        """
        return self._time_finished

    @time_finished.setter
    def time_finished(self, time_finished):
        """
        Sets the time_finished of this DatasetJobStatisticsSummary.
        The date and time the job request was finished in the timestamp format defined by `RFC3339`__.

        __ https://tools.ietf.org/html/rfc3339


        :param time_finished: The time_finished of this DatasetJobStatisticsSummary.
        :type: str
        """
        self._time_finished = time_finished

    @property
    def statistics_details(self):
        """
        Gets the statistics_details of this DatasetJobStatisticsSummary.

        :return: The statistics_details of this DatasetJobStatisticsSummary.
        :rtype: oci.feature_store.models.DatasetJobStatisticsDetails
        """
        return self._statistics_details

    @statistics_details.setter
    def statistics_details(self, statistics_details):
        """
        Sets the statistics_details of this DatasetJobStatisticsSummary.

        :param statistics_details: The statistics_details of this DatasetJobStatisticsSummary.
        :type: oci.feature_store.models.DatasetJobStatisticsDetails
        """
        self._statistics_details = statistics_details

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
