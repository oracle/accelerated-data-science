# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class FeatureGroupJobValidationSummary(object):
    """
    Feature group job
    """

    def __init__(self, **kwargs):
        """
        Initializes a new FeatureGroupJobValidationSummary object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param id:
            The value to assign to the id property of this FeatureGroupJobValidationSummary.
        :type id: str

        :param compartment_id:
            The value to assign to the compartment_id property of this FeatureGroupJobValidationSummary.
        :type compartment_id: str

        :param display_name:
            The value to assign to the display_name property of this FeatureGroupJobValidationSummary.
        :type display_name: str

        :param time_created:
            The value to assign to the time_created property of this FeatureGroupJobValidationSummary.
        :type time_created: str

        :param time_started:
            The value to assign to the time_started property of this FeatureGroupJobValidationSummary.
        :type time_started: str

        :param time_finished:
            The value to assign to the time_finished property of this FeatureGroupJobValidationSummary.
        :type time_finished: str

        :param validation_details:
            The value to assign to the validation_details property of this FeatureGroupJobValidationSummary.
        :type validation_details: oci.feature_store.models.FeatureGroupJobValidationOutputDetails

        """
        self.swagger_types = {
            'id': 'str',
            'compartment_id': 'str',
            'display_name': 'str',
            'time_created': 'str',
            'time_started': 'str',
            'time_finished': 'str',
            'validation_details': 'FeatureGroupJobValidationOutputDetails'
        }

        self.attribute_map = {
            'id': 'id',
            'compartment_id': 'compartmentId',
            'display_name': 'displayName',
            'time_created': 'timeCreated',
            'time_started': 'timeStarted',
            'time_finished': 'timeFinished',
            'validation_details': 'validationDetails'
        }

        self._id = None
        self._compartment_id = None
        self._display_name = None
        self._time_created = None
        self._time_started = None
        self._time_finished = None
        self._validation_details = None

    @property
    def id(self):
        """
        **[Required]** Gets the id of this FeatureGroupJobValidationSummary.
        Feature group job id.


        :return: The id of this FeatureGroupJobValidationSummary.
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id):
        """
        Sets the id of this FeatureGroupJobValidationSummary.
        Feature group job id.


        :param id: The id of this FeatureGroupJobValidationSummary.
        :type: str
        """
        self._id = id

    @property
    def compartment_id(self):
        """
        **[Required]** Gets the compartment_id of this FeatureGroupJobValidationSummary.
        Compartment Identifier


        :return: The compartment_id of this FeatureGroupJobValidationSummary.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this FeatureGroupJobValidationSummary.
        Compartment Identifier


        :param compartment_id: The compartment_id of this FeatureGroupJobValidationSummary.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def display_name(self):
        """
        Gets the display_name of this FeatureGroupJobValidationSummary.
        A user-friendly display name for the resource.


        :return: The display_name of this FeatureGroupJobValidationSummary.
        :rtype: str
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name):
        """
        Sets the display_name of this FeatureGroupJobValidationSummary.
        A user-friendly display name for the resource.


        :param display_name: The display_name of this FeatureGroupJobValidationSummary.
        :type: str
        """
        self._display_name = display_name

    @property
    def time_created(self):
        """
        Gets the time_created of this FeatureGroupJobValidationSummary.
        The date and time the job request was created in the timestamp format defined by `RFC3339`__.

        __ https://tools.ietf.org/html/rfc3339


        :return: The time_created of this FeatureGroupJobValidationSummary.
        :rtype: str
        """
        return self._time_created

    @time_created.setter
    def time_created(self, time_created):
        """
        Sets the time_created of this FeatureGroupJobValidationSummary.
        The date and time the job request was created in the timestamp format defined by `RFC3339`__.

        __ https://tools.ietf.org/html/rfc3339


        :param time_created: The time_created of this FeatureGroupJobValidationSummary.
        :type: str
        """
        self._time_created = time_created

    @property
    def time_started(self):
        """
        **[Required]** Gets the time_started of this FeatureGroupJobValidationSummary.
        The date and time the job request was started in the timestamp format defined by `RFC3339`__.

        __ https://tools.ietf.org/html/rfc3339


        :return: The time_started of this FeatureGroupJobValidationSummary.
        :rtype: str
        """
        return self._time_started

    @time_started.setter
    def time_started(self, time_started):
        """
        Sets the time_started of this FeatureGroupJobValidationSummary.
        The date and time the job request was started in the timestamp format defined by `RFC3339`__.

        __ https://tools.ietf.org/html/rfc3339


        :param time_started: The time_started of this FeatureGroupJobValidationSummary.
        :type: str
        """
        self._time_started = time_started

    @property
    def time_finished(self):
        """
        Gets the time_finished of this FeatureGroupJobValidationSummary.
        The date and time the job request was finished in the timestamp format defined by `RFC3339`__.

        __ https://tools.ietf.org/html/rfc3339


        :return: The time_finished of this FeatureGroupJobValidationSummary.
        :rtype: str
        """
        return self._time_finished

    @time_finished.setter
    def time_finished(self, time_finished):
        """
        Sets the time_finished of this FeatureGroupJobValidationSummary.
        The date and time the job request was finished in the timestamp format defined by `RFC3339`__.

        __ https://tools.ietf.org/html/rfc3339


        :param time_finished: The time_finished of this FeatureGroupJobValidationSummary.
        :type: str
        """
        self._time_finished = time_finished

    @property
    def validation_details(self):
        """
        Gets the validation_details of this FeatureGroupJobValidationSummary.

        :return: The validation_details of this FeatureGroupJobValidationSummary.
        :rtype: oci.feature_store.models.FeatureGroupJobValidationOutputDetails
        """
        return self._validation_details

    @validation_details.setter
    def validation_details(self, validation_details):
        """
        Sets the validation_details of this FeatureGroupJobValidationSummary.

        :param validation_details: The validation_details of this FeatureGroupJobValidationSummary.
        :type: oci.feature_store.models.FeatureGroupJobValidationOutputDetails
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
