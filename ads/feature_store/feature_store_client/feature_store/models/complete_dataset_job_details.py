# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class CompleteDatasetJobDetails(object):
    """
    Parameters needed to update an existing dataset Job.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new CompleteDatasetJobDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param job_output_details:
            The value to assign to the job_output_details property of this CompleteDatasetJobDetails.
        :type job_output_details: oci.feature_store.models.DatasetJobOutputDetails

        """
        self.swagger_types = {
            'job_output_details': 'DatasetJobOutputDetails'
        }

        self.attribute_map = {
            'job_output_details': 'jobOutputDetails'
        }

        self._job_output_details = None

    @property
    def job_output_details(self):
        """
        Gets the job_output_details of this CompleteDatasetJobDetails.

        :return: The job_output_details of this CompleteDatasetJobDetails.
        :rtype: oci.feature_store.models.DatasetJobOutputDetails
        """
        return self._job_output_details

    @job_output_details.setter
    def job_output_details(self, job_output_details):
        """
        Sets the job_output_details of this CompleteDatasetJobDetails.

        :param job_output_details: The job_output_details of this CompleteDatasetJobDetails.
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
