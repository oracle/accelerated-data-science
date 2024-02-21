# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class DatasetJobOutputDetails(object):
    """
    Dataset job execution output details
    """

    def __init__(self, **kwargs):
        """
        Initializes a new DatasetJobOutputDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param error_details:
            The value to assign to the error_details property of this DatasetJobOutputDetails.
        :type error_details: str

        :param validation_output:
            The value to assign to the validation_output property of this DatasetJobOutputDetails.
        :type validation_output: str

        :param feature_statistics:
            The value to assign to the feature_statistics property of this DatasetJobOutputDetails.
        :type feature_statistics: str

        :param commit_id:
            The value to assign to the commit_id property of this DatasetJobOutputDetails.
        :type commit_id: str

        :param version:
            The value to assign to the version property of this DatasetJobOutputDetails.
        :type version: int

        """
        self.swagger_types = {
            'error_details': 'str',
            'validation_output': 'str',
            'feature_statistics': 'str',
            'commit_id': 'str',
            'version': 'int'
        }

        self.attribute_map = {
            'error_details': 'errorDetails',
            'validation_output': 'validationOutput',
            'feature_statistics': 'featureStatistics',
            'commit_id': 'commitId',
            'version': 'version'
        }

        self._error_details = None
        self._validation_output = None
        self._feature_statistics = None
        self._commit_id = None
        self._version = None

    @property
    def error_details(self):
        """
        Gets the error_details of this DatasetJobOutputDetails.
        error details for the job if any errors are encountered while executing the job


        :return: The error_details of this DatasetJobOutputDetails.
        :rtype: str
        """
        return self._error_details

    @error_details.setter
    def error_details(self, error_details):
        """
        Sets the error_details of this DatasetJobOutputDetails.
        error details for the job if any errors are encountered while executing the job


        :param error_details: The error_details of this DatasetJobOutputDetails.
        :type: str
        """
        self._error_details = error_details

    @property
    def validation_output(self):
        """
        Gets the validation_output of this DatasetJobOutputDetails.
        validation output


        :return: The validation_output of this DatasetJobOutputDetails.
        :rtype: str
        """
        return self._validation_output

    @validation_output.setter
    def validation_output(self, validation_output):
        """
        Sets the validation_output of this DatasetJobOutputDetails.
        validation output


        :param validation_output: The validation_output of this DatasetJobOutputDetails.
        :type: str
        """
        self._validation_output = validation_output

    @property
    def feature_statistics(self):
        """
        Gets the feature_statistics of this DatasetJobOutputDetails.
        feature statistics for the selected features in the feature group group


        :return: The feature_statistics of this DatasetJobOutputDetails.
        :rtype: str
        """
        return self._feature_statistics

    @feature_statistics.setter
    def feature_statistics(self, feature_statistics):
        """
        Sets the feature_statistics of this DatasetJobOutputDetails.
        feature statistics for the selected features in the feature group group


        :param feature_statistics: The feature_statistics of this DatasetJobOutputDetails.
        :type: str
        """
        self._feature_statistics = feature_statistics

    @property
    def commit_id(self):
        """
        Gets the commit_id of this DatasetJobOutputDetails.
        Commit id for the job


        :return: The commit_id of this DatasetJobOutputDetails.
        :rtype: str
        """
        return self._commit_id

    @commit_id.setter
    def commit_id(self, commit_id):
        """
        Sets the commit_id of this DatasetJobOutputDetails.
        Commit id for the job


        :param commit_id: The commit_id of this DatasetJobOutputDetails.
        :type: str
        """
        self._commit_id = commit_id

    @property
    def version(self):
        """
        Gets the version of this DatasetJobOutputDetails.
        Version number for statistics or validation changes


        :return: The version of this DatasetJobOutputDetails.
        :rtype: int
        """
        return self._version

    @version.setter
    def version(self, version):
        """
        Sets the version of this DatasetJobOutputDetails.
        Version number for statistics or validation changes


        :param version: The version of this DatasetJobOutputDetails.
        :type: int
        """
        self._version = version

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
