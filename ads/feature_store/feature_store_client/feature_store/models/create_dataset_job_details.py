# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class CreateDatasetJobDetails(object):
    """
    Dataset save request
    """

    #: A constant which can be used with the ingestion_mode property of a CreateDatasetJobDetails.
    #: This constant has a value of "APPEND"
    INGESTION_MODE_APPEND = "APPEND"

    #: A constant which can be used with the ingestion_mode property of a CreateDatasetJobDetails.
    #: This constant has a value of "OVERWRITE"
    INGESTION_MODE_OVERWRITE = "OVERWRITE"

    #: A constant which can be used with the ingestion_mode property of a CreateDatasetJobDetails.
    #: This constant has a value of "UPSERT"
    INGESTION_MODE_UPSERT = "UPSERT"

    #: A constant which can be used with the ingestion_mode property of a CreateDatasetJobDetails.
    #: This constant has a value of "COMPLETE"
    INGESTION_MODE_COMPLETE = "COMPLETE"

    #: A constant which can be used with the ingestion_mode property of a CreateDatasetJobDetails.
    #: This constant has a value of "UPDATE"
    INGESTION_MODE_UPDATE = "UPDATE"

    #: A constant which can be used with the ingestion_mode property of a CreateDatasetJobDetails.
    #: This constant has a value of "DEFAULT"
    INGESTION_MODE_DEFAULT = "DEFAULT"

    def __init__(self, **kwargs):
        """
        Initializes a new CreateDatasetJobDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param display_name:
            The value to assign to the display_name property of this CreateDatasetJobDetails.
        :type display_name: str

        :param compartment_id:
            The value to assign to the compartment_id property of this CreateDatasetJobDetails.
        :type compartment_id: str

        :param ingestion_mode:
            The value to assign to the ingestion_mode property of this CreateDatasetJobDetails.
            Allowed values for this property are: "APPEND", "OVERWRITE", "UPSERT", "COMPLETE", "UPDATE", "DEFAULT"
        :type ingestion_mode: str

        :param feature_option_details:
            The value to assign to the feature_option_details property of this CreateDatasetJobDetails.
        :type feature_option_details: oci.feature_store.models.FeatureOptionDetails

        :param dataset_id:
            The value to assign to the dataset_id property of this CreateDatasetJobDetails.
        :type dataset_id: str

        """
        self.swagger_types = {
            'display_name': 'str',
            'compartment_id': 'str',
            'ingestion_mode': 'str',
            'feature_option_details': 'FeatureOptionDetails',
            'dataset_id': 'str'
        }

        self.attribute_map = {
            'display_name': 'displayName',
            'compartment_id': 'compartmentId',
            'ingestion_mode': 'ingestionMode',
            'feature_option_details': 'featureOptionDetails',
            'dataset_id': 'datasetId'
        }

        self._display_name = None
        self._compartment_id = None
        self._ingestion_mode = None
        self._feature_option_details = None
        self._dataset_id = None

    @property
    def display_name(self):
        """
        **[Required]** Gets the display_name of this CreateDatasetJobDetails.
        FeatureStore dataset job Identifier, can be renamed


        :return: The display_name of this CreateDatasetJobDetails.
        :rtype: str
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name):
        """
        Sets the display_name of this CreateDatasetJobDetails.
        FeatureStore dataset job Identifier, can be renamed


        :param display_name: The display_name of this CreateDatasetJobDetails.
        :type: str
        """
        self._display_name = display_name

    @property
    def compartment_id(self):
        """
        **[Required]** Gets the compartment_id of this CreateDatasetJobDetails.
        The OCID of the compartment containing the Dataset.


        :return: The compartment_id of this CreateDatasetJobDetails.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this CreateDatasetJobDetails.
        The OCID of the compartment containing the Dataset.


        :param compartment_id: The compartment_id of this CreateDatasetJobDetails.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def ingestion_mode(self):
        """
        **[Required]** Gets the ingestion_mode of this CreateDatasetJobDetails.
        The type of the ingestion mode

        Allowed values for this property are: "APPEND", "OVERWRITE", "UPSERT", "COMPLETE", "UPDATE", "DEFAULT"


        :return: The ingestion_mode of this CreateDatasetJobDetails.
        :rtype: str
        """
        return self._ingestion_mode

    @ingestion_mode.setter
    def ingestion_mode(self, ingestion_mode):
        """
        Sets the ingestion_mode of this CreateDatasetJobDetails.
        The type of the ingestion mode


        :param ingestion_mode: The ingestion_mode of this CreateDatasetJobDetails.
        :type: str
        """
        allowed_values = ["APPEND", "OVERWRITE", "UPSERT", "COMPLETE", "UPDATE", "DEFAULT"]
        if not value_allowed_none_or_none_sentinel(ingestion_mode, allowed_values):
            raise ValueError(
                "Invalid value for `ingestion_mode`, must be None or one of {0}"
                .format(allowed_values)
            )
        self._ingestion_mode = ingestion_mode

    @property
    def feature_option_details(self):
        """
        Gets the feature_option_details of this CreateDatasetJobDetails.

        :return: The feature_option_details of this CreateDatasetJobDetails.
        :rtype: oci.feature_store.models.FeatureOptionDetails
        """
        return self._feature_option_details

    @feature_option_details.setter
    def feature_option_details(self, feature_option_details):
        """
        Sets the feature_option_details of this CreateDatasetJobDetails.

        :param feature_option_details: The feature_option_details of this CreateDatasetJobDetails.
        :type: oci.feature_store.models.FeatureOptionDetails
        """
        self._feature_option_details = feature_option_details

    @property
    def dataset_id(self):
        """
        **[Required]** Gets the dataset_id of this CreateDatasetJobDetails.
        Id of the dataset for which job is to be created


        :return: The dataset_id of this CreateDatasetJobDetails.
        :rtype: str
        """
        return self._dataset_id

    @dataset_id.setter
    def dataset_id(self, dataset_id):
        """
        Sets the dataset_id of this CreateDatasetJobDetails.
        Id of the dataset for which job is to be created


        :param dataset_id: The dataset_id of this CreateDatasetJobDetails.
        :type: str
        """
        self._dataset_id = dataset_id

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
