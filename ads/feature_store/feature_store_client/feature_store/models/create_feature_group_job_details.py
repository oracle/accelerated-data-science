# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class CreateFeatureGroupJobDetails(object):
    """
    Dataset save request
    """

    #: A constant which can be used with the ingestion_mode property of a CreateFeatureGroupJobDetails.
    #: This constant has a value of "APPEND"
    INGESTION_MODE_APPEND = "APPEND"

    #: A constant which can be used with the ingestion_mode property of a CreateFeatureGroupJobDetails.
    #: This constant has a value of "OVERWRITE"
    INGESTION_MODE_OVERWRITE = "OVERWRITE"

    #: A constant which can be used with the ingestion_mode property of a CreateFeatureGroupJobDetails.
    #: This constant has a value of "UPSERT"
    INGESTION_MODE_UPSERT = "UPSERT"

    #: A constant which can be used with the ingestion_mode property of a CreateFeatureGroupJobDetails.
    #: This constant has a value of "COMPLETE"
    INGESTION_MODE_COMPLETE = "COMPLETE"

    #: A constant which can be used with the ingestion_mode property of a CreateFeatureGroupJobDetails.
    #: This constant has a value of "UPDATE"
    INGESTION_MODE_UPDATE = "UPDATE"

    #: A constant which can be used with the ingestion_mode property of a CreateFeatureGroupJobDetails.
    #: This constant has a value of "DEFAULT"
    INGESTION_MODE_DEFAULT = "DEFAULT"

    def __init__(self, **kwargs):
        """
        Initializes a new CreateFeatureGroupJobDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param compartment_id:
            The value to assign to the compartment_id property of this CreateFeatureGroupJobDetails.
        :type compartment_id: str

        :param display_name:
            The value to assign to the display_name property of this CreateFeatureGroupJobDetails.
        :type display_name: str

        :param time_from:
            The value to assign to the time_from property of this CreateFeatureGroupJobDetails.
        :type time_from: datetime

        :param time_to:
            The value to assign to the time_to property of this CreateFeatureGroupJobDetails.
        :type time_to: datetime

        :param ingestion_mode:
            The value to assign to the ingestion_mode property of this CreateFeatureGroupJobDetails.
            Allowed values for this property are: "APPEND", "OVERWRITE", "UPSERT", "COMPLETE", "UPDATE", "DEFAULT"
        :type ingestion_mode: str

        :param feature_option_details:
            The value to assign to the feature_option_details property of this CreateFeatureGroupJobDetails.
        :type feature_option_details: oci.feature_store.models.FeatureOptionDetails

        :param feature_group_id:
            The value to assign to the feature_group_id property of this CreateFeatureGroupJobDetails.
        :type feature_group_id: str

        """
        self.swagger_types = {
            'compartment_id': 'str',
            'display_name': 'str',
            'time_from': 'datetime',
            'time_to': 'datetime',
            'ingestion_mode': 'str',
            'feature_option_details': 'FeatureOptionDetails',
            'feature_group_id': 'str'
        }

        self.attribute_map = {
            'compartment_id': 'compartmentId',
            'display_name': 'displayName',
            'time_from': 'timeFrom',
            'time_to': 'timeTo',
            'ingestion_mode': 'ingestionMode',
            'feature_option_details': 'featureOptionDetails',
            'feature_group_id': 'featureGroupId'
        }

        self._compartment_id = None
        self._display_name = None
        self._time_from = None
        self._time_to = None
        self._ingestion_mode = None
        self._feature_option_details = None
        self._feature_group_id = None

    @property
    def compartment_id(self):
        """
        **[Required]** Gets the compartment_id of this CreateFeatureGroupJobDetails.
        The OCID of the compartment containing the DataAsset.


        :return: The compartment_id of this CreateFeatureGroupJobDetails.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this CreateFeatureGroupJobDetails.
        The OCID of the compartment containing the DataAsset.


        :param compartment_id: The compartment_id of this CreateFeatureGroupJobDetails.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def display_name(self):
        """
        Gets the display_name of this CreateFeatureGroupJobDetails.
        A user-friendly display name for the resource.


        :return: The display_name of this CreateFeatureGroupJobDetails.
        :rtype: str
        """
        return self._display_name

    @display_name.setter
    def display_name(self, display_name):
        """
        Sets the display_name of this CreateFeatureGroupJobDetails.
        A user-friendly display name for the resource.


        :param display_name: The display_name of this CreateFeatureGroupJobDetails.
        :type: str
        """
        self._display_name = display_name

    @property
    def time_from(self):
        """
        Gets the time_from of this CreateFeatureGroupJobDetails.
        From timestamp for dataset


        :return: The time_from of this CreateFeatureGroupJobDetails.
        :rtype: datetime
        """
        return self._time_from

    @time_from.setter
    def time_from(self, time_from):
        """
        Sets the time_from of this CreateFeatureGroupJobDetails.
        From timestamp for dataset


        :param time_from: The time_from of this CreateFeatureGroupJobDetails.
        :type: datetime
        """
        self._time_from = time_from

    @property
    def time_to(self):
        """
        Gets the time_to of this CreateFeatureGroupJobDetails.
        From timestamp for dataset


        :return: The time_to of this CreateFeatureGroupJobDetails.
        :rtype: datetime
        """
        return self._time_to

    @time_to.setter
    def time_to(self, time_to):
        """
        Sets the time_to of this CreateFeatureGroupJobDetails.
        From timestamp for dataset


        :param time_to: The time_to of this CreateFeatureGroupJobDetails.
        :type: datetime
        """
        self._time_to = time_to

    @property
    def ingestion_mode(self):
        """
        **[Required]** Gets the ingestion_mode of this CreateFeatureGroupJobDetails.
        The type of the ingestion mode

        Allowed values for this property are: "APPEND", "OVERWRITE", "UPSERT", "COMPLETE", "UPDATE", "DEFAULT"


        :return: The ingestion_mode of this CreateFeatureGroupJobDetails.
        :rtype: str
        """
        return self._ingestion_mode

    @ingestion_mode.setter
    def ingestion_mode(self, ingestion_mode):
        """
        Sets the ingestion_mode of this CreateFeatureGroupJobDetails.
        The type of the ingestion mode


        :param ingestion_mode: The ingestion_mode of this CreateFeatureGroupJobDetails.
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
        Gets the feature_option_details of this CreateFeatureGroupJobDetails.

        :return: The feature_option_details of this CreateFeatureGroupJobDetails.
        :rtype: oci.feature_store.models.FeatureOptionDetails
        """
        return self._feature_option_details

    @feature_option_details.setter
    def feature_option_details(self, feature_option_details):
        """
        Sets the feature_option_details of this CreateFeatureGroupJobDetails.

        :param feature_option_details: The feature_option_details of this CreateFeatureGroupJobDetails.
        :type: oci.feature_store.models.FeatureOptionDetails
        """
        self._feature_option_details = feature_option_details

    @property
    def feature_group_id(self):
        """
        **[Required]** Gets the feature_group_id of this CreateFeatureGroupJobDetails.
        Id of the feature group for which job is to be created


        :return: The feature_group_id of this CreateFeatureGroupJobDetails.
        :rtype: str
        """
        return self._feature_group_id

    @feature_group_id.setter
    def feature_group_id(self, feature_group_id):
        """
        Sets the feature_group_id of this CreateFeatureGroupJobDetails.
        Id of the feature group for which job is to be created


        :param feature_group_id: The feature_group_id of this CreateFeatureGroupJobDetails.
        :type: str
        """
        self._feature_group_id = feature_group_id

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
