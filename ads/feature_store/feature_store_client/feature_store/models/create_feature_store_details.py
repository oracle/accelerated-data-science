# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class CreateFeatureStoreDetails(object):
    """
    The information about new FeatureStore.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new CreateFeatureStoreDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param name:
            The value to assign to the name property of this CreateFeatureStoreDetails.
        :type name: str

        :param compartment_id:
            The value to assign to the compartment_id property of this CreateFeatureStoreDetails.
        :type compartment_id: str

        :param offline_config:
            The value to assign to the offline_config property of this CreateFeatureStoreDetails.
        :type offline_config: oci.feature_store.models.OfflineConfig

        :param online_config:
            The value to assign to the online_config property of this CreateFeatureStoreDetails.
        :type online_config: oci.feature_store.models.OnlineConfig

        :param description:
            The value to assign to the description property of this CreateFeatureStoreDetails.
        :type description: str

        :param freeform_tags:
            The value to assign to the freeform_tags property of this CreateFeatureStoreDetails.
        :type freeform_tags: dict(str, str)

        :param defined_tags:
            The value to assign to the defined_tags property of this CreateFeatureStoreDetails.
        :type defined_tags: dict(str, dict(str, object))

        """
        self.swagger_types = {
            'name': 'str',
            'compartment_id': 'str',
            'offline_config': 'OfflineConfig',
            'online_config': 'OnlineConfig',
            'description': 'str',
            'freeform_tags': 'dict(str, str)',
            'defined_tags': 'dict(str, dict(str, object))'
        }

        self.attribute_map = {
            'name': 'name',
            'compartment_id': 'compartmentId',
            'offline_config': 'offlineConfig',
            'online_config': 'onlineConfig',
            'description': 'description',
            'freeform_tags': 'freeformTags',
            'defined_tags': 'definedTags'
        }

        self._name = None
        self._compartment_id = None
        self._offline_config = None
        self._online_config = None
        self._description = None
        self._freeform_tags = None
        self._defined_tags = None

    @property
    def name(self):
        """
        **[Required]** Gets the name of this CreateFeatureStoreDetails.
        FeatureStore Identifier


        :return: The name of this CreateFeatureStoreDetails.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Sets the name of this CreateFeatureStoreDetails.
        FeatureStore Identifier


        :param name: The name of this CreateFeatureStoreDetails.
        :type: str
        """
        self._name = name

    @property
    def compartment_id(self):
        """
        **[Required]** Gets the compartment_id of this CreateFeatureStoreDetails.
        Compartment Identifier


        :return: The compartment_id of this CreateFeatureStoreDetails.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this CreateFeatureStoreDetails.
        Compartment Identifier


        :param compartment_id: The compartment_id of this CreateFeatureStoreDetails.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def offline_config(self):
        """
        Gets the offline_config of this CreateFeatureStoreDetails.

        :return: The offline_config of this CreateFeatureStoreDetails.
        :rtype: oci.feature_store.models.OfflineConfig
        """
        return self._offline_config

    @offline_config.setter
    def offline_config(self, offline_config):
        """
        Sets the offline_config of this CreateFeatureStoreDetails.

        :param offline_config: The offline_config of this CreateFeatureStoreDetails.
        :type: oci.feature_store.models.OfflineConfig
        """
        self._offline_config = offline_config

    @property
    def online_config(self):
        """
        Gets the online_config of this CreateFeatureStoreDetails.

        :return: The online_config of this CreateFeatureStoreDetails.
        :rtype: oci.feature_store.models.OnlineConfig
        """
        return self._online_config

    @online_config.setter
    def online_config(self, online_config):
        """
        Sets the online_config of this CreateFeatureStoreDetails.

        :param online_config: The online_config of this CreateFeatureStoreDetails.
        :type: oci.feature_store.models.OnlineConfig
        """
        self._online_config = online_config

    @property
    def description(self):
        """
        Gets the description of this CreateFeatureStoreDetails.
        A short description of the feature store


        :return: The description of this CreateFeatureStoreDetails.
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """
        Sets the description of this CreateFeatureStoreDetails.
        A short description of the feature store


        :param description: The description of this CreateFeatureStoreDetails.
        :type: str
        """
        self._description = description

    @property
    def freeform_tags(self):
        """
        Gets the freeform_tags of this CreateFeatureStoreDetails.
        Simple key-value pair that is applied without any predefined name, type or scope. Exists for cross-compatibility only.
        Example: `{\"bar-key\": \"value\"}`


        :return: The freeform_tags of this CreateFeatureStoreDetails.
        :rtype: dict(str, str)
        """
        return self._freeform_tags

    @freeform_tags.setter
    def freeform_tags(self, freeform_tags):
        """
        Sets the freeform_tags of this CreateFeatureStoreDetails.
        Simple key-value pair that is applied without any predefined name, type or scope. Exists for cross-compatibility only.
        Example: `{\"bar-key\": \"value\"}`


        :param freeform_tags: The freeform_tags of this CreateFeatureStoreDetails.
        :type: dict(str, str)
        """
        self._freeform_tags = freeform_tags

    @property
    def defined_tags(self):
        """
        Gets the defined_tags of this CreateFeatureStoreDetails.
        Defined tags for this resource. Each key is predefined and scoped to a namespace.
        Example: `{\"foo-namespace\": {\"bar-key\": \"value\"}}`


        :return: The defined_tags of this CreateFeatureStoreDetails.
        :rtype: dict(str, dict(str, object))
        """
        return self._defined_tags

    @defined_tags.setter
    def defined_tags(self, defined_tags):
        """
        Sets the defined_tags of this CreateFeatureStoreDetails.
        Defined tags for this resource. Each key is predefined and scoped to a namespace.
        Example: `{\"foo-namespace\": {\"bar-key\": \"value\"}}`


        :param defined_tags: The defined_tags of this CreateFeatureStoreDetails.
        :type: dict(str, dict(str, object))
        """
        self._defined_tags = defined_tags

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
