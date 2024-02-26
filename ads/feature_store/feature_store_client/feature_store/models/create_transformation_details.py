# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class CreateTransformationDetails(object):
    """
    Parameters needed to create a new transformation.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new CreateTransformationDetails object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param name:
            The value to assign to the name property of this CreateTransformationDetails.
        :type name: str

        :param compartment_id:
            The value to assign to the compartment_id property of this CreateTransformationDetails.
        :type compartment_id: str

        :param feature_store_id:
            The value to assign to the feature_store_id property of this CreateTransformationDetails.
        :type feature_store_id: str

        :param description:
            The value to assign to the description property of this CreateTransformationDetails.
        :type description: str

        :param source_code:
            The value to assign to the source_code property of this CreateTransformationDetails.
        :type source_code: str

        :param transformation_mode:
            The value to assign to the transformation_mode property of this CreateTransformationDetails.
        :type transformation_mode: str

        """
        self.swagger_types = {
            'name': 'str',
            'compartment_id': 'str',
            'feature_store_id': 'str',
            'description': 'str',
            'source_code': 'str',
            'transformation_mode': 'str'
        }

        self.attribute_map = {
            'name': 'name',
            'compartment_id': 'compartmentId',
            'feature_store_id': 'featureStoreId',
            'description': 'description',
            'source_code': 'sourceCode',
            'transformation_mode': 'transformationMode'
        }

        self._name = None
        self._compartment_id = None
        self._feature_store_id = None
        self._description = None
        self._source_code = None
        self._transformation_mode = None

    @property
    def name(self):
        """
        **[Required]** Gets the name of this CreateTransformationDetails.
        A user-friendly display name for the resource. It does not have to be unique and can be modified. Avoid entering confidential information.


        :return: The name of this CreateTransformationDetails.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Sets the name of this CreateTransformationDetails.
        A user-friendly display name for the resource. It does not have to be unique and can be modified. Avoid entering confidential information.


        :param name: The name of this CreateTransformationDetails.
        :type: str
        """
        self._name = name

    @property
    def compartment_id(self):
        """
        **[Required]** Gets the compartment_id of this CreateTransformationDetails.
        The OCID for the data asset's compartment.


        :return: The compartment_id of this CreateTransformationDetails.
        :rtype: str
        """
        return self._compartment_id

    @compartment_id.setter
    def compartment_id(self, compartment_id):
        """
        Sets the compartment_id of this CreateTransformationDetails.
        The OCID for the data asset's compartment.


        :param compartment_id: The compartment_id of this CreateTransformationDetails.
        :type: str
        """
        self._compartment_id = compartment_id

    @property
    def feature_store_id(self):
        """
        **[Required]** Gets the feature_store_id of this CreateTransformationDetails.
        The OCID of feature store


        :return: The feature_store_id of this CreateTransformationDetails.
        :rtype: str
        """
        return self._feature_store_id

    @feature_store_id.setter
    def feature_store_id(self, feature_store_id):
        """
        Sets the feature_store_id of this CreateTransformationDetails.
        The OCID of feature store


        :param feature_store_id: The feature_store_id of this CreateTransformationDetails.
        :type: str
        """
        self._feature_store_id = feature_store_id

    @property
    def description(self):
        """
        Gets the description of this CreateTransformationDetails.
        A short description of the Ai data asset


        :return: The description of this CreateTransformationDetails.
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description):
        """
        Sets the description of this CreateTransformationDetails.
        A short description of the Ai data asset


        :param description: The description of this CreateTransformationDetails.
        :type: str
        """
        self._description = description

    @property
    def source_code(self):
        """
        **[Required]** Gets the source_code of this CreateTransformationDetails.
        Source code for the transformation


        :return: The source_code of this CreateTransformationDetails.
        :rtype: str
        """
        return self._source_code

    @source_code.setter
    def source_code(self, source_code):
        """
        Sets the source_code of this CreateTransformationDetails.
        Source code for the transformation


        :param source_code: The source_code of this CreateTransformationDetails.
        :type: str
        """
        self._source_code = source_code

    @property
    def transformation_mode(self):
        """
        **[Required]** Gets the transformation_mode of this CreateTransformationDetails.
        Mode of the transformation


        :return: The transformation_mode of this CreateTransformationDetails.
        :rtype: str
        """
        return self._transformation_mode

    @transformation_mode.setter
    def transformation_mode(self, transformation_mode):
        """
        Sets the transformation_mode of this CreateTransformationDetails.
        Mode of the transformation


        :param transformation_mode: The transformation_mode of this CreateTransformationDetails.
        :type: str
        """
        self._transformation_mode = transformation_mode

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
