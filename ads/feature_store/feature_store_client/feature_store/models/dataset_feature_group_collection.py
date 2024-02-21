# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class DatasetFeatureGroupCollection(object):
    """
    Results of a dataset feature group collection
    """

    def __init__(self, **kwargs):
        """
        Initializes a new DatasetFeatureGroupCollection object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param is_manual_association:
            The value to assign to the is_manual_association property of this DatasetFeatureGroupCollection.
        :type is_manual_association: bool

        :param items:
            The value to assign to the items property of this DatasetFeatureGroupCollection.
        :type items: list[oci.feature_store.models.DatasetFeatureGroupSummary]

        """
        self.swagger_types = {
            'is_manual_association': 'bool',
            'items': 'list[DatasetFeatureGroupSummary]'
        }

        self.attribute_map = {
            'is_manual_association': 'isManualAssociation',
            'items': 'items'
        }

        self._is_manual_association = None
        self._items = None

    @property
    def is_manual_association(self):
        """
        **[Required]** Gets the is_manual_association of this DatasetFeatureGroupCollection.
        Boolean value indicating whether the collection was manually associated


        :return: The is_manual_association of this DatasetFeatureGroupCollection.
        :rtype: bool
        """
        return self._is_manual_association

    @is_manual_association.setter
    def is_manual_association(self, is_manual_association):
        """
        Sets the is_manual_association of this DatasetFeatureGroupCollection.
        Boolean value indicating whether the collection was manually associated


        :param is_manual_association: The is_manual_association of this DatasetFeatureGroupCollection.
        :type: bool
        """
        self._is_manual_association = is_manual_association

    @property
    def items(self):
        """
        **[Required]** Gets the items of this DatasetFeatureGroupCollection.
        List of dataset feature group collection


        :return: The items of this DatasetFeatureGroupCollection.
        :rtype: list[oci.feature_store.models.DatasetFeatureGroupSummary]
        """
        return self._items

    @items.setter
    def items(self, items):
        """
        Sets the items of this DatasetFeatureGroupCollection.
        List of dataset feature group collection


        :param items: The items of this DatasetFeatureGroupCollection.
        :type: list[oci.feature_store.models.DatasetFeatureGroupSummary]
        """
        self._items = items

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
