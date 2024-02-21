# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.


from oci.util import formatted_flat_dict, NONE_SENTINEL, value_allowed_none_or_none_sentinel  # noqa: F401
from oci.decorators import init_model_state_from_kwargs


@init_model_state_from_kwargs
class LineageSummary(object):
    """
    Response of lineage query.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new LineageSummary object with values from keyword arguments.
        The following keyword arguments are supported (corresponding to the getters/setters of this class):

        :param feature_store:
            The value to assign to the feature_store property of this LineageSummary.
        :type feature_store: oci.feature_store.models.LineageDetail

        :param entity:
            The value to assign to the entity property of this LineageSummary.
        :type entity: oci.feature_store.models.LineageDetail

        :param transformation:
            The value to assign to the transformation property of this LineageSummary.
        :type transformation: oci.feature_store.models.LineageDetail

        :param feature_group:
            The value to assign to the feature_group property of this LineageSummary.
        :type feature_group: oci.feature_store.models.LineageDetail

        :param dataset:
            The value to assign to the dataset property of this LineageSummary.
        :type dataset: oci.feature_store.models.LineageDetail

        :param model_details:
            The value to assign to the model_details property of this LineageSummary.
        :type model_details: oci.feature_store.models.ModelCollection

        """
        self.swagger_types = {
            'feature_store': 'LineageDetail',
            'entity': 'LineageDetail',
            'transformation': 'LineageDetail',
            'feature_group': 'LineageDetail',
            'dataset': 'LineageDetail',
            'model_details': 'ModelCollection'
        }

        self.attribute_map = {
            'feature_store': 'featureStore',
            'entity': 'entity',
            'transformation': 'transformation',
            'feature_group': 'featureGroup',
            'dataset': 'dataset',
            'model_details': 'modelDetails'
        }

        self._feature_store = None
        self._entity = None
        self._transformation = None
        self._feature_group = None
        self._dataset = None
        self._model_details = None

    @property
    def feature_store(self):
        """
        **[Required]** Gets the feature_store of this LineageSummary.

        :return: The feature_store of this LineageSummary.
        :rtype: oci.feature_store.models.LineageDetail
        """
        return self._feature_store

    @feature_store.setter
    def feature_store(self, feature_store):
        """
        Sets the feature_store of this LineageSummary.

        :param feature_store: The feature_store of this LineageSummary.
        :type: oci.feature_store.models.LineageDetail
        """
        self._feature_store = feature_store

    @property
    def entity(self):
        """
        **[Required]** Gets the entity of this LineageSummary.

        :return: The entity of this LineageSummary.
        :rtype: oci.feature_store.models.LineageDetail
        """
        return self._entity

    @entity.setter
    def entity(self, entity):
        """
        Sets the entity of this LineageSummary.

        :param entity: The entity of this LineageSummary.
        :type: oci.feature_store.models.LineageDetail
        """
        self._entity = entity

    @property
    def transformation(self):
        """
        Gets the transformation of this LineageSummary.

        :return: The transformation of this LineageSummary.
        :rtype: oci.feature_store.models.LineageDetail
        """
        return self._transformation

    @transformation.setter
    def transformation(self, transformation):
        """
        Sets the transformation of this LineageSummary.

        :param transformation: The transformation of this LineageSummary.
        :type: oci.feature_store.models.LineageDetail
        """
        self._transformation = transformation

    @property
    def feature_group(self):
        """
        **[Required]** Gets the feature_group of this LineageSummary.

        :return: The feature_group of this LineageSummary.
        :rtype: oci.feature_store.models.LineageDetail
        """
        return self._feature_group

    @feature_group.setter
    def feature_group(self, feature_group):
        """
        Sets the feature_group of this LineageSummary.

        :param feature_group: The feature_group of this LineageSummary.
        :type: oci.feature_store.models.LineageDetail
        """
        self._feature_group = feature_group

    @property
    def dataset(self):
        """
        Gets the dataset of this LineageSummary.

        :return: The dataset of this LineageSummary.
        :rtype: oci.feature_store.models.LineageDetail
        """
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        """
        Sets the dataset of this LineageSummary.

        :param dataset: The dataset of this LineageSummary.
        :type: oci.feature_store.models.LineageDetail
        """
        self._dataset = dataset

    @property
    def model_details(self):
        """
        Gets the model_details of this LineageSummary.

        :return: The model_details of this LineageSummary.
        :rtype: oci.feature_store.models.ModelCollection
        """
        return self._model_details

    @model_details.setter
    def model_details(self, model_details):
        """
        Sets the model_details of this LineageSummary.

        :param model_details: The model_details of this LineageSummary.
        :type: oci.feature_store.models.ModelCollection
        """
        self._model_details = model_details

    def __repr__(self):
        return formatted_flat_dict(self)

    def __eq__(self, other):
        if other is None:
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other
