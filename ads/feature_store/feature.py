#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from copy import deepcopy
from typing import Dict

from ads.feature_store.common.enums import FilterOperators
from ads.jobs.builders.base import Builder
from ads.feature_store.query import filter


class Feature(Builder):
    """
    A class that represents a feature and allows comparison with other features using various operators.
    The Feature class  has six comparison methods (__lt__, __le__, __eq__, __ne__, __ge__, and __gt__) that return instances
    of a Filter class. These comparison methods allow instances of the Feature class to be compared using the
    corresponding comparison operators.

    """

    CONST_FEATURE_NAME = "name"
    CONST_FEATURE_TYPE = "featureType"
    CONST_FEATURE_GROUP_ID = "featureGroupId"

    def __init__(self, name, featureType, featureGroupId):
        super().__init__()

        self.with_feature_name(name)
        self.with_feature_type(featureType)
        self.with_feature_group_id(featureGroupId)

    @property
    def feature_name(self):
        return self.get_spec(self.CONST_FEATURE_NAME)

    def with_feature_name(self, name: str):
        """
        Sets the name attribute of the feature.

        Args:
            name (str): The new name for the feature.

        Returns:
            Feature: This instance of the Feature class.
        """
        return self.set_spec(self.CONST_FEATURE_NAME, name)

    @property
    def feature_type(self):
        return self.get_spec(self.CONST_FEATURE_TYPE)

    def with_feature_type(self, feature_type: str):
        """
        Sets the type attribute of the feature.

        Args:
            feature_type (str): The new type for the feature.

        Returns:
            Feature: This instance of the Feature class.
        """
        return self.set_spec(self.CONST_FEATURE_TYPE, feature_type)

    @property
    def feature_group_id(self):
        return self.get_spec(self.CONST_FEATURE_GROUP_ID)

    def with_feature_group_id(self, feature_group_id):
        """
        Sets the group attribute of the feature.

        Args:
           feature_group_id: FeatureGroup id which contains the feature.

        Returns:
           Feature: This instance of the Feature class.
        """
        return self.set_spec(self.CONST_FEATURE_GROUP_ID, feature_group_id)

    def __lt__(self, other):
        return filter.Filter(self, FilterOperators.LT.value, other)

    def __le__(self, other):
        return filter.Filter(self, FilterOperators.LE.value, other)

    def __eq__(self, other):
        return filter.Filter(self, FilterOperators.EQ.value, other)

    def __ne__(self, other):
        return filter.Filter(self, FilterOperators.NE.value, other)

    def __ge__(self, other):
        return filter.Filter(self, FilterOperators.GE.value, other)

    def __gt__(self, other):
        return filter.Filter(self, FilterOperators.GT.value, other)

    def to_dict(self) -> Dict:
        """Serializes feature   to a dictionary.

        Returns
        -------
        dict
            The feature serialized as a dictionary.
        """

        spec = deepcopy(self._spec)
        return spec


class DatasetFeature(Builder):
    """
    A class that represents a feature and allows comparison with other features using various operators.
    The Feature class  has six comparison methods (__lt__, __le__, __eq__, __ne__, __ge__, and __gt__) that return instances
    of a Filter class. These comparison methods allow instances of the Feature class to be compared using the
    corresponding comparison operators.

    """

    CONST_FEATURE_NAME = "name"
    CONST_FEATURE_TYPE = "featureType"
    CONST_DATASET_ID = "datasetId"

    def __init__(self, name, featureType, datasetId):
        super().__init__()

        self.with_feature_name(name)
        self.with_feature_type(featureType)
        self.with_dataset_id(datasetId)

    @property
    def feature_name(self):
        return self.get_spec(self.CONST_FEATURE_NAME)

    def with_feature_name(self, name: str):
        """
        Sets the name attribute of the feature.

        Args:
            name (str): The new name for the feature.

        Returns:
            Feature: This instance of the Feature class.
        """
        return self.set_spec(self.CONST_FEATURE_NAME, name)

    @property
    def feature_type(self):
        return self.get_spec(self.CONST_FEATURE_TYPE)

    def with_feature_type(self, feature_type: str):
        """
        Sets the type attribute of the feature.

        Args:
            feature_type (str): The new type for the feature.

        Returns:
            Feature: This instance of the Feature class.
        """
        return self.set_spec(self.CONST_FEATURE_TYPE, feature_type)

    @property
    def dataset_id(self):
        return self.get_spec(self.CONST_DATASET_ID)

    def with_dataset_id(self, dataset_id):
        """
        Sets the group attribute of the feature.

        Args:
           dataset_id: Dataset id which contains the feature.

        Returns:
           Feature: This instance of the Feature class.
        """
        return self.set_spec(self.CONST_DATASET_ID, dataset_id)

    def to_dict(self) -> Dict:
        """Serializes feature   to a dictionary.

        Returns
        -------
        dict
            The feature serialized as a dictionary.
        """

        spec = deepcopy(self._spec)
        return spec
