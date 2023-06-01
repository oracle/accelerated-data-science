#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy

from ads.feature_store.common.enums import FeatureType
from ads.jobs.builders.base import Builder


class FeatureDetail(Builder):
    """Represents input Feature Schema.

    Methods
    -------
    with_feature_type(self, feature_type: FeatureType) -> "FeatureDetail"
        Sets the feature_type.
    with_order_number(self, order_number: int) -> "FeatureDetail"
        Sets the order_number.
    with_event_timestamp_format(self, event_timestamp_format: str) -> "FeatureDetail"
        Sets the timestamp format for the feature.
    with_is_event_timestamp_format(self, is_event_timestamp_format: bool) -> "FeatureDetail"
        Sets the is_event_timestamp_format.

    """

    CONST_NAME = "name"
    CONST_FEATURE_TYPE = "featureType"
    CONST_ORDER_NUMBER = "orderNumber"
    CONST_IS_EVENT_TIMESTAMP = "isEventTimestamp"
    CONST_EVENT_TIMESTAMP_FORMAT = "eventTimestampFormat"

    def __init__(
        self,
        name: str,
        feature_type: FeatureType = None,
        order_number: int = None,
        is_event_timestamp: bool = False,
        event_timestamp_format: str = None,
    ):
        super().__init__()

        if not name:
            raise ValueError("Feature name must be specified.")

        self.set_spec(self.CONST_NAME, name)

        if feature_type:
            self.with_feature_type(feature_type)
        if order_number:
            self.with_order_number(order_number)
        if is_event_timestamp:
            self.with_is_event_timestamp(is_event_timestamp)
        if event_timestamp_format:
            self.with_event_timestamp_format(event_timestamp_format)

    @property
    def feature_name(self):
        return self.get_spec(self.CONST_NAME)

    @property
    def feature_type(self):
        return self.get_spec(self.CONST_FEATURE_TYPE)

    def with_feature_type(self, feature_type: FeatureType) -> "FeatureDetail":
        """Sets the feature_type.

        Parameters
        ----------
        feature_type: FeatureType
            The feature_type of the Feature.

        Returns
        -------
        FeatureDetails
            The FeatureDetails instance (self)
        """

        return self.set_spec(self.CONST_FEATURE_TYPE, feature_type.value)

    @property
    def order_number(self):
        return self.get_spec(self.CONST_ORDER_NUMBER)

    def with_order_number(self, order_number: int) -> "FeatureDetail":
        """Sets the order number.

        Parameters
        ----------
        order_number: int
            The order_number of the Feature.

        Returns
        -------
        FeatureDetail
            The FeatureDetail instance (self)
        """

        return self.set_spec(self.CONST_ORDER_NUMBER, order_number)

    @property
    def event_timestamp_format(self):
        return self.get_spec(self.CONST_EVENT_TIMESTAMP_FORMAT)

    def with_event_timestamp_format(
        self, event_timestamp_format: str
    ) -> "FeatureDetail":
        """Sets the event_timestamp_format.

        Parameters
        ----------
        event_timestamp_format: str
            The event_timestamp_format of the Feature.

        Returns
        -------
        FeatureDetail
            The FeatureDetail instance (self)
        """

        return self.set_spec(self.CONST_EVENT_TIMESTAMP_FORMAT, event_timestamp_format)

    @property
    def is_event_timestamp(self):
        return self.get_spec(self.CONST_IS_EVENT_TIMESTAMP)

    def with_is_event_timestamp(self, is_event_timestamp: bool) -> "FeatureDetail":
        """Sets the is_event_timestamp.

        Parameters
        ----------
        is_event_timestamp: bool
            The is_event_timestamp of the Feature.

        Returns
        -------
        FeatureDetail
            The FeatureDetail instance (self)
        """

        return self.set_spec(self.CONST_IS_EVENT_TIMESTAMP, is_event_timestamp)

    def to_dict(self):
        """Returns the FeatureDetail as dictionary."""

        feature_detail = copy.deepcopy(self._spec)
        return feature_detail
