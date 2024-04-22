#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import copy

from ads.jobs.builders.base import Builder


class FeatureOptionDetails(Builder):
    CONST_FEATURE_OPTION_WRITE_CONFIG_DETAILS = "featureOptionWriteConfigDetails"
    CONST_MERGE_SCHEMA = "mergeSchema"
    CONST_OVERWRITE_SCHEMA = "overwriteSchema"

    def __init__(self):
        super().__init__()

    def with_feature_option_write_config_details(
        self, merge_schema: bool = False, overwrite_schema: bool = False
    ) -> "FeatureOptionDetails":
        """Sets the feature option write configuration details.

        Parameters
        ----------
        merge_schema: bool
            The merge_schema.
        overwrite_schema: bool
            The overwrite_schema.

        Returns
        -------
        FeatureOptionDetails
            The FeatureOptionDetails instance (self)

        """
        return self.set_spec(
            self.CONST_FEATURE_OPTION_WRITE_CONFIG_DETAILS,
            {
                self.CONST_MERGE_SCHEMA: merge_schema,
                self.CONST_OVERWRITE_SCHEMA: overwrite_schema,
            },
        )

    def to_dict(self):
        """Returns the FeatureOptionDetails as dictionary."""

        feature_option_details = copy.deepcopy(self._spec)
        return feature_option_details
