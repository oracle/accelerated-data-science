#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.dataset.dataset_with_target import ADSDatasetWithTarget
from ads.type_discovery.typed_feature import DateTimeTypedFeature


class ForecastingDataset(ADSDatasetWithTarget):
    def __init__(self, df, sampled_df, target, target_type, shape, **kwargs):
        # index on target
        if isinstance(target, DateTimeTypedFeature):
            df = df.set_index(target)
        ADSDatasetWithTarget.__init__(
            self, df=df, sampled_df=sampled_df, target=target, target_type=target_type, shape=shape, **kwargs
        )

    def select_best_features(self, score_func=None, k=12):
        """
        Not yet implemented
        """
        raise NotImplementedError(
            "Feature selection for forecasting dataset is not yet supported"
        )
