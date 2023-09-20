#!/usr/bin/env python
# -*- coding: utf-8; -*-
# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List
from ads.feature_store.statistics.feature_stat import FeatureStatistics
from ads.feature_store.response.response_builder import ResponseBuilder
import json


class Statistics(ResponseBuilder):
    """
    Represents statistical information.
    """

    @property
    def kind(self) -> str:
        """
        Gets the kind of the statistics object.

        Returns
        -------
        str
            The kind of the statistics object, which is always "statistics".
        """
        return "statistics"

    def to_viz(self, feature_list: List[str] = None):
        if self.content is not None:
            stats: dict = json.loads(self.content)
            [
                FeatureStatistics.from_json(feature, stat).to_viz()
                for feature, stat in stats.items()
                if (feature_list is None or feature in feature_list)
            ]
