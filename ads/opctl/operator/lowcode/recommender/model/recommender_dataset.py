#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd

from ..operator_config import RecommenderOperatorConfig


class RecommenderDatasets:
    def __init__(self, config: RecommenderOperatorConfig):
        """Instantiates the DataIO instance.

        Properties
        ----------
        spec: RecommenderOperatorSpec
            The recommender operator spec.
        """
        spec = config.spec
        self.interactions: pd.DataFrame = pd.read_csv(spec.interactions_data.url)
        self.users: pd.DataFrame = pd.read_csv(spec.user_data.url)
        self.items: pd.DataFrame = pd.read_csv(spec.item_data.url)