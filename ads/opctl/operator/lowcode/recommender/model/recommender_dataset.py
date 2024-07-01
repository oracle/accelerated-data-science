#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd

from ads.opctl.operator.lowcode.common.utils import load_data
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
        self.interactions: pd.DataFrame = load_data(getattr(spec, "interactions_data"))
        self.users: pd.DataFrame = load_data(getattr(spec, "user_data"))
        self.items: pd.DataFrame = load_data(getattr(spec, "item_data"))
