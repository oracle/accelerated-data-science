#!/usr/bin/env python
# -*- coding: utf-8 -*--
from typing import Tuple, Dict, Any

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
from pandas import DataFrame

from .recommender_dataset import RecommenderDatasets
from ..operator_config import RecommenderOperatorConfig
from .factory import RecommenderOperatorBaseModel
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise.accuracy import rmse, mae
import report_creator as rc
from ..constant import SupportedMetrics


class SVDOperatorModel(RecommenderOperatorBaseModel):
    """Class representing scikit surprise SVD operator model."""

    def __init__(self, config: RecommenderOperatorConfig, datasets: RecommenderDatasets):
        super().__init__(config, datasets)
        self.interactions = datasets.interactions
        self.users = datasets.users
        self.items = datasets.items
        self.user_id = config.spec.user_column
        self.item_id = config.spec.item_column
        self.interaction_column = config.spec.interaction_column
        self.test_size = 0.2
        self.algo = SVD()

    def _get_recommendations(self, user_id, n):
        all_item_ids = self.items[self.item_id].unique()
        rated_items = self.interactions[self.interactions[self.user_id] == user_id][self.item_id]
        unrated_items = [item_id for item_id in all_item_ids if item_id not in rated_items.values]
        predictions = [self.algo.predict(user_id, item_id) for item_id in unrated_items]
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_n_recommendations = predictions[:n]
        return [(pred.iid, pred.est) for pred in top_n_recommendations]

    def _build_model(self) -> Tuple[DataFrame, Dict]:
        min_rating = self.interactions[self.interaction_column].min()
        max_rating = self.interactions[self.interaction_column].max()
        reader = Reader(rating_scale=(min_rating, max_rating))
        data = Dataset.load_from_df(self.interactions[[self.user_id, self.item_id, self.interaction_column]], reader)
        trainset, testset = train_test_split(data, test_size=self.test_size)
        self.algo.fit(trainset)
        predictions = self.algo.test(testset)

        metric = {}
        metric[SupportedMetrics.RMSE] = rmse(predictions, verbose=True)
        metric[SupportedMetrics.MAE] = mae(predictions, verbose=True)
        all_recommendations = []
        for user_id in self.users[self.user_id]:
            recommendations = self._get_recommendations(user_id, n=self.spec.top_k)
            for item_id, est_rating in recommendations:
                all_recommendations.append({
                    self.user_id: user_id,
                    self.item_id: item_id,
                    self.interaction_column: est_rating
                })
        recommendations_df = pd.DataFrame(all_recommendations)
        return recommendations_df, metric

    def _generate_report(self):
        model_description = """
            Singular Value Decomposition (SVD) is a matrix factorization technique used in recommendation systems to
            decompose a user-item interaction matrix into three constituent matrices. These matrices capture the
            latent factors that explain the observed interactions.
            """
        new_user_recommendations = self._get_recommendations("__new_user__", self.spec.top_k)
        new_recommendations = []
        for item_id, est_rating in new_user_recommendations:
            new_recommendations.append({
                self.user_id: "__new_user__",
                self.item_id: item_id,
                self.interaction_column: est_rating
            })
        title = rc.Heading("Recommendations for new users", level=2)
        other_sections = [title, rc.DataTable(new_recommendations)]
        return (
            model_description,
            other_sections
        )
