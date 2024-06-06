#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
from .recommender_dataset import RecommenderDatasets
from ..operator_config import RecommenderOperatorConfig
from .factory import RecommenderOperatorBaseModel
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy


class SVDOperatorModel(RecommenderOperatorBaseModel):
    """Class representing scikit surprise SVD operator model."""

    def __init__(self, config: RecommenderOperatorConfig, datasets: RecommenderDatasets):
        super().__init__(config, datasets)
        self.interactions = datasets.interactions
        self.users = datasets.users
        self.items = datasets.items
        self.user_id = config.spec.user_column_name
        self.item_id = config.spec.item_column_name
        self.rating_col = config.spec.ratings_column_name
        self.test_size = 0.2

    def _get_recommendations(self, user_id, algo, items, n=10):
        all_item_ids = items[self.item_id].unique()
        rated_items = self.interactions[self.interactions[self.user_id] == user_id][self.item_id]
        unrated_items = [item_id for item_id in all_item_ids if item_id not in rated_items.values]
        predictions = [algo.predict(user_id, item_id) for item_id in unrated_items]
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_n_recommendations = predictions[:n]
        return [(pred.iid, pred.est) for pred in top_n_recommendations]

    def _build_model(self) -> pd.DataFrame:
        min_rating = self.interactions[self.rating_col].min()
        max_rating = self.interactions[self.rating_col].max()
        reader = Reader(rating_scale=(min_rating, max_rating))
        data = Dataset.load_from_df(self.interactions[[self.user_id, self.item_id, self.rating_col]], reader)
        trainset, testset = train_test_split(data, test_size=self.test_size)
        algo = SVD()
        algo.fit(trainset)
        predictions = algo.test(testset)
        accuracy.rmse(predictions)
        all_recommendations = []
        for user_id in self.users[self.user_id]:
            recommendations = self._get_recommendations(user_id, algo, self.items, n=self.spec.top_k)
            for item_id, est_rating in recommendations:
                all_recommendations.append({
                    self.user_id: user_id,
                    self.item_id: item_id,
                    self.rating_col: est_rating
                })
        recommendations_df = pd.DataFrame(all_recommendations)
        return recommendations_df
