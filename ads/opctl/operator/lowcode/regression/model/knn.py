#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from .shared_model import SharedRegressionOperatorModel


class KNNRegressionOperatorModel(SharedRegressionOperatorModel):
    def _build_estimator(self):
        params = {
            "n_neighbors": 5,
            "weights": "uniform",
            "algorithm": "auto",
            "p": 2,
        }
        params.update(self.spec.model.params or {})
        params.update(self.spec.model_kwargs or {})
        return KNeighborsRegressor(**params)

    def _train_and_predict(self, x_train, y_train):
        preprocessor = self._build_preprocessor(x_train)
        estimator = self._build_estimator()

        self.pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", estimator)]
        )
        self.pipeline.fit(x_train, y_train)
        self._derive_feature_names()

        yhat_train = self.pipeline.predict(x_train)
        self.train_predictions = pd.DataFrame(
            {
                "actual": y_train.to_numpy(),
                "prediction": yhat_train,
                "residual": y_train.to_numpy() - yhat_train,
            }
        )

        train_metric_dict = self._compute_metrics(y_train, yhat_train)
        self.train_metrics = pd.DataFrame(
            [{"metric": k, "value": v} for k, v in train_metric_dict.items()]
        )

        if (
            self.datasets.test_data is not None
            and self.target_column in self.datasets.test_data.columns
        ):
            x_test = self.datasets.test_data[self.feature_columns]
            y_test = self.datasets.test_data[self.target_column]
            yhat_test = self.pipeline.predict(x_test)
            self.test_predictions = pd.DataFrame(
                {
                    "actual": y_test.to_numpy(),
                    "prediction": yhat_test,
                    "residual": y_test.to_numpy() - yhat_test,
                }
            )
            test_metric_dict = self._compute_metrics(y_test, yhat_test)
            self.test_metrics = pd.DataFrame(
                [{"metric": k, "value": v} for k, v in test_metric_dict.items()]
            )

        return self._compute_feature_importance(x_train, y_train)
