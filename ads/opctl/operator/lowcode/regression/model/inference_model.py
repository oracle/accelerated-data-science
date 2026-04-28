#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class RegressionInferenceModel:
    """Lightweight inference wrapper around the trained preprocessor and regressor."""

    def __init__(self, preprocessor, regressor):
        self.preprocessor = preprocessor
        self.regressor = regressor

    def predict(self, X):
        features = self.preprocessor.preprocess_for_prediction(X)
        return self.regressor.predict(features)
