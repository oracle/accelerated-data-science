#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import abstractmethod

from ads.common.decorator.deprecate import deprecated


class Explainer:
    """Base class for `GlobalExplainer` and `LocalExplainer`"""

    @deprecated(
        details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
        raise_error=True,
    )
    def __init__(self):
        self.X_train_ = None
        self.y_train_ = None
        self.X_test_ = None
        self.y_test_ = None
        self.est_ = None
        self.mode_ = None
        self.class_names_ = None

    def setup(self, model, X_test, y_test, X_train=None, y_train=None):
        """
        Sets up required attributes for global explainer.

        Parameters
        ----------
        model : instance of `ADSModel`
            The model being explained
        X_test : pandas.DataFrame
            Test data to explain.
        y_test : pandas.Series
            Labels for test data.
        X_train : pandas.DataFrame, optional
            Training data to build explainer. Defaults to `None`.
        y_train : pandas.Series, optional
            Labels for training data. Defaults to `None`.
        """
        # assert isinstance(model, ADSModel)
        self.X_train_ = X_train if X_train is not None else X_test
        self.y_train_ = y_train if y_train is not None else y_test
        self.X_test_ = X_test
        self.y_test_ = y_test
        self.est_ = model
        if hasattr(model, "classes_") and model.classes_ is not None:
            self.mode_ = "classification"
            self.class_names_ = model.classes_
        else:
            self.mode_ = "regression"

    @property
    def X_train(self):
        """
        Training data.
        """
        return self.X_train_

    @property
    def y_train(self):
        """
        Labels for training data.
        """
        return self.y_train_

    @property
    def X_test(self):
        """
        Test data.
        """
        return self.X_test_

    @property
    def y_test(self):
        """
        Labels for test data.
        """
        return self.y_test_

    @property
    def mode(self):
        """
        Mode of explanation.
        Either `classification` or `regression`.
        """
        return self.mode_

    @property
    def est(self):
        """
        Model Estimator.
        """
        return self.est_

    @property
    def class_names(self):
        """
        Class names.
        Returns `None` for regression.
        """
        return self.class_names_


class GlobalExplainer(Explainer):
    """Abstract `GlobalExplainer` class. Must be subclassed to create instances."""

    @abstractmethod
    def compute_feature_importance(self, **kwargs):
        pass

    @abstractmethod
    def compute_partial_dependence(self, **kwargs):
        pass

    @abstractmethod
    def show_in_notebook(self):
        pass


class LocalExplainer(Explainer):
    """Abstract `LocalExplainer` class. Must be subclassed to create instances."""

    @abstractmethod
    def explain(self, **kwargs):
        pass


class WhatIfExplainer(Explainer):
    """Abstract `WhatIfExplainer` class. Must be subclassed to create instances."""

    @abstractmethod
    def explore_sample(self, **kwargs):
        pass

    @abstractmethod
    def explore_predictions(self, **kwargs):
        pass
