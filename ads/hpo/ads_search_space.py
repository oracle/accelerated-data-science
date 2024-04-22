#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import ABC

from ads.hpo.distributions import *
from ads.hpo.utils import _update_space_name

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class ModelSearchSpace(ABC):
    """Defines an abstract base class for setting the search space and strategy used
    during hyperparameter optimization
    """

    def __init__(self, strategy):
        self.strategy = strategy
        self.space = {}
        self.step_name = ""
        super().__init__()

    def suggest_space(self, **kwargs):
        pass


class RidgeSearchSpace(ModelSearchSpace):
    def __init__(self, strategy):
        super(RidgeSearchSpace, self).__init__(strategy)

    def suggest_space(self, **kwargs):
        space = {"alpha": LogUniformDistribution(10**-4, 10**-1)}
        if self.strategy != "perfunctory":
            space.update(
                {
                    "alpha": LogUniformDistribution(10**-5, 10**1),
                    "fit_intercept": CategoricalDistribution([True, False]),
                    "normalize": CategoricalDistribution([True, False]),
                }
            )
        return _update_space_name(space, **kwargs)


class LassoSearchSpace(RidgeSearchSpace):
    pass


class RidgeClassifierSearchSpace(RidgeSearchSpace):
    pass


class ElasticNetSearchSpace(ModelSearchSpace):
    def __init__(self, strategy):
        super(ElasticNetSearchSpace, self).__init__(strategy)

    def suggest_space(self, **kwargs):
        space = {
            "alpha": LogUniformDistribution(10**-4, 10**-1),
            "l1_ratio": UniformDistribution(0, 1),
        }
        if self.strategy != "perfunctory":
            space.update(
                {
                    "alpha": LogUniformDistribution(10**-5, 10),
                    "fit_intercept": CategoricalDistribution([True, False]),
                    "l1_ratio": UniformDistribution(0, 1),
                    "normalize": CategoricalDistribution([True, False]),
                }
            )
        return _update_space_name(space, **kwargs)


class LogisticRegressionSearchSpace(ModelSearchSpace):
    def __init__(self, strategy):
        super(LogisticRegressionSearchSpace, self).__init__(strategy)

    def suggest_space(self, **kwargs):
        space = {
            "C": LogUniformDistribution(10**-4, 10**-1),
            "dual": CategoricalDistribution([False]),
            "penalty": CategoricalDistribution(["l1", "l2"]),
            "solver": CategoricalDistribution(["saga"]),
        }

        if self.strategy != "perfunctory":
            space.update(
                {
                    "C": LogUniformDistribution(10**-5, 10),
                    "l1_ratio": UniformDistribution(0, 1),
                    "penalty": CategoricalDistribution(["elasticnet"]),
                }
            )
        return _update_space_name(space, **kwargs)


class SGDClassifierSearchSpace(ModelSearchSpace):
    def __init__(self, strategy):
        super(SGDClassifierSearchSpace, self).__init__(strategy)

    def suggest_space(self, **kwargs):
        space = {
            "alpha": LogUniformDistribution(10**-4, 10**-1),
            "penalty": CategoricalDistribution(["l1", "l2", None]),
        }
        if self.strategy != "perfunctory":
            space.update(
                {
                    "alpha": LogUniformDistribution(10**-5, 10**1),
                    "l1_ratio": UniformDistribution(0, 1),
                    "penalty": CategoricalDistribution(["elasticnet"]),
                }
            )
        return _update_space_name(space, **kwargs)


class SGDRegressorSearchSpace(SGDClassifierSearchSpace):
    def __init__(self, strategy):
        super(SGDRegressorSearchSpace, self).__init__(strategy)


class SVCSearchSpace(ModelSearchSpace):
    def __init__(self, strategy):
        super(SVCSearchSpace, self).__init__(strategy)

    def suggest_space(self, **kwargs):
        space = {
            "C": LogUniformDistribution(10**-4, 10**-1),
            "max_iter": CategoricalDistribution([1000]),
        }
        if self.strategy != "perfunctory":
            space.update(
                {
                    "C": LogUniformDistribution(10**-5, 5),
                    "gamma": CategoricalDistribution(["scale", "auto"]),
                    "kernel": CategoricalDistribution(
                        ["linear", "poly", "rbf", "sigmoid"]
                    ),
                    "max_iter": CategoricalDistribution([5000]),
                }
            )
        return _update_space_name(space, **kwargs)


class SVRSearchSpace(SVCSearchSpace):
    pass


class LinearSVCSearchSpace(ModelSearchSpace):
    def __init__(self, strategy):
        super(LinearSVCSearchSpace, self).__init__(strategy)

    def suggest_space(self, **kwargs):
        space = {
            "C": LogUniformDistribution(10**-4, 10**-1),
            "dual": CategoricalDistribution([False]),
        }
        if self.strategy != "perfunctory":
            space.update(
                {
                    "C": LogUniformDistribution(10**-5, 5),
                    "class_weight": CategoricalDistribution(
                        ["balanced", None]
                    ),  # max_iter defaults to 1000
                    "fit_intercept": CategoricalDistribution([True, False]),
                    "loss": CategoricalDistribution(["squared_hinge"]),
                    "penalty": CategoricalDistribution(["l1"]),
                }
            )
        return _update_space_name(space, **kwargs)


class LinearSVRSearchSpace(ModelSearchSpace):
    def __init__(self, strategy):
        super(LinearSVRSearchSpace, self).__init__(strategy)

    def suggest_space(self, **kwargs):
        space = {"C": LogUniformDistribution(10**-4, 10**-1)}

        if self.strategy != "perfunctory":
            space.update(
                {
                    "C": LogUniformDistribution(10**-5, 10**1),
                    "dual": CategoricalDistribution([False]),
                    "fit_intercept": CategoricalDistribution([True, False]),
                    "loss": CategoricalDistribution(["squared_epsilon_insensitive"]),
                }
            )
        return _update_space_name(space, **kwargs)


class DecisionTreeClassifierSearchSpace(ModelSearchSpace):
    def __init__(self, strategy):
        super(DecisionTreeClassifierSearchSpace, self).__init__(strategy)

    def suggest_space(self, **kwargs):
        space = {
            "max_depth": IntUniformDistribution(1, 5),
            "min_impurity_decrease": UniformDistribution(0, 0.05),
            "min_samples_split": IntUniformDistribution(2, 500),
        }

        if self.strategy != "perfunctory":
            space.update(
                {
                    "criterion": CategoricalDistribution(["gini", "entropy"]),
                    "max_depth": IntUniformDistribution(1, 10),
                    "min_samples_leaf": IntUniformDistribution(2, 500),
                }
            )  # max_iter defaults to 1000

        return _update_space_name(space, **kwargs)


class DecisionTreeRegressorSearchSpace(ModelSearchSpace):
    def __init__(self, strategy):
        super(DecisionTreeRegressorSearchSpace, self).__init__(strategy)

    def suggest_space(self, **kwargs):
        space = {
            "max_depth": IntUniformDistribution(1, 5),
            "min_impurity_decrease": UniformDistribution(0, 0.05),
            "min_samples_split": IntUniformDistribution(2, 500),
        }

        if self.strategy != "perfunctory":
            space.update(
                {
                    "criterion": CategoricalDistribution(
                        [
                            "squared_error",
                            "friedman_mse",
                            "absolute_error",
                        ]
                    ),
                    "min_samples_leaf": IntUniformDistribution(2, 500),
                }
            )  # max_iter defaults to 1000

        return _update_space_name(space, **kwargs)


class XGBClassifierSearchSpace(ModelSearchSpace):
    def __init__(self, strategy):
        super(XGBClassifierSearchSpace, self).__init__(strategy)

    def suggest_space(self, **kwargs):
        space = {
            "colsample_bytree": UniformDistribution(0.6, 0.8),
            "learning_rate": LogUniformDistribution(0.3, 0.4),
            "max_depth": IntUniformDistribution(1, 5),
            "n_estimators": IntUniformDistribution(50, 250),
            "subsample": UniformDistribution(0.5, 1),
        }

        if self.strategy != "perfunctory":
            space.update(
                {
                    "colsample_bytree": UniformDistribution(0.3, 0.7),
                    "gamma": UniformDistribution(0, 10),
                    "learning_rate": LogUniformDistribution(0.001, 0.6),
                    "max_depth": IntUniformDistribution(1, 10),
                    "min_child_weight": IntUniformDistribution(0, 20),
                    "n_estimators": IntUniformDistribution(50, 500),
                    # 'scale_pos_weight': LogUniformDistribution(10 ** -5, 1),
                    "subsample": UniformDistribution(0.25, 1),
                    "reg_alpha": LogUniformDistribution(10**-5, 1),
                    "reg_lambda": LogUniformDistribution(10**-5, 1),
                }
            )
        return _update_space_name(space, **kwargs)


class XGBRegressorSearchSpace(XGBClassifierSearchSpace):
    pass


class LGBMClassifierSearchSpace(ModelSearchSpace):
    def __init__(self, strategy):
        super(LGBMClassifierSearchSpace, self).__init__(strategy)

    def suggest_space(self, **kwargs):
        space = {
            "colsample_bytree": UniformDistribution(0.3, 0.7),
            "learning_rate": LogUniformDistribution(0.3, 0.4),
            "max_depth": IntUniformDistribution(1, 5),
            "n_estimators": IntUniformDistribution(50, 250),
            "subsample": UniformDistribution(0.5, 1),
        }

        if self.strategy != "perfunctory":
            space.update(
                {
                    "boosting_type": CategoricalDistribution(["dart"]),
                    "learning_rate": LogUniformDistribution(0.001, 0.6),
                    "max_depth": IntUniformDistribution(1, 10),
                    "min_child_weight": IntUniformDistribution(0, 20),
                    "n_estimators": IntUniformDistribution(50, 500),
                    "num_leaves": IntLogUniformDistribution(7, 40),
                    "reg_alpha": LogUniformDistribution(10**-5, 1),
                    "reg_lambda": LogUniformDistribution(10**-5, 1),
                }
            )

        return _update_space_name(space, **kwargs)


class LGBMRegressorSearchSpace(LGBMClassifierSearchSpace):
    pass


class ExtraTreesClassifierSearchSpace(ModelSearchSpace):
    def __init__(self, strategy):
        super(ExtraTreesClassifierSearchSpace, self).__init__(strategy)

    def suggest_space(self, **kwargs):
        space = {
            "n_estimators": IntUniformDistribution(50, 250),
            "max_depth": IntUniformDistribution(1, 5),
            "max_features": CategoricalDistribution(["sqrt", "log2"]),
        }

        if self.strategy != "perfunctory":
            space.update(
                {
                    "max_depth": IntUniformDistribution(1, 10),
                    "min_impurity_decrease": UniformDistribution(0.0, 0.05),
                    "min_samples_split": IntUniformDistribution(2, 500),
                    "min_samples_leaf": IntUniformDistribution(5, 25),
                    "min_weight_fraction_leaf": UniformDistribution(0.0, 0.5),
                    "n_estimators": IntUniformDistribution(50, 500),
                }
            )

        return _update_space_name(space, **kwargs)


class ExtraTreesRegressorSearchSpace(ExtraTreesClassifierSearchSpace):
    pass


class RandomForestClassifierSearchSpace(ExtraTreesClassifierSearchSpace):
    pass


class RandomForestRegressorSearchSpace(ExtraTreesClassifierSearchSpace):
    pass


class GradientBoostingRegressorSearchSpace(ModelSearchSpace):
    def __init__(self, strategy):
        super(GradientBoostingRegressorSearchSpace, self).__init__(strategy)

    def suggest_space(self, **kwargs):
        space = {
            "max_depth": IntUniformDistribution(1, 5),
            "max_features": CategoricalDistribution(["sqrt", "log2"]),
            "n_estimators": IntUniformDistribution(50, 250),
        }

        if self.strategy != "perfunctory":
            space.update(
                {
                    "learning_rate": LogUniformDistribution(0.001, 0.6),
                    "max_depth": IntUniformDistribution(1, 10),
                    "min_samples_leaf": IntUniformDistribution(5, 25),
                    "min_samples_split": IntUniformDistribution(2, 500),
                    "n_estimators": IntUniformDistribution(50, 500),
                    "subsample": UniformDistribution(0.5, 1),
                }
            )

        return _update_space_name(space, **kwargs)


class GradientBoostingClassifierSearchSpace(GradientBoostingRegressorSearchSpace):
    pass


def get_model2searchspace():
    model2searchspace = {
        Ridge: RidgeSearchSpace,
        RidgeClassifier: RidgeClassifierSearchSpace,
        Lasso: LassoSearchSpace,
        ElasticNet: ElasticNetSearchSpace,
        LogisticRegression: LogisticRegressionSearchSpace,
        SVC: SVCSearchSpace,
        SVR: SVRSearchSpace,
        LinearSVC: LinearSVCSearchSpace,
        LinearSVR: LinearSVRSearchSpace,
        DecisionTreeClassifier: DecisionTreeClassifierSearchSpace,
        DecisionTreeRegressor: DecisionTreeRegressorSearchSpace,
        RandomForestClassifier: RandomForestClassifierSearchSpace,
        RandomForestRegressor: RandomForestRegressorSearchSpace,
        GradientBoostingClassifier: GradientBoostingClassifierSearchSpace,
        GradientBoostingRegressor: GradientBoostingRegressorSearchSpace,
        ExtraTreesClassifier: ExtraTreesClassifierSearchSpace,
        ExtraTreesRegressor: ExtraTreesRegressorSearchSpace,
        SGDClassifier: SGDClassifierSearchSpace,
        SGDRegressor: SGDRegressorSearchSpace,
    }
    try:
        from xgboost import XGBClassifier, XGBRegressor

        model2searchspace[XGBClassifier] = XGBClassifierSearchSpace
        model2searchspace[XGBRegressor] = XGBRegressorSearchSpace
    except:
        pass
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        model2searchspace[LGBMClassifier] = LGBMClassifierSearchSpace
        model2searchspace[LGBMRegressor] = LGBMRegressorSearchSpace
    except:
        pass
    return model2searchspace


model_list = list(get_model2searchspace().keys())
