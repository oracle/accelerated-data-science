#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np

from ads.explanations.base_explainer import WhatIfExplainer
from ads.explanations.mlx_interface import init_whatif_explainer
from ads.common import logger
from ads.common.utils import _log_multivalue_feature_column_error
from ads.common.decorator.deprecate import deprecated


class MLXWhatIfExplainer(WhatIfExplainer):
    """
    MLX WhatIf class.

    Generates "what if" explanations to conduct sensitivity test. Supported explanations:

        - (Interactive Widgets and Plotly Visualization) Explore sample.
        - (Interactive Widgets and Plotly Visualization) Explore predictions.

    Supports:

        - Binary classification. (Tabular Data only)
        - Regression. (Tabular Data only)

    """

    @deprecated(
        details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
        raise_error=True,
    )
    def __init__(self):
        super(MLXWhatIfExplainer, self).__init__()
        self.explainer = None

    def explore_sample(self, row_idx=None, features=None, max_features=32, **kwargs):
        """
        Parameters
        ----------
        row_idx: int, optional
            Row index of the sample to explore. Default value is `None`
        features: list of str or int (dataframe column names), optional
            Feature columns to explore. Default value is `None`.
        max_features: int, optional
            Maximum number of features for modification. Default value is 32
        """

        if self.explainer is None:
            self.explainer = init_whatif_explainer(
                explainer=self.explainer,
                est=self.est_,
                X_test=self.X_test_,
                y_test=self.y_test_,
                mode=self.mode_,
                class_names=self.class_names_
                if self.mode_ == "classification"
                else ["prediction"],
                train=self.X_train_,
                target_title=None,
                **kwargs,
            )

        kwargs.pop("width", None)
        if row_idx is not None and row_idx not in self.X_test.index:
            logger.info(f"`row_idx` {row_idx} is not in the index. Resetting to 0.")
            # print('row_idx (%s) not in the index, reset to 0.'% str(row_idx))
            row_idx = 0

            # Wrap in a list if a list is not provided

        if features is not None and not isinstance(features, list):
            features = [features]
        if features:
            feature_names = self.X_train.columns.tolist()

            # Fail if we were not provided valid feature names
            if not all(np.isin(features, feature_names)):
                logger.error(
                    f"One or more `features` {str(features)} does not exist in data. "
                )
                logger.info(f"Existing features: {feature_names}.")
                return

        return self.explainer.explore_sample(
            row_idx=row_idx,
            features=features,
            max_features=max_features,
            plot_type="pyplot",
            **kwargs,
        )

    def explore_predictions(
        self,
        x=None,
        y=None,
        label=None,
        plot_type="scatter",
        discretization=None,
        **kwargs,
    ):
        """
        Explore explanation predictions

        Parameters
        ----------
        x: str, optional
            Feature column on x-axis. The default is `None`.
        y: str, optional
            Feature column or model prediction column on the y-axis, by default it is the target.
        label: str or int, optional
            Target label or target class name to explore only for classification problems.
            The default is `None`.
        plot_type: str, optional
            Type of plot. For classification problems the valid options are 'scatter',
            'box', or 'bar'. For a regression problem, the valid options are 'scatter' or 'box'.
            The default is 'scatter'.
        discretization: str, optional
            Discretization method applies the x-axis if the feature x is continuous.
            The valid options are 'quartile', 'decile', or 'percentile'. The default is `None`.
        """

        if self.explainer is None:
            self.explainer = init_whatif_explainer(
                explainer=self.explainer,
                est=self.est_,
                X_test=self.X_test_,
                y_test=self.y_test_,
                mode=self.mode_,
                class_names=self.class_names_
                if self.mode_ == "classification"
                else ["prediction"],
                train=self.X_train_,
                target_title=None,
                **kwargs,
            )

        kwargs.pop("width", None)
        if isinstance(x, list) or isinstance(y, list):
            _log_multivalue_feature_column_error()
            return
        for feature in [x, y]:
            if feature is not None:
                if isinstance(feature, list):
                    _log_multivalue_feature_column_error()
                if not isinstance(feature, str):
                    feature = str(feature)

                # Convert to uppercase to be case-insensitive
                feature_upper = feature.upper()
                feature_names = np.char.upper(self.X_test.columns.tolist())

                # Fail if we were not provided valid feature names
                if not np.isin(feature_upper, feature_names):
                    logger.error(f"`{feature_upper}` does not exist in data. ")
                    logger.info(
                        f"Existing features: `{str(self.X_test.columns.tolist())}`."
                    )
                    return
        if self.class_names_ is not None:
            num_classes = len(self.class_names_)
            if isinstance(label, str) and label not in self.class_names_:
                logger.error(f"`label` must be one of {self.class_names_}.")
                return
            if isinstance(label, int) and label > num_classes - 1:
                logger.error(
                    f"`label` cannot exceed the value of {int(num_classes - 1)}."
                )
                return

        plot_type_options = ["scatter", "bar", "box"]
        if plot_type not in plot_type_options:
            logger.error(f"`plot_type` must be one of {str(plot_type_options)}.")
            return
        discretization_options = [None, "quartile", "decile", "percentile"]
        if discretization not in discretization_options:
            logger.error(
                f"`discretization` must be one of {str(discretization_options)}."
            )
            return

        return self.explainer.explore_predictions(
            x=x,
            y=y,
            label=label,
            plot_type=plot_type,
            discretization=discretization,
            **kwargs,
        )


class WhatIfExplanationsException(TypeError):
    @deprecated(
        details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
        raise_error=True,
    )
    def __init__(self, msg):
        super(WhatIfExplanationsException, self).__init__(msg)
