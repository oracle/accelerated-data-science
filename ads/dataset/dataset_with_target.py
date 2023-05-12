#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import absolute_import, print_function

import abc
import importlib
from collections import defaultdict
from numbers import Number
from typing import Tuple, Union

import pandas as pd
from ads.common import utils, logger
from ads.common.data import ADSData
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.dataset import helper
from ads.dataset.dataset import ADSDataset
from ads.dataset.feature_engineering_transformer import FeatureEngineeringTransformer
from ads.dataset.feature_selection import FeatureImportance
from ads.dataset.helper import (
    DatasetDefaults,
    deprecate_default_value, 
    deprecate_variable, 
    generate_sample,
    get_target_type,
    is_text_data,
)
from ads.dataset.label_encoder import DataFrameLabelEncoder
from ads.dataset.pipeline import TransformerPipeline
from ads.dataset.progress import DummyProgressBar
from ads.dataset.recommendation import Recommendation
from ads.dataset.recommendation_transformer import RecommendationTransformer
from ads.dataset.target import TargetVariable
from ads.type_discovery.typed_feature import (
    CategoricalTypedFeature,
    ContinuousTypedFeature,
    DocumentTypedFeature,
    GISTypedFeature,
    OrdinalTypedFeature,
    TypedFeature,
    DateTimeTypedFeature, 
    TypedFeature
)
from sklearn.model_selection import train_test_split
from pandas.io.formats.printing import pprint_thing
from sklearn.preprocessing import FunctionTransformer
from abc import ABCMeta


class ADSDatasetWithTarget(ADSDataset, metaclass=ABCMeta):
    """
    This class provides APIs for preparing dataset for modeling.
    """

    def __init__(
        self,
        df,
        target,
        sampled_df=None,
        shape=None,
        target_type=None,
        sample_max_rows=-1,
        type_discovery=True,
        types={},
        parent=None,
        name="",
        metadata=None,
        transformer_pipeline=None,
        description=None,
        progress=DummyProgressBar(),
        **kwargs,
    ):
        self.recommendation_transformer = None
        if shape is None:
            shape = df.shape
        if sampled_df is None:
            sampled_df = generate_sample(
                df,
                shape[0],
                DatasetDefaults.sampling_confidence_level,
                DatasetDefaults.sampling_confidence_interval,
                **kwargs,
            )

        if parent is None:
            cols = sampled_df.columns.tolist()
            cols.insert(0, cols.pop(cols.index(target)))
            ADSDataset.__init__(
                self,
                df,
                sampled_df[[*cols]],
                shape,
                name=name,
                description=description,
                type_discovery=type_discovery,
                types=types,
                progress=progress,
                metadata=metadata,
                transformer_pipeline=transformer_pipeline,
                sample_max_rows=sample_max_rows,
            )
        else:
            self.__dict__ = parent.__dict__.copy()
            cols = self.sampled_df.columns.tolist()
            cols.insert(0, cols.pop(cols.index(target)))

            self.sampled_df = parent.sampled_df[[*cols]]

            # if parent has already been built, just reorder the columns to display the plot for target at beginning
            if parent.correlation is None:
                self.corr_futures = parent.corr_futures
            else:
                corr_cols = parent.sampled_df.select_dtypes(
                    exclude=["object"]
                ).columns.values.tolist()
                corr_cols.insert(0, corr_cols.pop(corr_cols.index(target)))
                self.correlation = parent.correlation.reindex(corr_cols)[[corr_cols]]
            self.feature_types = parent.feature_types
            self.feature_dist_html_dict = {}
            if len(parent.feature_dist_html_dict) > 0:
                parent_feature_dist_html_dict = parent.feature_dist_html_dict.copy()
                self.feature_dist_html_dict = {
                    target: parent_feature_dist_html_dict.pop(target)
                }
                self.feature_dist_html_dict.update(parent_feature_dist_html_dict)

        # drop all rows where target is nan
        target = target.strip().replace(" ", "_")

        #
        # as an optimization only dropna and regenerate sample when the target
        # has na values
        #

        if self.df[target].isna().sum():
            #
            # remove rows for which the target is null
            #
            self.df = self.df.dropna(subset=[target])

            #
            # we cannot simply drop null values from the sampled_df after a change
            # to the df - we must rebuild the sample from the new df
            #
            self.sampled_df = helper.generate_sample(
                self.df,
                sampled_df.shape[0],
                helper.DatasetDefaults.sampling_confidence_level,
                helper.DatasetDefaults.sampling_confidence_interval,
            )
            #
            # after regenerating the sample we need to move the target back to the head
            #
            cols = self.sampled_df.columns.tolist()
            cols.insert(0, cols.pop(cols.index(target)))
            self.sampled_df = self.sampled_df[[*cols]]

        if target_type is None:
            target_type = get_target_type(target, sampled_df, **kwargs)
        self.target = TargetVariable(self, target, target_type)

        # remove target from type discovery conversion
        for step in self.transformer_pipeline.steps:
            if (
                step[0] == "type_discovery"
                and self.target.name in step[1].kw_args["dtypes"]
            ):
                step[1].kw_args["dtypes"].pop(self.target.name)

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        target: str,
        sampled_df: pd.DataFrame = None,
        shape: Tuple[int, int] = None,
        target_type: TypedFeature = None,
        positive_class=None,
        **init_kwargs,
    ):
        from ads.dataset.classification_dataset import (
            BinaryClassificationDataset, 
            BinaryTextClassificationDataset, 
            MultiClassClassificationDataset, 
            MultiClassTextClassificationDataset
        )
        from ads.dataset.forecasting_dataset import ForecastingDataset
        from ads.dataset.regression_dataset import RegressionDataset

        if sampled_df is None:
            sampled_df = generate_sample(
                df,
                (shape or df.shape)[0],
                DatasetDefaults.sampling_confidence_level,
                DatasetDefaults.sampling_confidence_interval,
                **init_kwargs,
            )

        if target not in df:
            raise ValueError(
                f"{target} column doesn't exist in data frame. Specify a valid one instead."
            )
            
        if target_type is None:
            target_type = get_target_type(target, sampled_df, **init_kwargs)

        if len(df[target].dropna()) == 0:
            logger.warning(
                "It is not recommended to use an empty column as the target variable."
            )
            raise ValueError(
                f"We do not support using empty columns as the chosen target"
            )
        if utils.is_same_class(target_type, ContinuousTypedFeature):
            return RegressionDataset(
                df=df,
                sampled_df=sampled_df,
                target=target,
                target_type=target_type,
                shape=shape,
                **init_kwargs,
            )
        elif utils.is_same_class(
            target_type, DateTimeTypedFeature
        ) or df.index.dtype.name.startswith("datetime"):
            return ForecastingDataset(
                df=df,
                sampled_df=sampled_df,
                target=target,
                target_type=target_type,
                shape=shape,
                **init_kwargs,
            )

        # Adding ordinal typed feature, but ultimately we should rethink how we want to model this type
        elif utils.is_same_class(target_type, CategoricalTypedFeature) or utils.is_same_class(
            target_type, OrdinalTypedFeature
        ):
            if target_type.meta_data["internal"]["unique"] == 2:
                if is_text_data(sampled_df, target):
                    return BinaryTextClassificationDataset(
                        df=df,
                        sampled_df=sampled_df,
                        target=target,
                        shape=shape,
                        target_type=target_type,
                        positive_class=positive_class,
                        **init_kwargs,
                    )

                return BinaryClassificationDataset(
                    df=df,
                    sampled_df=sampled_df,
                    target=target,
                    shape=shape,
                    target_type=target_type,
                    positive_class=positive_class,
                    **init_kwargs,
                )
            else:
                if is_text_data(sampled_df, target):
                    return MultiClassTextClassificationDataset(
                        df=df,
                        sampled_df=sampled_df,
                        target=target,
                        target_type=target_type,
                        shape=shape,
                        **init_kwargs,
                    )
                return MultiClassClassificationDataset(
                    df=df,
                    sampled_df=sampled_df,
                    target=target,
                    target_type=target_type,
                    shape=shape,
                    **init_kwargs,
                )
        elif (
            utils.is_same_class(target, DocumentTypedFeature)
            or "text" in target_type["type"]
            or "text" in target
        ):
            raise ValueError(
                f"The column {target} cannot be used as the target column."
            )
        elif (
            utils.is_same_class(target_type, GISTypedFeature)
            or "coord" in target_type["type"]
            or "coord" in target
        ):
            raise ValueError(
                f"The column {target} cannot be used as the target column."
            )
        # This is to catch constant columns that are boolean. Added as a fix for pd.isnull(), and datasets with a
        #   binary target, but only data on one instance
        elif target_type and target_type["low_level_type"] == "bool":
            return BinaryClassificationDataset(
                df=df,
                sampled_df=sampled_df,
                target=target,
                shape=shape,
                target_type=target_type,
                positive_class=positive_class,
                **init_kwargs,
            )
        raise ValueError(
            f"Unable to identify problem type. Specify the data type of {target} using 'types'. "
            f"For example, types = {{{target}: 'category'}}"
        )

    def rename_columns(self, columns):
        """
        Returns a dataset with columns renamed.
        """
        if isinstance(columns, list):
            assert len(columns) == len(
                self.columns.values
            ), "columns length do not match the dataset"
            columns = dict(zip(self.columns.values, columns))
        assert isinstance(columns, dict)
        new_target = None
        if self.target.name in columns:
            new_target = columns[self.target.name]
        return self.rename(columns=columns, _new_target=new_target)

    def select_best_features(self, score_func=None, k=12):
        """
        Return new dataset containing only the top k features.

        Parameters
        ----------
        k: int, default 12
            The top 'k' features to select.
        score_func: function
            Scoring function to use to rank the features. This scoring function should take a 2d array X(features)
            and an array like y(target) and return a numeric score for each feature in the same order as X.

        Notes
        -----
        See also https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html
        and https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html

        Examples
        --------
        >>> ds = DatasetBrowser("sklearn").open("iris")
        >>> ds_small = ds.select_best_features(k=2)
        """
        tf = self._get_best_features_transformer(score_func=score_func, k=k)
        return self._build_new_dataset(
            tf[1].transform(self.df), tf[1].transform(self.sampled_df), transformers=tf
        )

    def auto_transform(
        self,
        correlation_threshold: float = 0.7,
        frac: float = 1.0,
        sample_size=1.0,
        correlation_methods: Union[str, list] = "pearson",
    ):
        """
        Return transformed dataset with several optimizations applied automatically.
        The optimizations include:

        - Dropping constant and primary key columns, which has no predictive quality,
        - Imputation, to fill in missing values in noisy data:

            - For continuous variables, fill with mean if less than 40% is missing, else drop,
            - For categorical variables, fill with most frequent if less than 40% is missing, else drop,

        - Dropping strongly co-correlated columns that tend to produce less generalizable models.

        Parameters
        ----------
        correlation_threshold: float, defaults to 0.7. It must be between 0 and 1, inclusive
            the correlation threshold where columns with correlation higher than the threshold will
            be considered as strongly co-correlated and recommended to be taken care of.
        frac: Is superseded by sample_size
        sample_size: float, defaults to 1.0. Float, Range -> (0, 1]
            What fraction of the data should be used in the calculation?
        correlation_methods: Union[list, str], defaults to 'pearson'

            - 'pearson': Use Pearson's Correlation between continuous features,
            - 'cramers v': Use Cramer's V correlations between categorical features,
            - 'correlation ratio': Use Correlation Ratio Correlation between categorical and continuous features,
            - 'all': Is equivalent to ['pearson', 'cramers v', 'correlation ratio'].

            Or a list containing any combination of these methods, for example, ['pearson', 'cramers v'].

        Returns
        -------
        transformed_dataset : ADSDatasetWithTarget

        Examples
        --------
        >>> ds_clean = ds.auto_transform()
        """
        frac = deprecate_default_value(
            frac,
            None,
            1,
            "<code>frac=None</code> is deprecated. Use <code>sample_size=1.0</code> instead.",
            FutureWarning,
        )

        if frac != 1.0:
            deprecate_frac = deprecate_variable(
                frac,
                sample_size,
                "<code>frac</code> is superseded by <code>sample_size</code>.",
                DeprecationWarning,
            )
            if sample_size == 1.0:
                sample_size = deprecate_frac

        if correlation_threshold > 1 or correlation_threshold < 0:
            raise AssertionError("correlation_threshold has to be between 0 and 1.")
        with utils.get_progress_bar(5) as progress:
            df, sampled_df, transformer_pipeline = self._transform(
                progress=progress,
                correlation_threshold=correlation_threshold,
                frac=sample_size,
                correlation_methods=correlation_methods,
            )
            return self._build_new_dataset(
                df,
                sampled_df=sampled_df,
                transformers=transformer_pipeline.steps,
                progress=progress,
            )

    def visualize_transforms(self):
        """
        Render a representation of the dataset's transform DAG.
        """

        helper.visualize_transformation(
            self.transformer_pipeline,
            text="- rows: {}\\l- columns: {}\\l".format(*self.shape),
        )

    def _suggested_code(self, action, recommendation_type, variable):
        if action == "Drop":
            return ".drop_columns([{}])".format('"' + variable + '"')
        if action == "Do nothing":
            return ""
        if "Drop " in action:
            return ".drop_columns([{}])".format('"' + action.split(" ")[1] + '"')
        if action == "Down-sample":
            return ".down_sample()"
        if action == "Up-sample":
            if importlib.util.find_spec("imblearn") is None:
                return ".up_sample(sampler='default') \\n `pip install imbalanced-learn` to use default up-sampler."
            else:
                return ".up_sample(sampler='default')"
        if recommendation_type == "positive_class" and action != "Do nothing":
            return ".set_positive_class({}, missing_value=False)".format(
                '"' + action + '"'
            )
        if recommendation_type == "imputation":
            fill_val = helper.get_fill_val(
                self.feature_types, variable, action, constant="constant"
            )

            fill_val = (
                fill_val if isinstance(fill_val, Number) else '"' + fill_val + '"'
            )
            return ".fillna({}{}: {}{})".format(
                "{", '"' + variable + '"', fill_val, "}"
            )
        else:
            return ""

    def suggest_recommendations(
        self,
        correlation_methods: Union[str, list] = "pearson",
        print_code: bool = True,
        correlation_threshold: float = 0.7,
        overwrite: bool = None,
        force_recompute: bool = False,
        frac: float = 1.0,
        sample_size: float = 1.0,
        **kwargs,
    ):
        """
        Returns a pandas dataframe with suggestions for dataset optimization. This includes:

        - Identifying constant and primary key columns, which has no predictive quality,
        - Imputation, to fill in missing values in noisy data:

            - For continuous variables, fill with mean if less than 40% is missing, else drop,
            - For categorical variables, fill with most frequent if less than 40% is missing, else drop,

        - Identifying strongly co-correlated columns that tend to produce less generalizable models,
        - Automatically balancing dataset for classification problems using up or down sampling.

        Parameters
        ----------
        correlation_methods: Union[list, str], default to 'pearson'

            - 'pearson': Use Pearson's Correlation between continuous features,
            - 'cramers v': Use Cramer's V correlations between categorical features,
            - 'correlation ratio': Use Correlation Ratio Correlation between categorical and continuous features,
            - 'all': Is equivalent to ['pearson', 'cramers v', 'correlation ratio'].

            Or a list containing any combination of these methods, for example, ['pearson', 'cramers v']
        print_code: bool, Defaults to True
            Print Python code for the suggested actions.
        correlation_threshold: float. Defaults to 0.7. It must be between 0 and 1, inclusive
            the correlation threshold where columns with correlation higher than the threshold will
            be considered as strongly co-correated and recommended to be taken care of.
        frac: Is superseded by sample_size
        sample_size: float, defaults to 1.0. Float, Range -> (0, 1]
            What fraction of the data should be used in the calculation?
        overwrite:
            Is deprecated and replaced by force_recompute.
        force_recompute: bool, default to be False

            - If False, it calculates the correlation matrix if there is no cached correlation matrix. Otherwise,
              it returns the cached correlation matrix.
            - If True, it calculates the correlation matrix regardless whether there is cached result or not.

        Returns
        -------
        suggestion dataframe : pandas.DataFrame

        Examples
        --------
        >>> suggestion_df = ds.suggest_recommendations(correlation_threshold=0.7)
        """
        frac = deprecate_default_value(
            frac,
            None,
            1,
            "<code>frac=None</code> is deprecated. Use <code>sample_size=1.0</code>.",
            FutureWarning,
        )

        if frac != 1.0:
            deprecate_frac = deprecate_variable(
                frac,
                sample_size,
                "<code>frac</code> is superseded by <code>sample_size</code>.",
                DeprecationWarning,
            )
            if sample_size == 1.0:
                sample_size = deprecate_frac

        force_recompute = deprecate_variable(
            overwrite,
            force_recompute,
            f"<code>overwrite=None</code> is deprecated. Use <code>force_recompute</code> instead.",
            DeprecationWarning,
        )

        recommended = self._get_recommendations_transformer(
            auto_transform=False,
            correlation_threshold=correlation_threshold,
            correlation_methods=correlation_methods,
            force_recompute=force_recompute,
            frac=sample_size,
            **kwargs,
        ).fit(self.sampled_df)

        if len(recommended.reco_dict_) == 0:
            logger.info("No recommendations.")
            return pd.DataFrame()

        column_names = [
            "Message",
            "Variables",
            "Action",
            "Selected Action",
            "Recommendation Type",
        ]

        df_dict = defaultdict(list)

        for recommendation_type, column_dict in recommended.reco_dict_.items():
            if recommendation_type == "constant_column":
                n_constant = len(column_dict)
                df_dict["Recommendation Type"].extend(
                    [recommendation_type] * n_constant
                )
                df_dict["Variables"].extend(column_dict)
                df_dict["Message"].extend(["Constant Column"] * n_constant)
                df_dict["Action"].extend(["Drop"] * n_constant)
                df_dict["Selected Action"].extend(["Drop"] * n_constant)
                continue

            for column, details_dict in column_dict.items():
                max_length = len(details_dict["Action"])
                for key, value in details_dict.items():
                    if isinstance(value, list):
                        df_dict[key].extend(value)
                    else:
                        df_dict[key].extend([value] * max_length)
                df_dict["Recommendation Type"].extend(
                    [recommendation_type] * max_length
                )
                df_dict["Variables"].extend([column] * max_length)

        suggestions_df = pd.DataFrame.from_dict(df_dict)[column_names]
        suggestions_df["Code"] = suggestions_df.apply(
            lambda x: self._suggested_code(
                x["Action"], x["Recommendation Type"], x["Variables"]
            ),
            axis=1,
        )
        suggestion_df = (
            suggestions_df.drop(columns=["Recommendation Type"])
            .rename(columns={"Selected Action": "Suggested"})
            .set_index(["Message", "Variables", "Suggested", "Action"])
            .fillna("")
        )
        if print_code:
            columns_to_impute = {}
            columns_to_drop = []
            consolidated_code = ""
            suggestion_df_ = suggestion_df.reset_index()
            suggested_code = suggestion_df_.loc[
                suggestion_df_.Suggested == suggestion_df_.Action
            ].Code.unique()
            for code in suggested_code:
                if ".drop_columns" in code:
                    columns_to_drop.append(code.split("[")[1].split("]")[0][1:-1])
                elif ".fillna" in code:
                    impute_pair = code.split("{")[1].split("}")[0]
                    columns_to_impute[impute_pair.split(":")[0].replace('"', "")] = (
                        float(impute_pair.split(":")[1].strip())
                        if impute_pair.split(":")[1].strip().replace(".", "").isdigit()
                        else impute_pair.split(":")[1].strip().replace('"', "")
                    )
                else:
                    consolidated_code += code
            consolidated_code = (
                "No more!" if len(consolidated_code) == 0 else consolidated_code
            )

            logger.info(f"Suggested columns to drop: {columns_to_drop}.")
            logger.info(f"Suggested columns to impute: {columns_to_impute}.")
            logger.info(f"Others: {consolidated_code}.")

        return suggestion_df

    @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
    def get_recommendations(
        self,
        correlation_methods: str = "pearson",
        correlation_threshold: float = 0.7,
        frac: float = 1.0,
        sample_size: float = 1.0,
        overwrite: bool = None,
        force_recompute: bool = False,
        display_format: str = "widget",
    ):
        """
        Generate recommendations for dataset optimization. This includes:

        - Identifying constant and primary key columns, which has no predictive quality,
        - Imputation, to fill in missing values in noisy data:

            - For continuous variables, fill with mean if less than 40% is missing, else drop,
            - For categorical variables, fill with most frequent if less than 40% is missing, else drop,

        - Identifying strongly co-correlated columns that tend to produce less generalizable models,
        - Automatically balancing dataset for classification problems using up or down sampling.

        Parameters
        ----------
        correlation_methods: Union[list, str], default to 'pearson'

            - 'pearson': Use Pearson's Correlation between continuous features,
            - 'cramers v': Use Cramer's V correlations between categorical features,
            - 'correlation ratio': Use Correlation Ratio Correlation between categorical and continuous features,
            - 'all': Is equivalent to ['pearson', 'cramers v', 'correlation ratio'].

            Or a list containing any combination of these methods, for example, ['pearson', 'cramers v'].
        correlation_threshold: float, defaults to 0.7. It must be between 0 and 1, inclusive
            The correlation threshold where columns with correlation higher than the threshold will
            be considered as strongly co-correlated and recommended to be taken care of.
        frac: Is superseded by sample_size
        sample_size: float, defaults to 1.0. Float, Range -> (0, 1]
            What fraction of the data should be used in the calculation?
        overwrite:
            Is deprecated and replaced by force_recompute.
        force_recompute: bool, default to be False

            - If False, it calculates the correlation matrix if there is no cached correlation matrix. Otherwise,
              it returns the cached correlation matrix.
            - If True, it calculates the correlation matrix regardless whether there is cached result or not.

        display_format: string, defaults to 'widget'.
            Should be either 'widget' or 'table'. If 'widget',
            a GUI style interface is popped out; if 'table', a table of suggestions is shown.
        """
        frac = deprecate_default_value(
            frac,
            None,
            1,
            "<code>frac=None</code> is superseded by <code>sample_size=1.0</code>.",
            FutureWarning,
        )

        if frac != 1.0:
            deprecate_frac = deprecate_variable(
                frac,
                sample_size,
                "<code>frac</code> is superseded by <code>sample_size</code>.",
                DeprecationWarning,
            )
            if sample_size == 1.0:
                sample_size = deprecate_frac

        force_recompute = deprecate_variable(
            overwrite,
            force_recompute,
            f"<code>overwrite=None</code> is deprecated. Use <code>force_recompute</code> instead.",
            DeprecationWarning,
        )

        if display_format == "widget":
            recommended = Recommendation(
                self,
                self._get_recommendations_transformer(
                    auto_transform=False,
                    correlation_threshold=correlation_threshold,
                    correlation_methods=correlation_methods,
                    frac=sample_size,
                    force_recompute=force_recompute,
                ).fit(self.sampled_df),
            )

            if len(recommended.reco_dict) == 0:
                logger.info("No recommendations.")

            return recommended

        elif display_format == "table":
            df_suggestion = self.suggest_recommendations(
                correlation_threshold=correlation_threshold,
                frac=sample_size,
                force_recompute=force_recompute,
            )

            from IPython.display import HTML, display

            display(
                HTML(
                    df_suggestion.to_html()
                    .replace(" `", "<code>")
                    .replace("` ", "</code>")
                    .replace("\\n", "<br>")
                )
            )

    def get_transformed_dataset(self):
        """
        Return the transformed dataset with the recommendations applied.

        This method should be called after applying the recommendations using the Recommendation#show_in_notebook() API.
        """
        if hasattr(self, "new_ds"):
            return self.new_ds
        logger.info(
            "Use `get_recommendations()` to view or update recommendation or `auto_tranform()` first."
        )
        logger.warning(
            "`get_transformed_dataset` is deprecated and will be removed in a future release."
        )
        return

    def type_of_target(self):
        """
        Return the target type for the dataset.

        Returns
        -------
        target_type: TypedFeature
            an object of TypedFeature

        Examples
        --------
        >>> ds = ds.set_target('target_class')
        >>> assert(ds.type_of_target() == 'categorical')
        """
        return self.target.type

    def train_test_split(self, test_size=0.1, random_state=utils.random_state):
        """
        Splits  dataset to train and test data.

        Parameters
        ----------
        test_size: Union[float, int], optional, default=0.1
        random_state: Union[int, RandomState], optional, default=None

                - If int, random_state is the seed used by the random number generator;
                - If RandomState instance, random_state is the random number generator;
                - If None, the random number generator is the RandomState instance used by np.random.

        Returns
        -------
        train_data, test_data: tuple
            tuple of ADSData instances

        Examples
        --------
        >>> ds = DatasetFactory.open("data.csv")
        >>> train, test = ds.train_test_split()
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.df.drop(self.target.name, axis=1),
            self.df[self.target.name],
            test_size=test_size,
            train_size=1 - test_size,
            random_state=random_state,
        )
        train = ADSData.build(
            X=X_train, y=y_train, name="Train Data", dataset_type=self.__class__
        )
        train.transformer_pipeline = self.transformer_pipeline
        test = ADSData.build(
            X=X_test, y=y_test, name="Test Data", dataset_type=self.__class__
        )
        return train, test

    def train_validation_test_split(
        self, test_size=0.1, validation_size=0.1, random_state=utils.random_state
    ):
        """
        Splits  dataset to train, validation and test data.

        Parameters
        ----------
        test_size: Union[float, int], optional, default=0.1
        validation_size: Union[float, int], optional, default=0.1
        random_state: Union[int, RandomState], optional, default=None

                - If int, random_state is the seed used by the random number generator;
                - If RandomState instance, random_state is the random number generator;
                - If None, the random number generator is the RandomState instance used by np.random.

        Returns
        -------
        train_data, validation_data, test_data: tuple
            tuple of ADSData instances

        Examples
        --------
        >>> ds = DatasetFactory.open("data.csv")
        >>> train, valid, test = ds.train_validation_test_split()
        """
        train, test = self.train_test_split(
            test_size=test_size, random_state=random_state
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            train.X, train.y, test_size=validation_size, random_state=random_state
        )
        train.X = X_train
        train.y = y_train
        valid = ADSData.build(
            X=X_valid, y=y_valid, name="Validation Data", dataset_type=self.__class__
        )
        return train, valid, test

    """
    Internal methods
    """

    def __repr__(self):
        rows, cols = self.shape
        return f"{self.__class__.__name__}(target: {self.target.name}) {rows:,} rows, {cols:,} columns"

    def _transform(
        self,
        progress=DummyProgressBar(),
        fix_imbalance=True,
        correlation_threshold=0.7,
        frac=None,
        correlation_methods="pearson",
    ):
        progress.update("Building the transformer pipeline")
        if self.recommendation_transformer is None:
            transformer_pipeline = TransformerPipeline(
                steps=[
                    (
                        "recommendations",
                        self._get_recommendations_transformer(
                            fix_imbalance=fix_imbalance,
                            correlation_threshold=correlation_threshold,
                            frac=frac,
                            correlation_methods=correlation_methods,
                        ),
                    ),
                    (
                        "feature_engineering",
                        FeatureEngineeringTransformer(
                            feature_metadata=self.feature_types
                        ),
                    ),
                ]
            )
        else:
            # recommendations are already generated using get_recommendations().show_in_notebook() API
            transformer_pipeline = TransformerPipeline(
                steps=[
                    (
                        "feature_engineering",
                        FeatureEngineeringTransformer(
                            feature_metadata=self.feature_types
                        ),
                    )
                ]
            )
            transformer_pipeline.steps = [
                ("recommendations", self.recommendation_transformer)
            ] + transformer_pipeline.steps
        sampled_df = self.sampled_df.copy()
        self.recommendation_transformer = None
        df = self.df.copy()
        for step in transformer_pipeline.steps:
            progress.update("Applying transformation for %s" % step[0])
            sampled_df = step[1].fit_transform(sampled_df)
            df = step[1].transform(df, progress=progress, fit_transform=True)
        return df, sampled_df, transformer_pipeline

    def _get_best_features(self, score_func=None, k=12):
        if isinstance(self.target.type, DateTimeTypedFeature):
            return FeatureImportance._get_feature_ranking(
                self.sampled_df.copy(),
                self.target.name,
                self.type_of_target(),
                score_func=score_func,
                k=k,
            )
        else:
            return FeatureImportance._get_feature_ranking(
                self.sampled_df.copy(),
                self.target.name,
                self.type_of_target(),
                score_func=score_func,
                k=k,
            )

    def _get_best_features_transformer(self, score_func=None, k=12):
        feature_set = self._get_best_features(k=k, score_func=score_func)[
            "features"
        ].tolist()

        def _select_features(df, feature_names, target):
            if target in df.columns:
                feature_names = feature_names + [target]
            return df[feature_names]

        return (
            "select_{0}_best_features".format(k),
            FunctionTransformer(
                func=_select_features,
                validate=False,
                kw_args={"feature_names": feature_set, "target": self.target.name},
            ).fit(self.sampled_df),
        )

    def _get_recommendations_transformer(
        self,
        fix_imbalance=True,
        auto_transform=True,
        correlation_threshold=0.7,
        **kwargs,
    ):
        force_recompute = kwargs.pop("force_recompute", False)
        frac = kwargs.pop("frac", 1)
        correlation_methods = kwargs.pop("correlation_methods", "pearson")
        return RecommendationTransformer(
            feature_metadata=self.feature_types,
            correlation=self.corr(
                force_recompute=force_recompute,
                frac=frac,
                correlation_methods=correlation_methods,
                **kwargs,
            ),
            target=self.target.name,
            target_type=self.target.type,
            is_balanced=self.target.is_balanced(),
            feature_ranking=self._get_best_features(k=len(self.sampled_df)),
            fix_imbalance=fix_imbalance,
            len=self.__len__(),
            auto_transform=auto_transform,
            correlation_threshold=correlation_threshold,
        )
