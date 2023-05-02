#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pandas as pd
import warnings

from ads.common import utils, logger
from ads.dataset import helper
from ads.dataset.exception import ValidationError
from ads.dataset.dataset_with_target import ADSDatasetWithTarget
from sklearn.preprocessing import FunctionTransformer
from ads.dataset.helper import deprecate_variable, deprecate_default_value


class ClassificationDataset(ADSDatasetWithTarget):
    """
    Dataset for classification task
    """

    def __init__(self, df, sampled_df, target, target_type, shape, **kwargs):
        ADSDatasetWithTarget.__init__(
            self, df=df, sampled_df=sampled_df, target=target, target_type=target_type, shape=shape, **kwargs
        )

    def auto_transform(
        self,
        fix_imbalance: bool = True,
        correlation_threshold: float = 0.7,
        frac: float = 1.0,
        correlation_methods: str = "pearson",
    ):
        """
        Return transformed dataset with several optimizations applied automatically.
        The optimizations include:

        - Dropping constant and primary key columns, which has no predictive quality,
        - Imputation, to fill in missing values in noisy data:

            - For continuous variables, fill with mean if less than 40% is missing, else drop,
            - For categorical variables, fill with most frequent if less than 40% is missing, else drop,

        - Dropping strongly co-correlated columns that tend to produce less generalizable models,
        - Balancing dataset using up or down sampling.

        Parameters
        ----------
        fix_imbalance : bool, defaults to True.
            Fix imbalance between classes in dataset. Used only for classification datasets.
        correlation_threshold: float, defaults to 0.7. It must be between 0 and 1, inclusive.
            The correlation threshold where columns with correlation higher than the threshold will
            be considered as strongly co-correlated and recommended to be taken care of.
        frac: float, defaults to 1.0. Range -> (0, 1].
            What fraction of the data should be used in the calculation?
        correlation_methods: Union[list, str], defaults to 'pearson'.

            - 'pearson': Use Pearson's Correlation between continuous features,
            - 'cramers v': Use Cramer's V correlations between categorical features,
            - 'correlation ratio': Use Correlation Ratio Correlation between categorical and continuous features,
            - 'all': Is equivalent to ['pearson', 'cramers v', 'correlation ratio'].

            Or a list containing any combination of these methods, for example, ['pearson', 'cramers v'].

        Returns
        -------
        transformed_dataset : ADSDatasetWithTarget
            The dataset after transformation

        Examples
        --------
        >>> ds_clean = ds.auto_transform(correlation_threshold=0.6)
        """
        frac = deprecate_default_value(
            frac,
            None,
            1,
            f"<code>frac=None</code> is deprecated. Use <code>frac=1.0</code> instead.",
            FutureWarning,
        )
        with utils.get_progress_bar(7) as progress:
            df, sampled_df, transformer_pipeline = self._transform(
                progress=progress,
                fix_imbalance=fix_imbalance,
                correlation_threshold=correlation_threshold,
                frac=frac,
                correlation_methods=correlation_methods,
            )
            return self._build_new_dataset(
                df,
                sampled_df=sampled_df,
                transformers=transformer_pipeline.steps,
                progress=progress,
            )

    def convert_to_text_classification(self, text_column: str):
        """
        Builds a new dataset with the given text column as the only feature besides target.

        Parameters
        ----------
        text_column: str
            Feature name to use for text classification task

        Returns
        -------
        ds: TextClassificationDataset
            Dataset with one text feature and a classification target

        Examples
        --------
        >>> review_ds = DatasetFactory.open("review_data.csv")
        >>> ds_text_class = review_ds.convert_to_text_classification('reviews')
        """

        def _select_features(df, feature_names, target):
            if target in df.columns:
                feature_names = feature_names + [target]
            return df[feature_names]

        transformer = (
            f"convert_to_text_classification using feature {text_column}",
            FunctionTransformer(
                func=_select_features,
                validate=False,
                kw_args={
                    "feature_names": [self.target.name, text_column],
                    "target": self.target.name,
                },
            ).fit(self.sampled_df),
        )
        if utils.is_same_class(self, BinaryClassificationDataset):
            new_ds = BinaryTextClassificationDataset(
                self.df[[self.target.name, text_column]],
                self.sampled_df[[self.target.name, text_column]],
                self.target.name,
                self.target.type,
                (len(self.df), 2),
                **self.init_kwargs,
            )

        else:
            new_ds = MultiClassTextClassificationDataset(
                self.df[[self.target.name, text_column]],
                self.sampled_df[[self.target.name, text_column]],
                self.target.name,
                self.target.type,
                (len(self.df), 2),
                **self.init_kwargs,
            )
        new_ds.transformer_pipeline = self._update_transformer_pipeline(transformer)
        return new_ds

    def down_sample(self, sampler=None):
        """
        Fixes an imbalanced dataset by down-sampling.

        Parameters
        ----------
        sampler: An instance of SamplerMixin
            Should implement fit_resample(X,y) method. If None, does random down sampling.

        Returns
        -------
        down_sampled_ds: ClassificationDataset
            A down-sampled dataset.

        Examples
        --------
        >>> ds = DatasetFactory.open("some_data.csv")
        >>> ds_balanced_small = ds.down_sample()
        """
        return self._build_new_dataset(
            helper.down_sample(self.df, self.target.name)
            if sampler is None
            else helper.sample(
                sampler,
                self.df.drop(self.target.name, axis=1),
                self.df[self.target.name],
            )
        )

    def up_sample(self, sampler="default"):
        """
        Fixes imbalanced dataset by up-sampling

        Parameters
        ----------
        sampler: An instance of SamplerMixin
            Should implement fit_resample(X,y) method.
            If 'default', either SMOTE or random sampler will be used
        fill_missing_type: a string
            Can either be 'mean', 'mode' or 'median'.

        Returns
        -------
        up_sampled_ds: ClassificationDataset
            an up-sampled dataset

        Examples
        --------
        >>> ds = DatasetFactory.open("some_data.csv")
        >>> ds_balanced_large = ds.up_sample()
        """
        return self._build_new_dataset(
            helper.up_sample(
                self.df,
                self.target.name,
                sampler=sampler,
                feature_types=self.feature_types,
            )
        )


class BinaryClassificationDataset(ClassificationDataset):
    """
    Dataset for binary classification
    """

    def __init__(
        self, df, sampled_df, target, target_type, shape, positive_class=None, **kwargs
    ):
        if positive_class is not None:
            # map positive_class to True
            update_arg = lambda x: x == positive_class

            def mapper(df, column_name, arg):
                df[column_name] = df[column_name].map(arg)
                return df

            df = mapper(df, target, update_arg)
            sampled_df = mapper(sampled_df, target, update_arg)
        ClassificationDataset.__init__(
            self, df, sampled_df, target, target_type, shape, **kwargs
        )

    def set_positive_class(self, positive_class, missing_value=False):
        """
        Return new dataset with values in target column mapped to True or False
        in accordance with the specified positive label.

        Parameters
        ----------
        positive_class : same dtype as target
            The target label which should be identified as positive outcome from model.
        missing_value : bool
            missing values will be converted to this

        Returns
        -------
        dataset: same type as the caller

        Raises
        ------
        ValidationError
             if the positive_class is not present in target

        Examples
        --------
        >>> ds = DatasetFactory.open("iris.csv")
        >>> ds_with_target = ds.set_target('class')
        >>> ds_with_pos_class = ds.set_positive_class('setosa')
        """
        if positive_class not in self.target.target_vals:
            raise ValidationError(
                "Positive label '%s' not in target values '%s'"
                % (positive_class, self.target.target_vals)
            )

        return self.assign_column(
            self.target.name,
            lambda x: pd.isnull(x) and missing_value or x == positive_class,
        )


class MultiClassClassificationDataset(ClassificationDataset):
    """
    Dataset for multi-class classification
    """

    def __init__(self, df, sampled_df, target, target_type, shape, **kwargs):
        ClassificationDataset.__init__(
            self, df, sampled_df, target, target_type, shape, **kwargs
        )


class BinaryTextClassificationDataset(BinaryClassificationDataset):
    """
    Dataset for binary text classification
    """

    def __init__(self, df, sampled_df, target, target_type, shape, **kwargs):
        BinaryClassificationDataset.__init__(
            self, df, sampled_df, target, target_type, shape, **kwargs
        )

    def auto_transform(self):
        """
        Automatically chooses the most effective dataset transformation
        """
        logger.info("No optimizations.")
        return self

    def select_best_features(self, score_func=None, k=12):
        """
        Automatically chooses the best features and removes the rest
        """
        logger.info(
            "There are an insufficient number of features to do feature selection."
        )
        return self


class MultiClassTextClassificationDataset(MultiClassClassificationDataset):
    """
    Dataset for multi-class text classification
    """

    def __init__(self, df, sampled_df, target, target_type, shape, **kwargs):
        MultiClassClassificationDataset.__init__(
            self, df, sampled_df, target, target_type, shape, **kwargs
        )

    def auto_transform(self):
        """
        Automatically chooses the most effective dataset transformation
        """
        logger.info("No optimizations.")
        return self

    def select_best_features(self, score_func=None, k=12):
        """
        Automatically chooses the best features and removes the rest
        """
        logger.info(
            "There are an insufficient number of features to do feature selection."
        )
        return self
