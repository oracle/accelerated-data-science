#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, absolute_import

import inspect

import pandas as pd
from sklearn.base import TransformerMixin


def expand_lambda_function(lambda_func):
    """
    Returns a lambda function after expansion.
    """
    lambda_function = inspect.getsource(lambda_func).split(",")[1].strip()
    if lambda_function.endswith(")"):
        return lambda_function[:-1]
    return lambda_function


class DataFrameTransformer(TransformerMixin):
    """
    A DataFrameTransformer object.
    """

    def __init__(
        self, func_name, target_name, target_sample_val, args=None, kw_args=None
    ):
        self.function_name_ = func_name
        self.target_name = target_name
        self.target_sample_val = target_sample_val
        self.function_args_ = args
        self.function_kwargs_ = kw_args

    def __repr__(self):
        return "\n{}({})\n\n{}".format(
            self.function_name_,
            self.target_name,
            "\n".join(
                [
                    expand_lambda_function(f)
                    if f.__name__ == "<lambda>"
                    else "{} {}".format(f.__name__, f.__code__.co_varnames)
                    for f in self.function_args_
                    if callable(f)
                ]
            ),
        )

    def fit(self, df):
        """
        Takes in a DF and returns a fitted model
        """
        return self

    def transform(self, df):
        """
        Takes in a DF and returns a transformed DF
        """
        return self._transform(df)[0]

    def _transform(self, df):
        """
        Transform the dataframe using the function provided

        If a given function is not supported by pandas dataframe, the transform would be a no-op

        Parameters
        ----------
        df: Union[pandas.DataFrame, dask.dataframe.core.DataFrame]

        Returns
        -------
        tuple(transformed_df, is_transformed)
            transformed_df is same type as the input
            is_transformed, bool
                True if the transformation function coulfunction_args_d be applied
        """
        # add a dummy target column
        drop_target = False
        if self.target_name is not None and self.target_name not in df.columns:
            drop_target = True
            df = df.assign(**{self.target_name: self.target_sample_val})

        # check if pandas dataframe has this function
        if hasattr(df, self.function_name_):
            function = getattr(df, self.function_name_)

        # if pandas dataframe does not have this function, it is possible that the function being accessed is
        # similar to dask.dataframe.core.DataFrame.map_partitions that takes python function as the first
        # argument and applies the function on each partition. The same effect can be achieved using pipe
        elif (
            isinstance(df, pd.DataFrame)
            and len(self.function_args_) > 0
            and callable(self.function_args_[0])
        ):
            function = getattr(df, "pipe")
        else:
            # this method cannot be applied on pandas dataframe
            return df, False

        function_args = self.function_args_
        result = function(*function_args, **self.function_kwargs_)
        if drop_target:
            result = result.drop(self.target_name, axis=1)
        return result, True
