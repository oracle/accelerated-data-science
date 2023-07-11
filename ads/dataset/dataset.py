#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, absolute_import, division

import copy
import datetime
import fsspec
import numpy as np
import os
import pandas as pd
import uuid

from collections import Counter
from sklearn.preprocessing import FunctionTransformer
from typing import Iterable, Tuple, Union

from ads import set_documentation_mode
from ads.common import utils
from ads.common.decorator.deprecate import deprecated
from ads.dataset import helper, logger
from ads.dataset.dataframe_transformer import DataFrameTransformer
from ads.dataset.exception import ValidationError
from ads.dataset.helper import (
    convert_columns,
    fix_column_names,
    generate_sample,
    DatasetDefaults,
    deprecate_default_value,
    deprecate_variable,
    get_dataset,
    infer_target_type,
)
from ads.dataset.label_encoder import DataFrameLabelEncoder
from ads.dataset.pipeline import TransformerPipeline
from ads.dataset.progress import DummyProgressBar
from ads.dataset.sampled_dataset import PandasDataset
from ads.type_discovery.type_discovery_driver import TypeDiscoveryDriver
from ads.dataset.helper import get_feature_type
from ads.dataset.correlation_plot import plot_correlation_heatmap
from ads.dataset.correlation import (
    _cat_vs_cts,
    _cat_vs_cat,
    _get_columns_by_type,
    _validate_correlation_methods,
)
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)

N_Features_Wide_Dataset = 64


pd.set_option("display.max_colwidth", None)


class ADSDataset(PandasDataset):
    """
    An ADSDataset Object.

    The ADSDataset object cannot be used for classification or regression problems until a
    target has been set using `set_target`. To see some rows in the data use any of the usual
    Pandas functions like `head()`. There are also a variety of converters, to_dask,
    to_pandas, to_h2o, to_xgb, to_csv, to_parquet, to_json & to_hdf .
    """

    df_read_functions = ["head", "describe", "_get_numeric_data"]

    def __init__(
        self,
        df,
        sampled_df=None,
        shape=None,
        name="",
        description=None,
        type_discovery=True,
        types={},
        metadata=None,
        progress=DummyProgressBar(),
        transformer_pipeline=None,
        interactive=False,
        **kwargs,
    ):

        #
        # to keep performance high and linear no matter the size of the distributed dataset we
        # create a pandas df that's used internally because this has a fixed upper size.
        #
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
        super().__init__(
            sampled_df,
            type_discovery=type_discovery,
            types=types,
            metadata=metadata,
            progress=progress,
        )
        self.df = fix_column_names(df)

        self.name = name
        self.description = description
        self.shape = shape
        # store these args to reapply when building a new dataset for delegate operations on dataframe
        self.init_kwargs = {**kwargs, "type_discovery": type_discovery}
        if transformer_pipeline is None:
            # Update transformer pipeline to convert column types and fix names
            self.transformer_pipeline = TransformerPipeline(
                steps=[
                    (
                        "prepare",
                        FunctionTransformer(func=fix_column_names, validate=False),
                    )
                ]
            )
            self.transformer_pipeline = self._update_transformer_pipeline(
                steps=[
                    (
                        "type_discovery",
                        FunctionTransformer(
                            func=convert_columns,
                            validate=False,
                            kw_args={"dtypes": self.sampled_df.dtypes},
                        ),
                    )
                ]
            )
        else:
            self.transformer_pipeline = transformer_pipeline

    def __repr__(self):
        rows, cols = self.shape
        return f"{rows:,} rows, {cols:,} columns"

    def __len__(self):
        return self.shape[0]

    @staticmethod
    def from_dataframe(
        df,
        sampled_df=None,
        shape=None,
        name="",
        description=None,
        type_discovery=True,
        types={},
        metadata=None,
        progress=DummyProgressBar(),
        transformer_pipeline=None,
        interactive=False,
        **kwargs,
    ) -> "ADSDataset":
        return ADSDataset(
            df=df,
            sampled_df=sampled_df,
            shape=shape,
            name=name,
            description=description,
            type_discovery=type_discovery,
            types=types,
            metadata=metadata,
            progress=progress,
            transformer_pipeline=transformer_pipeline,
            interactive=interactive,
            **kwargs,
        )

    @property
    @deprecated(
        "2.5.2", details="The ddf attribute is deprecated. Use the df attribute."
    )
    def ddf(self):
        return self.df

    @deprecated(
        "2.5.2", details="The compute method is deprecated. Use the df attribute."
    )
    def compute(self):
        return self.df

    @runtime_dependency(
        module="ipywidgets", object="HTML", install_from=OptionalDependency.NOTEBOOK
    )
    @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
    def _repr_html_(self):
        from IPython.core.display import display, HTML

        display(
            HTML(
                utils.horizontal_scrollable_div(
                    self.sampled_df.head(5)
                    .style.set_table_styles(utils.get_dataframe_styles())
                    .set_table_attributes("class=table")
                    .hide_index()
                    .render()
                )
            )
        )

    def _head(self, n=5):
        """
        Return the first `n` rows of the dataset.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.

        Returns
        -------
        dataset_head : pandas.DataFrame
            The first `n` rows of the dataset

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("classfication_data.csv"))
        >>> ds.head()
        * displays the first 5 rows of the dataset, just as the traditional head() function would *
        """
        df = self.df.head(n=n)

        #
        # we could just return the above but, jupyterlab doesn't render these well
        # when the width exceeds the screen area. To address that we wrap the dataframe
        # with a class that has an optimized _repr_html_ handler, this object
        # extends the pandas dataframe so it can still be used as-a dataframe
        #
        class FormattedDataFrame(pd.DataFrame):
            def __init__(self, *args, **kwargs):
                super(FormattedDataFrame, self).__init__(*args, **kwargs)

            @property
            def _constructor(self):
                return FormattedDataFrame

            @runtime_dependency(
                module="ipywidgets",
                object="HTML",
                install_from=OptionalDependency.NOTEBOOK,
            )
            @runtime_dependency(
                module="IPython", install_from=OptionalDependency.NOTEBOOK
            )
            def _repr_html_(self):
                from IPython.core.display import display, HTML

                display(
                    HTML(
                        utils.horizontal_scrollable_div(
                            self.style.set_table_styles(utils.get_dataframe_styles())
                            .set_table_attributes("class=table")
                            .hide_index()
                            .render()
                        )
                    )
                )
                return None

            def __repr__(self):
                return "{} rows, {} columns".format(*self.shape)

        return FormattedDataFrame(df)

    def call(self, func, *args, sample_size=None, **kwargs):
        r"""
        Runs a custom function on dataframe

        func will receive the pandas dataframe (which represents the dataset) as an argument named 'df' by default.
        This can be overridden by specifying the dataframe argument name in a tuple (func, dataframe_name).

        Parameters
        ----------
        func: Union[callable, tuple]
            Custom function that takes pandas dataframe as input
            Alternatively a (callable, data) tuple where data is a string indicating the keyword of callable
            that expects the dataframe name
        args: iterable, optional
            Positional arguments passed into func
        sample_size: int, Optional
            To use a sampled dataframe
        kwargs: mapping, optional
            A dictionary of keyword arguments passed into func

        Returns
        -------
        func: function
            a plotting function that contains `*args` and `**kwargs`

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("classfication_data.csv"))
        >>> def f1(df):
        ...  return(sum(df), axis=0)
        >>> sum_ds = ds.call(f1)
        """

        data = "df"
        if isinstance(func, tuple):
            func, data = func
            if data in kwargs:
                raise ValueError(
                    "'%s' is both the data argument and a keyword argument" % data
                )

        if sample_size is None:
            # user has asked not to do sampling
            df = self.df.copy()
        else:
            df = self.df.sample(n=sample_size)
        kwargs[data] = df
        return func(*args, **kwargs)

    def set_target(self, target, type_discovery=True, target_type=None):
        """
        Returns a dataset tagged based on the type of target.

        Parameters
        ----------
        target: str
            name of the feature to use as target.
        type_discovery: bool
            This is set as True by default.
        target_type: type
            If provided, then the target will be typed with the provided value.

        Returns
        -------
        ds: ADSDataset
            tagged according to the type of the target column.

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("classfication_data.csv"))
        >>> ds_with_target= ds.set_target("target_class")
        """
        if target_type:
            target_series = self.sampled_df[target].astype(target_type)
        else:
            target_series = self.sampled_df[target]
        return get_dataset(
            self.df,
            self.sampled_df,
            target,
            infer_target_type(target, target_series, type_discovery),
            self.shape,
            **self.init_kwargs,
        )

    @deprecated("2.5.2", details="Instead use `to_pandas`.")
    def to_pandas_dataframe(
        self, filter=None, frac=None, include_transformer_pipeline=False
    ):
        return self.to_pandas(
            filter=filter,
            frac=frac,
            include_transformer_pipeline=include_transformer_pipeline,
        )

    def to_pandas(self, filter=None, frac=None, include_transformer_pipeline=False):
        """
        Returns a copy of the data as pandas.DataFrame, and a sklearn pipeline optionally that holds the
        transformations run so far on the data.

        The pipeline returned can be updated with the transformations done offline and passed along with the
        dataframe to Dataset.open API if the transformations need to be reproduced at the time of scoring.

        Parameters
        ----------
        filter: str, optional
            The query string to filter the dataframe, for example
            ds.to_pandas(filter="age > 50 and location == 'san francisco")
            See also https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html
        frac: float, optional
            fraction of original data to return.
        include_transformer_pipeline: bool, default: False
            If True, (dataframe, transformer_pipeline) is returned as a tuple

        Returns
        -------
        dataframe : pandas.DataFrame
            if include_transformer_pipeline is False.
        (data, transformer_pipeline): tuple of pandas.DataFrame and dataset.pipeline.TransformerPipeline
            if include_transformer_pipeline is True.

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("data.csv"))
        >>> ds_as_df = ds.to_pandas()

        Notes
        -----
        See also https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline
        """
        df = self.df.query(filter) if filter is not None else self.df.copy()
        if frac is not None:
            df = df.sample(frac=frac)
        return (
            (df, copy.deepcopy(self.transformer_pipeline))
            if include_transformer_pipeline
            else df
        )

    @deprecated("2.5.2", details="Instead use `to_dask`.")
    def to_dask_dataframe(
        self,
        filter=None,
        frac=None,
        npartitions=None,
        include_transformer_pipeline=False,
    ):
        return self.to_dask(
            filter=filter,
            frac=frac,
            npartitions=npartitions,
            include_transformer_pipeline=include_transformer_pipeline,
        )

    @runtime_dependency(module="dask.dataframe", short_name="dd")
    def to_dask(
        self,
        filter=None,
        frac=None,
        npartitions=None,
        include_transformer_pipeline=False,
    ):
        """
        Returns a copy of the data as dask.dataframe.core.DataFrame, and a sklearn pipeline optionally that holds the
        transformations run so far on the data.

        The pipeline returned can be updated with the transformations done offline and passed along with the
        dataframe to Dataset.open API if the transformations need to be reproduced at the time of scoring.

        Parameters
        ----------
        filter: str, optional
            The query string to filter the dataframe, for example
            ds.to_dask(filter="age > 50 and location == 'san francisco")
            See also https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html
        frac: float, optional
            fraction of original data to return.
        include_transformer_pipeline: bool, default: False
            If True, (dataframe, transformer_pipeline) is returned as a tuple.

        Returns
        -------
        dataframe : dask.dataframe.core.DataFrame
            if include_transformer_pipeline is False.
        (data, transformer_pipeline): tuple of dask.dataframe.core.DataFrame and dataset.pipeline.TransformerPipeline
            if include_transformer_pipeline is True.

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("data.csv"))
        >>> ds_dask = ds.to_dask()

        Notes
        -----
        See also http://docs.dask.org/en/latest/dataframe-api.html#dataframe and
        https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline

        """
        res = self.to_pandas(
            filter=filter,
            frac=frac,
            include_transformer_pipeline=include_transformer_pipeline,
        )
        return (
            (dd.from_pandas(res[0], npartitions=npartitions), res[1])
            if include_transformer_pipeline
            else dd.from_pandas(res, npartitions=npartitions)
        )

    @deprecated("2.5.2", details="Instead use `to_h2o`.")
    def to_h2o_dataframe(
        self, filter=None, frac=None, include_transformer_pipeline=False
    ):
        return self.to_h2o(
            filter=filter,
            frac=frac,
            include_transformer_pipeline=include_transformer_pipeline,
        )

    @runtime_dependency(module="h2o")
    def to_h2o(self, filter=None, frac=None, include_transformer_pipeline=False):
        """
        Returns a copy of the data as h2o.H2OFrame, and a sklearn pipeline optionally that holds the
        transformations run so far on the data.

        The pipeline returned can be updated with the transformations done offline and passed along with the
        dataframe to Dataset.open API if the transformations need to be reproduced at the time of scoring.

        Parameters
        ----------
        filter: str, optional
            The query string to filter the dataframe, for example
            ds.to_h2o(filter="age > 50 and location == 'san francisco")
            See also https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html
        frac: float, optional
            fraction of original data to return.
        include_transformer_pipeline: bool, default: False
            If True, (dataframe, transformer_pipeline) is returned as a tuple.

        Returns
        -------
        dataframe : h2o.H2OFrame
            if include_transformer_pipeline is False.
        (data, transformer_pipeline): tuple of  h2o.H2OFrame and dataset.pipeline.TransformerPipeline
            if include_transformer_pipeline is True.

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("data.csv"))
        >>> ds_as_h2o = ds.to_h2o()

        Notes
        -----
        See also https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline
        """
        res = self.to_pandas(
            filter=filter,
            frac=frac,
            include_transformer_pipeline=include_transformer_pipeline,
        )
        return (
            (h2o.H2OFrame(res[0]), res[1])
            if include_transformer_pipeline
            else h2o.H2OFrame(res)
        )

    @deprecated("2.5.2", details="Instead use `to_xgb`.")
    def to_xgb_dmatrix(
        self, filter=None, frac=None, include_transformer_pipeline=False
    ):
        return self.to_xgb(
            filter=filter,
            frac=frac,
            include_transformer_pipeline=include_transformer_pipeline,
        )

    @runtime_dependency(module="xgboost", install_from=OptionalDependency.BOOSTED)
    def to_xgb(self, filter=None, frac=None, include_transformer_pipeline=False):
        """
        Returns a copy of the data as xgboost.DMatrix, and a sklearn pipeline optionally that holds the
        transformations run so far on the data.

        The pipeline returned can be updated with the transformations done offline and passed along with the
        dataframe to Dataset.open API if the transformations need to be reproduced at the time of scoring.

        Parameters
        ----------
        filter: str, optional
            The query string to filter the dataframe, for example
            ds.to_xgb(filter="age > 50 and location == 'san francisco")
            See also https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html
        frac: float, optional
            fraction of original data to return.
        include_transformer_pipeline: bool, default: False
            If True, (dataframe, transformer_pipeline) is returned as a tuple.

        Returns
        -------
        dataframe : xgboost.DMatrix
            if include_transformer_pipeline is False.
        (data, transformer_pipeline): tuple of xgboost.DMatrix and dataset.pipeline.TransformerPipeline
            if include_transformer_pipeline is True.

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("data.csv"))
        >>> xgb_dmat = ds.to_xgb()

        Notes
        -----
        See also https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline
        """
        res = self.to_pandas(
            filter=filter,
            frac=frac,
            include_transformer_pipeline=include_transformer_pipeline,
        )
        df = res[0] if include_transformer_pipeline else res
        le = DataFrameLabelEncoder()
        df = le.fit_transform(df)
        if include_transformer_pipeline:
            res[1].add(le)
        xgb_matrix = xgboost.DMatrix(df)
        return (xgb_matrix, res[1]) if include_transformer_pipeline else xgb_matrix

    def sample(self, frac=None, random_state=utils.random_state):
        """
        Returns random sample of dataset.

        Parameters
        ----------
        frac : float, optional
            Fraction of axis items to return.
        random_state : int or ``np.random.RandomState``
            If int we create a new RandomState with this as the seed
            Otherwise we draw from the passed RandomState

        Returns
        -------
        sampled_dataset: ADSDataset
            An ADSDataset which was randomly sampled.

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("data.csv"))
        >>> ds_sample = ds.sample()
        """
        df = self.df.sample(frac=frac, random_state=random_state)
        return self._build_new_dataset(df)

    def drop_columns(self, columns):
        """
        Return new dataset with specified columns removed.

        Parameters
        ----------
        columns : str or list
            columns to drop.

        Returns
        -------
        dataset: same type as the caller
            a dataset with specified columns dropped.

        Raises
        ------
        ValidationError
            If any of the feature names is not found in the dataset.

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("data.csv"))
        >>> ds_smaller = ds.drop_columns(['col1', 'col2'])
        """
        self._validate_feature(columns)
        return self.drop(columns, axis=1)

    def assign_column(self, column, arg):
        """
        Return new dataset with new column or values of the existing column mapped according to input correspondence.

        Used for adding a new column or substituting each value in a column with another value, that may be derived from
        a function, a :class:`pandas.Series` or a :class:`pandas.DataFrame`.

        Parameters
        ----------
        column : str
            Name of the feature to update.
        arg : function, dict, Series or DataFrame
            Mapping correspondence.

        Returns
        -------
        dataset: same type as the caller
            a dataset with the specified column assigned.

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("data.csv"))
        >>> ds_same_size = ds.assign_column('target',lambda x:  x>15 if x not None)
        >>> ds_bigger = ds.assign_column('new_col', np.arange(ds.shape[0]))
        """
        target_name = (
            self.target.name if not utils.is_same_class(self, ADSDataset) else None
        )
        if isinstance(arg, Iterable) or isinstance(arg, ADSDataset):
            df = self.df.copy()
            if type(arg) == pd.DataFrame:
                col_to_add = arg
            elif type(arg) == ADSDataset:
                col_to_add = arg.df
            elif type(arg) == dict:
                col_to_add = pd.DataFrame.from_dict(arg)
            elif type(arg) in [list, np.ndarray]:
                col_to_add = pd.DataFrame(arg, columns=["new_col"])
            elif type(arg) == pd.Series:
                col_to_add = arg.rename("new_col").to_frame()
            elif utils._is_dask_dataframe(arg):
                col_to_add = arg.compute()
            elif utils._is_dask_series(arg):
                col_to_add = arg.compute().rename("new_col").to_frame()
            else:
                raise ValueError(
                    f"assign_column currently does not support arg of type {type(arg)}. Reformat "
                    f"as types: Pandas, numpy, list, or dict"
                )
            if column in df.columns:
                df = df.drop(columns=column)
            new_df = pd.concat([df, col_to_add], axis=1).rename(
                columns={"new_col": column}
            )
            return self._build_new_dataset(new_df)

        else:
            sampled_df = self.sampled_df.copy()
            df = self.df.copy()
            sampled_df[column] = sampled_df[column].apply(arg)
            df[column] = df[column].apply(arg)
            if column == target_name:
                target_type = get_feature_type(target_name, sampled_df[target_name])
                return self._build_new_dataset(
                    df, sampled_df, target=target_name, target_type=target_type
                )
            else:
                return self._build_new_dataset(
                    df,
                    sampled_df,
                    target=target_name,
                    target_type=self.target.type
                    if target_name != column and target_name is not None
                    else None,
                )

    def rename_columns(self, columns):
        """
        Returns a new dataset with altered column names.

        dict values must be unique (1-to-1). Labels not contained in a dict will be left as-is.
        Extra labels listed don't throw an error.

        Parameters
        ----------
        columns: dict-like or function or list of str
            dict to rename columns selectively, or list of names to rename all columns, or a function like
            str.upper

        Returns
        -------
        dataset: same type as the caller
            A dataset with specified columns renamed.

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("data.csv"))
        >>> ds_renamed = ds.rename_columns({'col1': 'target'})
        """
        if isinstance(columns, list):
            assert len(columns) == len(
                self.columns.values
            ), "columns length do not match the dataset"
            columns = dict(zip(self.columns.values, columns))
        return self.rename(columns=columns)

    def set_name(self, name):
        """
        Sets name for the dataset.

        This name will be used to filter the datasets returned by ds.list() API.
        Calling this API is optional. By default name of the dataset is set to empty.

        Parameters
        ----------
        name: str
            Name of the dataset.

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("data1.csv"))
        >>> ds_renamed = ds.set_name("dataset1")
        """
        self.name = name

    def set_description(self, description):
        """
        Sets description for the dataset.

        Give your dataset a description.

        Parameters
        ----------
        description: str
            Description of the dataset.

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("data1.csv"))
        >>> ds_renamed = ds.set_description("dataset1 is from "data1.csv"")
        """
        self.description = description

    def snapshot(self, snapshot_dir=None, name="", storage_options=None):
        """
        Snapshot the dataset with modifications made so far.

        Optionally  caller can invoke ds.set_name() before saving to identify the dataset uniquely at the time of
        using ds.list().

        The snapshot can be reloaded by providing the URI returned by this API to DatasetFactory.open()

        Parameters
        ----------
        snapshot_dir: str, optional
            Directory path under which dataset snapshot will be created.
            Defaults to snapshots_dir set using DatasetFactory.set_default_storage().
        name: str, optional, default: ""
            Name to uniquely identify the snapshot using DatasetFactory.list_snapshots().
            If not provided, an auto-generated name is used.
        storage_options: dict, optional
            Parameters passed on to the backend filesystem class.
            Defaults to storage_options set using DatasetFactory.set_default_storage().

        Returns
        -------
        p_str: str
            the URI to access the snapshotted dataset.

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("data.csv"))
        >>> ds_uri = ds.snapshot()
        """
        if snapshot_dir is None:
            import ads.dataset.factory as factory

            snapshot_dir = factory.default_snapshots_dir
            if snapshot_dir is None:
                raise ValueError(
                    "Specify snapshot_dir or use DatasetFactory.set_default_storage() to set default \
                                  storage options"
                )
            else:
                logger.info("Using default snapshots dir %s" % snapshot_dir)
        name = self._get_unique_name(name)
        if not snapshot_dir.endswith("/"):
            snapshot_dir = snapshot_dir + "/"
        parquet_file = "%s%s.parquet" % (snapshot_dir, name)
        os.makedirs(snapshot_dir, exist_ok=True)
        if storage_options is None and parquet_file[:3] == "oci":
            import ads.dataset.factory as factory

            storage_options = factory.default_storage_options
            logger.info("Using default storage options.")

        return helper.write_parquet(
            path=parquet_file,
            data=self.df,
            metadata_dict={
                "metadata": self.feature_types,
                "transformer": self.transformer_pipeline,
            },
            storage_options=storage_options,
        )

    def to_csv(self, path, storage_options=None, **kwargs):
        """
        Save the materialized dataframe to csv file.

        Parameters
        ----------
        path: str
            Location to write to. If there are more than one partitions in df, should include a glob character to
            expand into a set of file names, or provide a `name_function=parameter`.
            Supports protocol specifications such as `"oci://"`, `"s3://"`.
        storage_options: dict, optional
             Parameters passed on to the backend filesystem class.
             Defaults to storage_options set using DatasetFactory.set_default_storage().
        kwargs: dict, optional

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("data.csv"))
        >>> [ds_link] = ds.to_csv("my/path.csv")
        """
        if storage_options is None:
            import ads.dataset.factory as factory

            storage_options = factory.default_storage_options
            logger.info("Using default storage options")
        return self.df.to_csv(path, storage_options=storage_options, **kwargs)

    def to_parquet(self, path, storage_options=None, **kwargs):
        """
        Save data to parquet file.

        Parameters
        ----------
        path: str
            Location to write to. If there are more than one partitions in df, should include a glob character to
            expand into a set of file names, or provide a `name_function=parameter`.
            Supports protocol specifications such as `"oci://"`, `"s3://"`.
        storage_options: dict, optional
             Parameters passed on to the backend filesystem class.
             Defaults to storage_options set using DatasetFactory.set_default_storage().
        kwargs: dict, optional

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("data.csv"))
        >>> ds.to_parquet("my/path")
        """
        if storage_options is None:
            import ads.dataset.factory as factory

            storage_options = factory.default_storage_options
            logger.info("Using default storage options")
        return self.df.to_parquet(path, storage_options=storage_options, **kwargs)

    def to_json(self, path, storage_options=None, **kwargs):
        """
        Save data to JSON files.

        Parameters
        ----------
        path: str
            Location to write to. If there are more than one partitions in df, should include a glob character to
            expand into a set of file names, or provide a `name_function=parameter`.
            Supports protocol specifications such as `"oci://"`, `"s3://"`.
        storage_options: dict, optional
            Parameters passed on to the backend filesystem class.
            Defaults to storage_options set using DatasetFactory.set_default_storage().
        kwargs: dict, optional

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("data.csv"))
        >>> ds.to_json("my/path.json")
        """
        if storage_options is None:
            import ads.dataset.factory as factory

            storage_options = factory.default_storage_options
            logger.info("Using default storage options")

        return self.df.to_json(path, storage_options=storage_options, **kwargs)

    def to_hdf(
        self, path: str, key: str, storage_options: dict = None, **kwargs
    ) -> str:
        """
        Save data to Hierarchical Data Format (HDF) files.

        Parameters
        ----------
        path : string
            Path to a target filename.
        key : string
            Datapath within the files.
        storage_options: dict, optional
            Parameters passed to the backend filesystem class.
            Defaults to storage_options set using DatasetFactory.set_default_storage().
        kwargs: dict, optional

        Returns
        -------
        str
            The filename of the HDF5 file created.

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("data.csv"))
        >>> ds.to_hdf(path="my/path.h5", key="df")
        """
        if storage_options is None:
            import ads.dataset.factory as factory

            storage_options = factory.default_storage_options
            logger.info("Using default storage options")

        with pd.HDFStore(
            "memory",
            mode="w",
            driver="H5FD_CORE",
            driver_core_backing_store=0,
        ) as hdf_store:
            hdf_store.put(key, self.df, format=kwargs.get("hdf5_format", "fixed"))
            data = hdf_store._handle.get_file_image()

        new_path = (
            path.replace("*", "0")
            if path[-3:] == ".h5"
            else path.replace("*", "0") + ".h5"
        )

        with fsspec.open(
            urlpath=new_path, mode="wb", storage_options=storage_options, **kwargs
        ) as fo:
            fo.write(data)

        return new_path

    @runtime_dependency(module="fastavro", install_from=OptionalDependency.DATA)
    def to_avro(self, path, schema=None, storage_options=None, **kwargs):
        """
        Save data to Avro files.
        Avro is a remote procedure call and data serialization framework developed within Apache's Hadoop project. It
        uses JSON for defining data types and protocols, and serializes data in a compact binary format.

        Parameters
        ----------
        path : string
            Path to a target filename.  May contain a ``*`` to denote many filenames.
        schema : dict
            Avro schema dictionary, see below.
        storage_options: dict, optional
            Parameters passed to the backend filesystem class.
            Defaults to storage_options set using DatasetFactory.set_default_storage().
        kwargs: dict, optional
            See https://fastavro.readthedocs.io/en/latest/writer.html

        Notes
        -----
        Avro schema is a complex dictionary describing the data,
        see https://avro.apache.org/docs/1.8.2/gettingstartedpython.html#Defining+a+schema
        and https://fastavro.readthedocs.io/en/latest/writer.html.
        Its structure is as follows::

            {'name': 'Test',
            'namespace': 'Test',
            'doc': 'Descriptive text',
            'type': 'record',
            'fields': [
                {'name': 'a', 'type': 'int'},
            ]}

        where the "name" field is required, but "namespace" and "doc" are optional
        descriptors; "type" must always be "record". The list of fields should
        have an entry for every key of the input records, and the types are
        like the primitive, complex or logical types of the Avro spec
        (https://avro.apache.org/docs/1.8.2/spec.html).

        Examples
        --------
        >>> import pandas
        >>> import fastavro
        >>> with open("data.avro", "rb") as fp:
        >>>     reader = fastavro.reader(fp)
        >>>     records = [r for r in reader]
        >>>     df = pandas.DataFrame.from_records(records)
        >>> ds = ADSDataset.from_dataframe(df)
        >>> ds.to_avro("my/path.avro")
        """
        # Get the row by row formatting
        data_row_by_row = []
        for i, row in self.df.iterrows():
            data_row_by_row.append(row.to_dict())
        # Try to auto-generate schema
        if schema is None:
            avro_types = self._convert_dtypes_to_avro_types()
            schema = {"name": self.name, "doc": self.description, "type": "record"}
            fields = []
            ## Add vars
            for col, dtype in avro_types:
                fields.append({"name": col, "type": ["null", dtype]})
            schema["fields"] = fields

        parsed_schema = fastavro.parse_schema(schema=schema)
        new_path = (
            path.replace("*", "0")
            if path[-5:] == ".avro"
            else path.replace("*", "0") + ".avro"
        )
        with fsspec.open(
            new_path, "wb", storage_options=storage_options, **kwargs
        ) as fo:
            fastavro.writer(fo, parsed_schema, data_row_by_row)
        return new_path

    def _convert_dtypes_to_avro_types(self):
        avro_types = []
        for name, dtype in zip(self.dtypes.index, self.dtypes.values):
            if dtype == np.int64:
                avro_dtype = "long"
            elif "int" in str(dtype):
                avro_dtype = "int"
            elif dtype == np.float64:
                avro_dtype = "double"
            elif "float" in str(dtype):
                avro_dtype = "float"
            elif dtype == np.bool_:
                avro_dtype = "boolean"
            else:
                avro_dtype = "string"
            avro_types.append((name, avro_dtype))
        return avro_types

    def astype(self, types):
        """
        Convert data type of features.

        Parameters
        ----------
        types: dict
            key is the existing feature name
            value is the data type to which the values of the feature should be converted.
            Valid data types: All numpy datatypes (Example: np.float64, np.int64, ...)
            or one of categorical, continuous, ordinal or datetime.

        Returns
        -------
        updated_dataset: `ADSDataset`
            an ADSDataset with new data types

        Examples
        --------
        >>> import pandas as pd
        >>> ds = ADSDataset.from_dataframe(pd.read_csv("data.csv"))
        >>> ds_reformatted = ds.astype({"target": "categorical"})
        """
        return self.__getattr__("astype")(helper.map_types(types))

    def merge(self, data, **kwargs):
        """
        Merges this dataset with another ADSDataset or pandas dataframe.

        Parameters
        ----------
        data : Union[ADSDataset, pandas.DataFrame]
            Data to merge.
        kwargs : dict, optional
            additional keyword arguments that would be passed to underlying dataframe's merge API.

        Examples
        --------
        >>> import pandas as pd
        >>> df1 = pd.read_csv("data1.csv")
        >>> df2 = pd.read_csv("data2.csv")
        >>> ds = ADSDataset.from_dataframe(df1.merge(df2))
        >>> ds_12 = ds1.merge(ds2)
        """
        assert isinstance(data, pd.DataFrame) or isinstance(
            data, ADSDataset
        ), "Can only merge datasets if they are of the types pandas or ads"
        df = self.df.merge(data.df if isinstance(data, ADSDataset) else data, **kwargs)
        return self._build_new_dataset(df, progress=utils.get_progress_bar(3))

    """
    Internal methods
    """

    def __getattr__(self, item):
        attr = getattr(self.df, item)
        if callable(attr):
            return self._apply(attr)
        else:
            return attr

    def __getitem__(self, key):
        if isinstance(key, str) or isinstance(key, (tuple, str)):
            return self.df[key]
        else:
            return self._build_new_dataset(self.df[key])

    def _apply(self, func):
        def df_func(*args, _new_target=None, **kwargs):
            has_dataframe_arg = False
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, ADSDataset) or isinstance(arg, pd.DataFrame):
                    has_dataframe_arg = True
                    # convert any argument that is of type ADSDataset to dataframe. This is useful in delegate calls
                    # like dataset1.concat(dataset2)
                    args[i] = arg.df if isinstance(arg, ADSDataset) else arg

            result = func(*args, **kwargs)

            # return the response as such  if the the result is not a dataframe and it is a read function such as head
            if (
                isinstance(result, pd.DataFrame)
                and func.__name__ not in self.df_read_functions
            ):
                target_name = None
                target_sample_val = None
                if not utils.is_same_class(self, ADSDataset):
                    target_name = (
                        self.target.name if _new_target is None else _new_target
                    )
                    target_sample_val = (
                        self.sampled_df[self.target.name].dropna().values[0]
                    )

                df = result
                n = len(df)
                trans_df = None
                transformed = False
                transformers = []

                # The sampled dataframe needs to be re-generated when this operation involves another dataframe.
                # Also, this kind of transformations cannot be reproduced at the time of scoring
                if not has_dataframe_arg:
                    ft = DataFrameTransformer(
                        func_name=func.__name__,
                        target_name=target_name,
                        target_sample_val=target_sample_val,
                        args=args,
                        kw_args=kwargs,
                    ).fit(result)
                    # transformed is set to false if the method fails to run on pandas dataframe. In this case a new
                    # sampled dataframe is added
                    trans_df, transformed = ft._transform(self.sampled_df.copy())
                    # if the dataset length changes as a result of transformation, these operations need not be added to
                    # pipeline as they do not need to be reproduced at the time of scoring.
                    transformers = (func.__name__, ft) if n == self.shape[0] else []

                init_kwargs = self.init_kwargs.copy()
                if func.__name__ == "astype":
                    if "types" in init_kwargs:
                        init_kwargs["types"] = init_kwargs["types"] + args[0]
                    else:
                        init_kwargs["types"] = args[0]

                # if the transforming function is not supported by pandas dataframe, we need to sample the dask
                # dataframe again to get a new representation
                return self._build_new_dataset(
                    df,
                    sampled_df=df,
                    target=target_name,
                    target_type=TypeDiscoveryDriver().discover(
                        target_name, df[target_name]
                    )
                    if target_name is not None and target_name in df
                    else None,
                    sample=not transformed,
                    transformers=transformers,
                    **init_kwargs,
                )
            return result

        return df_func

    def _handle_key_error(self, args):
        raise ValidationError("Column %s does not exist in data" % str(args))

    def _build_new_dataset(
        self,
        df,
        sampled_df=None,
        target=None,
        target_type=None,
        transformers=[],
        sample=False,
        progress=DummyProgressBar(),
        n=None,
        **init_kwargs,
    ):

        prev_doc_mode = utils.is_documentation_mode()

        set_documentation_mode(False)

        init_kwargs = (
            self.init_kwargs
            if init_kwargs is None or len(init_kwargs) == 0
            else init_kwargs.copy()
        )
        n = len(df) if n is None else n

        # re-calculate sample df if not provided
        if sampled_df is None or sample:
            if progress:
                progress.update("Sampling data")
            sampled_df = generate_sample(
                df,
                n,
                DatasetDefaults.sampling_confidence_level,
                DatasetDefaults.sampling_confidence_interval,
                **init_kwargs,
            )
        else:
            if progress:
                progress.update()
        shape = (n, len(df.columns))
        if not utils.is_same_class(self, ADSDataset) and target is None:
            target = self.target.name

        set_documentation_mode(prev_doc_mode)

        # return a  ADSDataset object if the target has been removed from the dataframe
        if target in sampled_df.columns:
            if progress:
                progress.update("Building new dataset")
            target_type = self.target.type if target_type is None else target_type

            new_ds = get_dataset(
                df,
                sampled_df,
                target,
                target_type,
                shape,
                progress=progress,
                **init_kwargs,
            )

            new_ds.transformer_pipeline = self._update_transformer_pipeline(
                transformers
            )
            return new_ds
        else:
            if target is not None and not isinstance(progress, DummyProgressBar):
                logger.info(
                    "The target variable does not exist. Use `set_target()` to specify the target."
                )
            if progress:
                progress.update("Building the dataset with no target.")
            dsp = ADSDataset(
                df,
                sampled_df,
                shape,
                progress=progress,
                interactive=False,
                **init_kwargs,
            )
            dsp.transformer_pipeline = self._update_transformer_pipeline(transformers)
            return dsp

    def _validate_feature(self, feature_names):
        if np.isscalar(feature_names):
            feature_names = [feature_names]
        for feature in feature_names:
            if feature not in self.df.columns:
                self._handle_key_error(feature)

    def _update_transformer_pipeline(self, steps=[]):
        if isinstance(steps, tuple):
            steps = [steps]
        if steps is None or len(steps) == 0:
            return copy.deepcopy(self.transformer_pipeline)
        if self.transformer_pipeline is not None:
            transformer_pipeline = TransformerPipeline(
                steps=self.transformer_pipeline.steps + steps
            )
        else:
            transformer_pipeline = TransformerPipeline(steps=steps)
        return transformer_pipeline

    def _get_unique_name(self, name):
        id = (
            uuid.uuid4().hex + "_" + datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        )
        if name == "":
            return id
        return name + "_" + id

    def corr(
        self,
        correlation_methods: Union[list, str] = "pearson",
        frac: float = 1.0,
        sample_size: float = 1.0,
        nan_threshold: float = 0.8,
        overwrite: bool = None,
        force_recompute: bool = False,
    ):
        """
        Compute pairwise correlation of numeric and categorical columns, output a matrix or a list of matrices computed
        using the correlation methods passed in.

        Parameters
        ----------
        correlation_methods: Union[list, str], default to 'pearson'

            - 'pearson': Use Pearson's Correlation between continuous features,
            - 'cramers v': Use Cramer's V correlations between categorical features,
            - 'correlation ratio': Use Correlation Ratio Correlation between categorical and continuous features,
            - 'all': Is equivalent to ['pearson', 'cramers v', 'correlation ratio'].

            Or a list containing any combination of these methods, for example, ['pearson', 'cramers v'].
        frac:
            Is deprecated and replaced by sample_size.
        sample_size: float, defaults to 1.0. Float, Range -> (0, 1]
            What fraction of the data should be used in the calculation?
        nan_threshold: float, default to 0.8, Range -> [0, 1]
            Only compute a correlation when the proportion of the values, in a column, is less than or equal to nan_threshold.
        overwrite:
            Is deprecated and replaced by force_recompute.
        force_recompute: bool, default to be False

            - If False, it calculates the correlation matrix if there is no cached correlation matrix. Otherwise,
              it returns the cached correlation matrix.
            - If True, it calculates the correlation matrix regardless whether there is cached result or not.

        Returns
        -------
        correlation: Union[list, pandas.DataFrame]
            The pairwise correlations as a matrix (DataFrame) or list of matrices
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
        if sample_size > 1 or sample_size <= 0:
            logger.error("`sample_size` must to be in the range of (0, 1].")
            return
        if nan_threshold > 1 or nan_threshold < 0:
            logger.error("`nan_threshold` must be between 0 and 1 (exclusive).")
            return
        return self._compute_correlation(
            frac=sample_size,
            threshold=nan_threshold,
            force_recompute=force_recompute,
            correlation_methods=correlation_methods,
        )

    def _compute_correlation(
        self,
        frac=1.0,
        threshold=0.8,
        include_n_features=16,
        correlation_methods="pearson",
        force_recompute=False,
    ):
        """
        returns a list of correlation matrix/matrices
        """

        # validate the correlation methods
        correlation_methods = _validate_correlation_methods(correlation_methods)

        # if users choose to sample a frac of the data
        corr_df = self.df if not frac else self.df.sample(frac=frac)

        # return columns by type and filter by threshold
        threshold = threshold * 100
        feature_types_df = pd.DataFrame.from_dict(self.feature_types).T

        # reduce the dim of wide data
        n_rows, n_columns = self.shape

        is_wide_dataset = n_columns >= N_Features_Wide_Dataset

        if is_wide_dataset and include_n_features:
            corr_df, feature_types_df = self._reduce_dim_for_wide_dataset(
                corr_df, feature_types_df, include_n_features
            )

        categorical_columns, continuous_columns, _ = _get_columns_by_type(
            feature_types_df, threshold=threshold
        )

        # get the correlation
        correlation_list = []
        for method in correlation_methods:
            correlation_list.append(
                self._return_correlation(
                    corr_df,
                    method,
                    categorical_columns,
                    continuous_columns,
                    force_recompute,
                )
            )
        return correlation_list[0] if len(correlation_list) == 1 else correlation_list

    def _calc_pearson(self, df: pd.DataFrame, continuous_columns: list) -> pd.DataFrame:
        self._pearson = (
            df[continuous_columns].corr()
            if len(continuous_columns) > 1
            else pd.DataFrame()
        )
        return self._pearson

    def _calc_cramers_v(
        self, df: pd.DataFrame, categorical_columns: list
    ) -> pd.DataFrame:
        self._cramers_v = _cat_vs_cat(df, categorical_columns)
        return self._cramers_v

    def _calc_correlation_ratio(
        self,
        df: pd.core.frame.DataFrame,
        categorical_columns: list,
        continuous_columns: list,
    ) -> pd.DataFrame:
        self._correlation_ratio = _cat_vs_cts(
            df, categorical_columns, continuous_columns
        )
        return self._correlation_ratio

    def _return_correlation(
        self,
        corr_df,
        method,
        categorical_columns,
        continuous_columns,
        force_recompute,
    ):
        if not force_recompute and hasattr(self, "_" + "_".join(method.split())):
            logger.info(
                f"Using cached results for {method} correlation. Use"
                " `force_recompute=True` to override."
            )
            return getattr(self, "_" + "_".join(method.split()))
        else:
            if method == "pearson":
                self._calc_pearson(corr_df, continuous_columns)
                return self._pearson
            elif method == "cramers v":
                self._calc_cramers_v(corr_df, categorical_columns)
                return self._cramers_v
            elif method == "correlation ratio":
                self._calc_correlation_ratio(
                    corr_df, categorical_columns, continuous_columns
                )
                return self._correlation_ratio
            else:
                raise ValueError(f"The {method} method is not supported.")

    @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
    def _reduce_dim_for_wide_dataset(
        self, corr_df: pd.DataFrame, feature_types_df: pd.DataFrame, include_n_features
    ):
        min_cores_for_correlation = 2
        n_rows, n_columns = self.shape

        from IPython.core.display import display, HTML

        if utils.get_cpu_count() <= min_cores_for_correlation:
            msg = (
                f"Not attempting to calculate correlations, too few cores ({utils.get_cpu_count()}) "
                f"for wide dataset ({n_columns} columns)"
            )
            display(HTML(f"<li>{msg}</li>"))
            return None, None

        display(HTML(f"<li>detected wide dataset ({n_columns} columns)</li>"))

        if "target" in self.__dict__:
            display(
                HTML(
                    f"<li>feature reduction using mutual information (max {include_n_features} columns)</li>"
                )
            )
            logger.info("Set `include_n_features=None` to include all features.")
            corr_sampled_df = self._find_feature_subset(
                self.sampled_df, self.target.name, include_n_features=include_n_features
            )
            corr_df, feature_types_df = self._update_dataframes(
                corr_sampled_df, corr_df, feature_types_df
            )
        else:
            #
            # in the absense of a target we simply use the first_n
            #
            logger.info(
                f"To include the first {include_n_features} features based on the feature"
                f"importance, use `.set_target`()."
            )
            feature_types_df = feature_types_df[
                (feature_types_df.index.isin(corr_df.columns.values))
                & feature_types_df.type.isin(
                    ["categorical", "ordinal", "continuous", "zipcode"]
                )
            ]
            corr_df = corr_df[feature_types_df.index[:include_n_features]]
            feature_types_df = feature_types_df.iloc[:include_n_features, :]
        return corr_df, feature_types_df

    def _update_dataframes(self, corr_sampled_df, corr_df, feature_types_df):
        """
        update the dataframe and feature types based on the reduced dataframe
        """
        cols = corr_sampled_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index(self.target.name)))
        corr_df_reduced = corr_df[[*cols]]
        feature_types_df_reduced = feature_types_df[feature_types_df.index.isin(cols)]
        return corr_df_reduced, feature_types_df_reduced

    def show_corr(
        self,
        frac: float = 1.0,
        sample_size: float = 1.0,
        nan_threshold: float = 0.8,
        overwrite: bool = None,
        force_recompute: bool = False,
        correlation_target: str = None,
        plot_type: str = "heatmap",
        correlation_threshold: float = -1,
        correlation_methods="pearson",
        **kwargs,
    ):
        """
        Show heatmap or barplot of pairwise correlation of numeric and categorical columns, output three tabs
        which are heatmap or barplot of correlation matrix of numeric columns vs numeric columns using pearson
        correlation method, categorical columns vs categorical columns using Cramer's V method,
        and numeric vs categorical columns, excluding NA/null values and columns which have more than
        80% of NA/null values. By default, only 'pearson' correlation is calculated and shown in the first tab.
        Set correlation_methods='all' to show all correlation charts.

        Parameters
        ----------
        frac: Is superseded by sample_size
        sample_size: float, defaults to 1.0. Float, Range -> (0, 1]
            What fraction of the data should be used in the calculation?
        nan_threshold: float, defaults to 0.8, Range -> [0, 1]
            In the default case, it will only calculate the correlation of the columns which has less than or equal to
            80% of missing values.
        overwrite:
            Is deprecated and replaced by force_recompute.
        force_recompute: bool, default to be False.

            - If False, it calculates the correlation matrix if there is no cached correlation matrix. Otherwise,
              it returns the cached correlation matrix.
            - If True, it calculates the correlation matrix regardless whether there is cached result or not.

        plot_type: str, default to "heatmap"
            It can only be "heatmap" or "bar". Note that if "bar" is chosen, correlation_target also has to be set and
            the bar chart will only show the correlation values of the pairs which have the target in them.
        correlation_target: str, default to Non
            It can be any columns of type continuous, ordinal, categorical or zipcode. When correlation_target is set,
            only pairs that contains correlation_target will show.
        correlation_threshold: float, default to -1
            It can be any number between -1 and 1.
        correlation_methods: Union[list, str], defaults to 'pearson'

            - 'pearson': Use Pearson's Correlation between continuous features,
            - 'cramers v': Use Cramer's V correlations between categorical features,
            - 'correlation ratio': Use Correlation Ratio Correlation between categorical and continuous features,
            - 'all': Is equivalent to ['pearson', 'cramers v', 'correlation ratio'].

            Or a list containing any combination of these methods, for example, ['pearson', 'cramers v'].

        Returns
        -------
        None
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
                "<code>frac</code> is deprecated. Use <code>sample_size</code> instead.",
                DeprecationWarning,
            )
            if sample_size == 1.0:
                sample_size = deprecate_frac

        feature_types_df = pd.DataFrame.from_dict(self.feature_types).loc["type", :]
        features_list = list(
            feature_types_df[
                feature_types_df.isin(
                    ["categorical", "zipcode", "continuous", "ordinal"]
                )
            ].index
        )
        if plot_type not in ["heatmap", "bar"]:
            raise ValueError('plot_type has to be "heatmap" ' 'or "bar"')

        if plot_type == "bar" and correlation_target is None:
            raise ValueError('correlation_target has to be set when plot_type="bar".')

        if correlation_target:
            if correlation_target not in features_list:
                raise ValueError(
                    "correlation_target has to be in {}.".format(features_list)
                )

        force_recompute = deprecate_variable(
            overwrite,
            force_recompute,
            f"<code>overwrite=None</code> is deprecated. Use <code>force_recompute</code> instead.",
            DeprecationWarning,
        )

        plot_correlation_heatmap(
            ds=self,
            frac=sample_size,
            force_recompute=force_recompute,
            correlation_target=correlation_target,
            plot_type=plot_type,
            correlation_threshold=correlation_threshold,
            nan_threshold=nan_threshold,
            correlation_methods=correlation_methods,
            **kwargs,
        )

    @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
    @runtime_dependency(module="ipywidgets", install_from=OptionalDependency.NOTEBOOK)
    def show_in_notebook(
        self,
        correlation_threshold=-1,
        selected_index=0,
        sample_size=0,
        visualize_features=True,
        correlation_methods="pearson",
        **kwargs,
    ):
        """
        Provide visualization of dataset.

        - Display feature distribution. The data table display will show a maximum of 8 digits,
        - Plot the correlation between the dataset features (as a heatmap) only when all the features are
          continuous or ordinal,
        - Display data head.

        Parameters
        ----------
        correlation_threshold : int, default -1
            The correlation threshold to select, which only show features that have larger or equal
            correlation values than the threshold.
        selected_index: int, str, default 0
            The displayed output is stacked into an accordion widget, use selected_index to force the display to open
            a specific element, use the (zero offset) index or any prefix string of the name (eg, 'corr' for
            correlations)
        sample_size: int, default 0
            The size (in rows) to sample for visualizations
        visualize_features: bool default False
            For the "Features" section control if feature visualizations are shown or not. If not only
            a summary of the numeric statistics is shown. The numeric statistics are also always shows
            for wide (>64 features) datasets
        correlation_methods: Union[list, str], default to 'pearson'

            - 'pearson': Use Pearson's Correlation between continuous features,
            - 'cramers v': Use Cramer's V correlations between categorical features,
            - 'correlation ratio': Use Correlation Ratio Correlation between categorical and continuous features,
            - 'all': Is equivalent to ['pearson', 'cramers v', 'correlation ratio'].

            Or a list containing any combination of these methods, for example, ['pearson', 'cramers v'].
        """

        if not utils.is_notebook():
            print("show_in_notebook called but not in notebook environment")
            return

        n_rows, n_columns = self.shape

        min_sample_size = 10000
        if sample_size == 0:
            sub_samp_size = len(self.sampled_df)
            sub_samp_df = self.sampled_df
        else:
            sub_samp_size = max(min(sample_size, len(self.sampled_df)), min_sample_size)
            sub_samp_df = self.sampled_df.sample(n=sub_samp_size)

        html_summary = ""
        if self.name:
            html_summary += "<h1>Name: %s</h1>" % (self.name)

        # dataset type (problem type)
        html_summary += "<h3>Type: %s</h3>" % self.__class__.__name__

        if self.description:
            html_summary += "<pre>%s</pre>" % self.description
            html_summary += "<hr>"

        html_summary += "<h3>{:,} Rows, {:,} Columns</h3>".format(n_rows, n_columns)
        html_summary += "<h4>Column Types:</h4><UL>"

        for group in Counter(
            [self.feature_types[k].meta_data["type"] for k in self.feature_types]
        ).most_common():
            html_summary += "<LI><b>%s:</b> %d features" % (group[0], group[1])

        html_summary += "</UL>"

        html_summary += """
                <p><b>
                    Note: Visualizations use a sampled subset of the dataset, this is to
                    improve plotting performance. The sample size is calculated to be statistically
                    significant within the confidence level: {} and confidence interval: {}.

                    The sampled data has {:,} rows
                    </b>
                </p>

                <ul>
                    <li>The confidence <i>level</i> refers to the long-term success rate of the
                    method, that is, how often this type of interval will capture the parameter
                    of interest.
                    </li>

                    <li>A specific confidence <i>interval</i> gives a range of plausible values for
                    the parameter of interest
                    </li>
                </ul>

            """.format(
            DatasetDefaults.sampling_confidence_level,
            DatasetDefaults.sampling_confidence_interval,
            sub_samp_df.shape[0],
        )

        html_summary += "</UL>"

        from ipywidgets import widgets

        summary = widgets.HTML(html_summary)

        features = widgets.HTML()
        correlations = widgets.Output()
        warningz = widgets.HTML()

        warningz.value = "Analyzing for warnings..."
        features.value = "Calculating full statistical info..."

        # with correlations:
        #     display(HTML("<li>calculating...</li>"))

        accordion = widgets.Accordion(
            children=[summary, features, correlations, warningz]
        )
        accordion.set_title(0, "Summary")
        accordion.set_title(1, "Features")
        accordion.set_title(2, "Correlations")
        accordion.set_title(3, "Warnings")

        if isinstance(selected_index, str):
            # lookup by title
            possible_titles = [
                accordion.get_title(i) for i in range(len(accordion.children))
            ]
            for i, title in enumerate(possible_titles):
                if title.lower().startswith(selected_index.lower()):
                    selected_index = i
                    break

            if isinstance(selected_index, str):
                # failed to match a title
                logger.info(
                    "`selected_index` should be one of: {}.".format(
                        ", ".join(possible_titles)
                    )
                )
                selected_index = 0

        accordion.selected_index = selected_index

        is_wide_dataset = n_columns >= N_Features_Wide_Dataset

        #
        # set up dataframe to use for correlation calculations
        #

        self.df_stats = self._calculate_dataset_statistics(
            is_wide_dataset, [features, warningz]
        )

        with correlations:
            feature_types_df = pd.DataFrame.from_dict(self.feature_types).loc["type", :]
            if not is_wide_dataset:
                feature_types_df = feature_types_df[
                    self.df_stats["missing"] < len(self.df)
                ]

            frac = kwargs.pop("frac", 1.0)
            overwrite = kwargs.pop("overwrite", None)
            force_recompute = kwargs.pop("force_recompute", False)
            force_recompute = deprecate_variable(
                overwrite,
                force_recompute,
                f"<code>overwrite=None</code> is deprecated. Use <code>force_recompute</code> instead.",
                DeprecationWarning,
            )
            plot_type = kwargs.pop("plot_type", "heatmap")
            correlation_target = kwargs.pop("correlation_target", None)
            nan_threshold = kwargs.pop("nan_threshold", 0.8)
            self.show_corr(
                correlation_threshold=correlation_threshold,
                sample_size=frac,
                force_recompute=force_recompute,
                plot_type=plot_type,
                correlation_target=correlation_target,
                nan_threshold=nan_threshold,
                correlation_methods=correlation_methods,
                **kwargs,
            )

        from IPython.core.display import display

        display(accordion)

        # generate html for feature_distribution & warnings

        accordion.set_title(
            1, f"Features ({n_columns})"
        )  # adjust for datasets with target

        #
        # compute missing value statistics
        # not done for wide datasets
        #

        features.value = self._generate_features_html(
            is_wide_dataset,
            n_columns,
            self.df_stats,
            visualizations_follow=bool(visualize_features),
        )

        warningz.value = self._generate_warnings_html(
            is_wide_dataset, n_rows, n_columns, self.df_stats, warningz, accordion
        )

        if visualize_features and not is_wide_dataset:
            self._visualize_feature_distribution(features)

    def get_recommendations(self, *args, **kwargs):  # real signature may change
        """
        Returns user-friendly error message to set target variable before invoking this API.

        Parameters
        ----------
        kwargs

        Returns
        -------
        NotImplementedError
            raises NotImplementedError, if target parameter value not provided

        """
        raise NotImplementedError(
            "Please set the target using set_target() before invoking this API. See "
            "https://accelerated-data-science.readthedocs.io/en/latest/ads.dataset.html#ads.dataset.dataset.ADSDataset.set_target "
            "for the API usage."
        )

    def suggest_recommendations(self, *args, **kwargs):  # real signature may change
        """
        Returns user-friendly error message to set target variable before invoking this API.

        Parameters
        ----------
        kwargs

        Returns
        -------
        NotImplementedError
            raises NotImplementedError, if target parameter value not provided

        """
        raise NotImplementedError(
            "Please set the target using set_target() before invoking this API. See "
            "https://accelerated-data-science.readthedocs.io/en/latest/ads.dataset.html#ads.dataset.dataset.ADSDataset.set_target "
            "for the API usage."
        )
