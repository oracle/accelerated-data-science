#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The ADS accessor for the Pandas DataFrame.
The accessor will be initialized with the pandas object the user is interacting with.

Examples
--------
>>> from ads.feature_engineering.accessor.dataframe_accessor import ADSDataFrameAccessor
    >>> from ads.feature_engineering.feature_type.continuous import Continuous
    >>> from ads.feature_engineering.feature_type.creditcard import CreditCard
    >>> from ads.feature_engineering.feature_type.string import String
    >>> from ads.feature_engineering.feature_type.base import Tag
>>> df = pd.DataFrame({'Name': ['Alex'], 'CreditCard': ["4532640527811543"]})
>>> df.ads.feature_type
{'Name': ['string'], 'Credit Card': ['string']}
>>> df.ads.feature_type_description
          Column   Feature Type                        Description
------------------------------------------------------------------
0           Name         string    Type representing string values.
1    Credit Card         string    Type representing string values.
>>> df.ads.default_type
{'Name': 'string', 'Credit Card': 'string'}
>>> df.ads.feature_type = {'Name':['string', Tag('abc')]}
>>> df.ads.tags
{'Name': ['abc']}
>>> df.ads.feature_type = {'Credit Card':['credit_card']}
>>> df.ads.feature_select(include=['credit_card'])
                    Credit Card
-------------------------------
0	          4532640527811543
"""

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from ads.common.utils import DATA_SCHEMA_MAX_COL_NUM
from ads.data_labeling.mixin.data_labeling import DataLabelingAccessMixin
from ads.dataset.mixin.dataset_accessor import ADSDatasetAccessMixin
from ads.dbmixin.db_pandas_accessor import DBAccessMixin
from ads.feature_engineering import schema
from ads.feature_engineering.accessor.mixin.eda_mixin import EDAMixin
from ads.feature_engineering.accessor.mixin.feature_types_mixin import (
    ADSFeatureTypesMixin,
)
from ads.feature_engineering.feature_type.base import FeatureType
from pandas.core.dtypes.common import is_list_like


@pd.api.extensions.register_dataframe_accessor("ads")
class ADSDataFrameAccessor(
    ADSFeatureTypesMixin,
    EDAMixin,
    DBAccessMixin,
    DataLabelingAccessMixin,
    ADSDatasetAccessMixin
):
    """ADS accessor for the Pandas DataFrame.

    Attributes
    ----------
    columns: List[str]
        The column labels of the DataFrame.

    tags(self) -> Dict[str, str]
        Gets the dictionary of user defined tags for the dataframe.
    default_type(self) -> Dict[str, str]
        Gets the map of columns and associated default feature type names.
    feature_type(self) -> Dict[str, List[str]]
        Gets the list of registered feature types.
    feature_type_description(self) -> pd.DataFrame
        Gets the list of registered feature types in a DataFrame format.

    Methods
    -------
    sync(self, src: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame
        Syncs feature types of current DataFrame with that from src.
    feature_select(self, include: List[Union[FeatureType, str]] = None, exclude: List[Union[FeatureType, str]] = None) -> pd.DataFrame
        Gets the list of registered feature types in a DataFrame format.
    help(self, prop: str = None) -> None
        Provids docstring for affordable methods and properties.

    Examples
    --------
    >>> from ads.feature_engineering.accessor.dataframe_accessor import ADSDataFrameAccessor
    >>> from ads.feature_engineering.feature_type.continuous import Continuous
    >>> from ads.feature_engineering.feature_type.creditcard import CreditCard
    >>> from ads.feature_engineering.feature_type.string import String
    >>> from ads.feature_engineering.feature_type.base import Tag
    df = pd.DataFrame({'Name': ['Alex'], 'CreditCard': ["4532640527811543"]})
    >>> df.ads.feature_type
    {'Name': ['string'], 'Credit Card': ['string']}
    >>> df.ads.feature_type_description
              Column   Feature Type                        Description
    -------------------------------------------------------------------
    0           Name         string    Type representing string values.
    1    Credit Card         string    Type representing string values.
    >>> df.ads.default_type
    {'Name': 'string', 'Credit Card': 'string'}
    >>> df.ads.feature_type = {'Name':['string', Tag('abc')]}
    >>> df.ads.tags
    {'Name': ['abc']}
    >>> df.ads.feature_type = {'Credit Card':['credit_card']}
    >>> df.ads.feature_select(include=['credit_card'])
                       Credit Card
    ------------------------------
    0	          4532640527811543
    """

    def __init__(self, pandas_obj) -> None:
        """Initializes ADS Pandas DataFrame Accessor.

        Parameters
        ----------
        pandas_obj : pandas.DataFrame
            Pandas dataframe

        Raises
        ------
        ValueError
            If provided DataFrame has duplicate columns.
        """
        if len(set(pandas_obj.columns)) != len(pandas_obj.columns):
            raise ValueError(
                "Failed to initialize a DataFrame accessor. " "Duplicate column found."
            )
        self._obj = pandas_obj
        super().__init__()
        self.columns = self._obj.columns
        self._info = None

    def info(self) -> Any:
        """Gets information about the dataframe.

        Returns
        -------
        Any
            The information about the dataframe.
        """
        return self._info

    @property
    def _feature_type(self) -> Dict[str, List[FeatureType]]:
        """Gets the map of columns and associated feature types.
        Key is column name and value is list of feature types.
        """
        return {
            self._obj[col].name: self._obj[col].ads._feature_type for col in self._obj
        }

    @property
    def _default_type(self) -> Dict[str, FeatureType]:
        """Gets the map of columns and associated default feature types.
        Key is column name and value is a default feature type.
        """
        return {
            self._obj[col].name: self._obj[col].ads._default_type for col in self._obj
        }

    @property
    def tags(self) -> Dict[str, List[str]]:
        """Gets the dictionary of user defined tags for the dataframe. Key is column name
        and value is list of tag names.

        Returns
        -------
        Dict[str, List[str]]
            The map of columns and associated default tags.
        """
        return {self._obj[col].name: self._obj[col].ads.tags for col in self._obj}

    @property
    def default_type(self) -> Dict[str, str]:
        """Gets the map of columns and associated default feature type names.

        Returns
        -------
        Dict[str, str]
            The dictionary where key is column name and value is the name of default feature
            type.
        """
        return {k: v.name for k, v in self._default_type.items()}

    @property
    def feature_type(self) -> Dict[str, List[str]]:
        """Gets the list of registered feature types.

        Returns
        -------
        Dict[str, List[str]]
            The dictionary where key is column name and value is list of associated feature type
            names.
        """
        return {col.name: col.ads.feature_type for _, col in self._obj.items()}

    @property
    def feature_type_description(self) -> pd.DataFrame:
        """Gets the list of registered feature types in a DataFrame format.

        Returns
        -------
        :class:`pandas.DataFrame`

        Examples
        ________
        >>> df.ads.feature_type_description()
                  Column   Feature Type                         Description
        -------------------------------------------------------------------
        0           City         string    Type representing string values.
        1   Phone Number         string    Type representing string values.
        """
        result_df = pd.DataFrame([], columns=["Column", "Feature Type", "Description"])
        for col in self._obj:
            series_feature_type_df = self._obj[col].ads.feature_type_description
            series_feature_type_df.insert(0, "Column", col)
            result_df = result_df.append(series_feature_type_df)
        result_df.reset_index(drop=True, inplace=True)
        return result_df

    @feature_type.setter
    def feature_type(
        self, feature_type_map: Dict[str, List[Union[FeatureType, str]]]
    ) -> None:
        """Sets feature types for the DataFrame.

        Parameters
        ----------
        feature_type_map : Dict[str, List[Union[FeatureType, str]]]
            The map of feature types where key is column name and value is list of feature
            types.

        Returns
        -------
        None
            Nothing.
        """
        for col, feature_types in feature_type_map.items():
            self._obj[col].ads.feature_type = feature_types

    def sync(self, src: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """Syncs feature types of current DataFrame with that from src.

        Syncs feature types of current dataframe with that from src, where src
        can be a dataframe or a series. In either case, only columns with
        matched names are synced.

        Parameters
        ----------
        src: `pd.DataFrame` | `pd.Series`
            The source to sync from.

        Returns
        -------
        :class:`pandas.DataFrame`
            Synced dataframe.
        """
        for _, col in self._obj.items():
            col.ads.sync(src)

    def _extract_columns_of_target_types(
        self, target_types: List[Union[FeatureType, str]]
    ) -> List:
        """Returns all the column names that are of the target types from the
        feature_type dictionary.

        Parameters
        ----------
        target_types: list
            A list of target feature types, can be either feature type names of
            feature type class.

        Returns:
        -------
        List[str]
            The list of columns names.
        """
        columns = []
        target_types = (
            np.unique(
                [self._get_type(feature_type).name for feature_type in target_types]
            )
            if target_types is not None
            else None
        )
        for target_type in target_types:
            for name, feature_types in self.feature_type.items():
                if target_type in feature_types:
                    columns.append(name)
        return columns

    def feature_select(
        self,
        include: List[Union[FeatureType, str]] = None,
        exclude: List[Union[FeatureType, str]] = None,
    ) -> pd.DataFrame:
        """Returns a subset of the DataFrame’s columns based on the column feature_types.

        Parameters
        ----------
        include: List[Union[FeatureType, str]], optional
            Defaults to None. A list of FeatureType subclass or str to be included.
        exclude: List[Union[FeatureType, str]], optional
            Defaults to None. A list of FeatureType subclass or str to be excluded.

        Raises
        ------
        ValueError
            If both of include and exclude are empty
        ValueError
            If include and exclude are used simultaneously

        Returns
        -------
        :class:`pandas.DataFrame`
            The subset of the frame including the feature types in include and excluding
            the feature types in exclude.
        """
        if not (include or exclude):
            raise ValueError("at least one of include or exclude must be nonempty")

        if not is_list_like(include):
            include = (include,) if include is not None else ()
        if not is_list_like(exclude):
            exclude = (exclude,) if exclude is not None else ()

        # unify the feature types to str representation
        include = (
            np.unique([self._get_type(feature_type).name for feature_type in include])
            if include is not None
            else None
        )
        exclude = (
            np.unique([self._get_type(feature_type).name for feature_type in exclude])
            if exclude is not None
            else None
        )

        # convert the myriad valid dtypes object to a single representation
        include = frozenset(include)
        exclude = frozenset(exclude)

        # can't both include AND exclude!
        if not include.isdisjoint(exclude):
            raise ValueError(f"include and exclude overlap on {(include & exclude)}")

        # We raise when both include and exclude are empty
        # Hence, we can just shrink the columns we want to keep
        keep_these = np.full(self._obj.shape[1], True)

        columns = self._obj.columns

        if include:
            included_columns = self._extract_columns_of_target_types(include)
            keep_these &= columns.isin(included_columns)

        if exclude:
            excluded_columns = self._extract_columns_of_target_types(exclude)
            keep_these &= ~columns.isin(excluded_columns)

        return self._obj.loc[:, keep_these]

    def _add_feature_type(
        self, col: str, feature_type: Union[FeatureType, str]
    ) -> bool:
        """Adds a feature type

        Parameters
        ----------
        col : str
            The column name.
        feature_type : Union[FeatureType, str]
            The feature type to add.

        Returns
        -------
        bool
            Whether add succeeded.
        """
        if col not in self._obj.columns:
            raise ValueError(f"Column {col} is not found.")
        return self._obj[col].ads._add_feature_type(feature_type)

    def _remove_feature_type(
        self, col: str, feature_type: Union[FeatureType, str]
    ) -> None:
        """Removes a feature type

        Parameters
        ----------
        col : str
            column name
        feature_type : Union[FeatureType, str]
            feature type

        Returns
        -------
        None
            Nothing
        """
        if col not in self._obj.columns:
            raise ValueError(f"Column {col} is not found.")
        self._obj[col].ads._remove_feature_type(feature_type)

    def model_schema(self, max_col_num: int = DATA_SCHEMA_MAX_COL_NUM):
        """
        Generates schema from the dataframe.

        Parameters
        ----------
        max_col_num : int, optional. Defaults to 1000
            The maximum column size of the data that allows to auto generate schema.

        Examples
        --------
        >>> df = pd.read_csv('./orcl_attrition.csv', usecols=['Age', 'Attrition'])
        >>> schema = df.ads.model_schema()
        >>> schema
        Schema:
            - description: Attrition
            domain:
                constraints: []
                stats:
                count: 1470
                unique: 2
                values: String
            dtype: object
            feature_type: String
            name: Attrition
            required: true
            - description: Age
            domain:
                constraints: []
                stats:
                25%: 31.0
                50%: 37.0
                75%: 44.0
                count: 1470.0
                max: 61.0
                mean: 37.923809523809524
                min: 19.0
                std: 9.135373489136732
                values: Integer
            dtype: int64
            feature_type: Integer
            name: Age
            required: true
        >>> schema.to_dict()
        {'Schema': [{'dtype': 'object',
            'feature_type': 'String',
            'name': 'Attrition',
            'domain': {'values': 'String',
                'stats': {'count': 1470, 'unique': 2},
                'constraints': []},
            'required': True,
            'description': 'Attrition'},
            {'dtype': 'int64',
            'feature_type': 'Integer',
            'name': 'Age',
            'domain': {'values': 'Integer',
                'stats': {'count': 1470.0,
                'mean': 37.923809523809524,
                'std': 9.135373489136732,
                'min': 19.0,
                '25%': 31.0,
                '50%': 37.0,
                '75%': 44.0,
                'max': 61.0},
                'constraints': []},
            'required': True,
            'description': 'Age'}]}

        Returns
        -------
        ads.feature_engineering.schema.Schema
            data schema.

        Raises
        ------
        ads.feature_engineering.schema.DataSizeTooWide
            If the number of columns of input data exceeds `max_col_num`.
        """
        if max_col_num and len(self._obj.columns) > max_col_num:
            raise schema.DataSizeTooWide(
                data_col_num=len(self._obj.columns), max_col_num=max_col_num
            )

        sc = schema.Schema()
        for i, col in enumerate(self._obj.columns):
            domain = schema.Domain()
            try:
                domain = self._obj[col].ads.feature_domain()
            except:
                pass

            sc.add(
                schema.Attribute(
                    self._obj[col].dtype.name,
                    domain.values,
                    col,
                    domain=domain,
                    description=str(col),
                    required=bool(~self._obj[col].isnull().any()),
                    order=i,
                )
            )

        return sc

    def __getattr__(self, attr):
        attr_map = dict()
        for col in self._obj:
            try:
                val = self._obj[col].ads.__getattr__(attr)
            except:
                val = None  # if a column does not have the request attr, return None
            attr_map[col] = val

        if any(
            callable(x) for x in list(attr_map.values())
        ):  # check if attr is a callable, and if yes apply args to all cols.

            def func(*args, **kwargs):
                out = dict()
                for k, v in attr_map.items():
                    out[k] = v(*args, **kwargs) if v else None
                return out

            return func

        return attr_map
