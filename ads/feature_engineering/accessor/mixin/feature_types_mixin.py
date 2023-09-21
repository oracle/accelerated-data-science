#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents the ADS Feature Types Mixin class that extends
Pandas Series and Dataframe accessors.

Classes
-------
    ADSFeatureTypesMixin
        ADS Feature Types Mixin class that extends Pandas Series and Dataframe accessors.
"""
import inspect
from typing import Union

import pandas as pd
import tabulate
from ads.feature_engineering.feature_type.base import FeatureType
from ads.feature_engineering.feature_type_manager import (
    FeatureTypeManager as feature_type_manager,
)
from ads.feature_engineering.feature_type_manager import _feature_type_by_dtype


class ADSFeatureTypesMixin:
    """ADS Feature Types Mixin class that extends Pandas Series and DataFrame accessors.

    Methods
    -------
    warning_registered(cls) -> pd.DataFrame
        Lists registered warnings for registered feature types.
    validator_registered(cls) -> pd.DataFrame
        Lists registered validators for registered feature types.
    help(self, prop: str = None) -> None
        Help method that prints either a table of available properties or, given a property,
        returns its docstring.
    """

    @staticmethod
    def _feature_type_by_dtype(dtype) -> FeatureType:
        """Determines feature type by DataFrame dtype.

        Parameters
        ----------
        dtype: pd.DataFrame.dtypes
            The Pandas series data type.

        Returns
        -------
        FeatureType
            The subclass of FeatureType.
        """
        return _feature_type_by_dtype(dtype)

    @staticmethod
    def _is_type_registered(feature_type: Union[FeatureType, str]) -> bool:
        """Checks if provided feature type is registered in the system.

        Parameters
        ----------
        feature_type: Union[FeatureType, str]
            The FeatureType subclass or a str indicating feature type.

        Returns
        -------
        bool
            True if provided feature type registered, False otherwise.
        """
        return feature_type_manager.is_type_registered(feature_type)

    @staticmethod
    def _get_type(feature_type: Union[FeatureType, str]) -> FeatureType:
        """Gets a feature type by class object or name.

        Parameters
        ----------
        feature_type: Union[FeatureType, str]
            The FeatureType subclass or a str indicating feature type.

        Returns
        -------
        FeatureType
            Found feature type.
        """
        return feature_type_manager.feature_type_object(feature_type)

    def warning_registered(self) -> pd.DataFrame:
        """Lists registered warnings for all registered feature types.

        Returns
        -------
        :class:`pandas.DataFrame`
            The list of registered warnings for registered feature types.

        Examples
        --------
        >>> df.ads.warning_registered()
               Column    Feature Type             Warning                    Handler
           -------------------------------------------------------------------------
           0      Age      continuous               zeros              zeros_handler
           1      Age      continuous    high_cardinality   high_cardinality_handler

        >>> df["Age"].ads.warning_registered()
               Feature Type             Warning                    Handler
           ---------------------------------------------------------------
           0     continuous               zeros              zeros_handler
           1     continuous    high_cardinality   high_cardinality_handler
        """
        common_columns = ["Feature Type", "Warning", "Handler"]
        if isinstance(self._obj, pd.DataFrame):
            result_df = pd.DataFrame((), columns=["Column"] + common_columns)
            for col in self._obj.columns:
                feature_type_df = self._obj[col].ads.warning_registered()
                feature_type_df.insert(0, "Column", col)
                result_df = pd.concat([result_df, feature_type_df])
        else:
            result_df = pd.DataFrame((), columns=common_columns)
            for feature_type in self._feature_type:
                feature_type_df = feature_type.warning.registered()
                feature_type_df.insert(0, "Feature Type", feature_type.name)
                feature_type_df = feature_type_df.rename(columns={"Name": "Warning"})
                result_df = pd.concat([result_df, feature_type_df])
        result_df.reset_index(drop=True, inplace=True)
        return result_df

    def validator_registered(self) -> pd.DataFrame:
        """Lists registered validators for registered feature types.

        Returns
        -------
        :class:`pandas.DataFrame`
            The list of registered validators for registered feature types

        Examples
        --------
        >>> df.ads.validator_registered()
                 Column     Feature Type        Validator                 Condition                    Handler
        ------------------------------------------------------------------------------------------------------
        0   PhoneNumber    phone_number   is_phone_number                        ()            default_handler
        1   PhoneNumber    phone_number   is_phone_number    {'country_code': '+7'}   specific_country_handler
        2    CreditCard    credit_card     is_credit_card                        ()            default_handler

        >>> df['PhoneNumber'].ads.validator_registered()
            Feature Type            Validator                 Condition                     Handler
        -------------------------------------------------------------------------------------------
        0   phone_number      is_phone_number                        ()             default_handler
        1   phone_number      is_phone_number    {'country_code': '+7'}    specific_country_handler
        """
        common_columns = ["Feature Type", "Validator", "Condition", "Handler"]
        if isinstance(self._obj, pd.DataFrame):
            result_df = pd.DataFrame((), columns=["Column"] + common_columns)
            for col in self._obj.columns:
                feature_type_df = self._obj[col].ads.validator_registered()
                feature_type_df.insert(0, "Column", col)
                result_df = pd.concat([result_df, feature_type_df])
        else:
            result_df = pd.DataFrame((), columns=common_columns)
            for feature_type in self._feature_type:
                feature_type_df = feature_type.validator.registered()
                feature_type_df.insert(0, "Feature Type", feature_type.name)
                feature_type_df = feature_type_df.rename(columns={"Name": "Validator"})
                result_df = pd.concat([result_df, feature_type_df])
        result_df.reset_index(drop=True, inplace=True)
        return result_df

    def help(self, prop: str = None) -> None:
        """Help method that prints either a table of available properties or, given an individual property,
        returns its docstring.

        Parameters
        ----------
        prop : str
            The Name of property.

        Returns
        -------
        None
            Nothing.
        """
        if prop:
            if hasattr(self, prop):
                print(inspect.getdoc(getattr(self, prop)))
                return
            print(f"Property {prop} not found.")
            return

        methods = set()
        attrs = set()

        def get_attr(c):
            for attr in dir(c):
                if not attr.startswith("__") and not attr.startswith("_"):
                    attr_obj = getattr(c, attr)
                    if callable(attr_obj):
                        doc = inspect.getdoc(attr_obj)
                        if doc and len(doc.split(".")) > 0:
                            methods.add((attr, doc.split(".")[0]))
                        else:
                            methods.add((attr, "method"))
                    else:
                        if hasattr(c.__class__, attr):
                            attr_obj = getattr(c.__class__, attr)
                            doc = inspect.getdoc(attr_obj)
                            attrs.add((attr, doc.split(".")[0]))

        get_attr(self)
        props = sorted(list(methods) + list(attrs))
        print(tabulate.tabulate(props, headers=("Property", "Description")))
