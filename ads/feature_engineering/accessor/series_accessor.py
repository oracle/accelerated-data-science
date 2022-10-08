#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The ADS accessor for the Pandas Series.
The accessor will be initialized with the pandas object the user is interacting with.

Examples
--------
    >>> from ads.feature_engineering.accessor.series_accessor import ADSSeriesAccessor
    >>> from ads.feature_engineering.feature_type.string import String
    >>> from ads.feature_engineering.feature_type.ordinal import Ordinal
    >>> from ads.feature_engineering.feature_type.base import Tag
    >>> series = pd.Series(['name1', 'name2', 'name3'])
    >>> series.ads.default_type
    'string'
    >>> series.ads.feature_type
    ['string']
    >>> series.ads.feature_type_description
        Feature Type                         Description
    ----------------------------------------------------
    0         string    Type representing string values.
    >>> series.ads.feature_type = ['string', Ordinal, Tag('abc')]
    >>> series.ads.feature_type
    ['string', 'ordinal', 'abc']
    >>> series1 = series.dropna()
    >>> series1.ads.sync(series)
    >>> series1.ads.feature_type
    ['string', 'ordinal', 'abc']
"""

import inspect
import logging
from typing import List, Union

import pandas as pd
from ads.feature_engineering.accessor.mixin.eda_mixin_series import EDAMixinSeries
from ads.feature_engineering.accessor.mixin.feature_types_mixin import (
    ADSFeatureTypesMixin,
)
from ads.feature_engineering.exceptions import TypeNotFound
from ads.feature_engineering.feature_type.base import FeatureType, Tag
from ads.feature_engineering.feature_type.handler.feature_validator import (
    FeatureValidator,
)
from ads.feature_engineering.feature_type.adsstring.string import ADSString

logger = logging.getLogger(__name__)


class ADSSeriesValidator:
    """Class helper to invoke registerred validator on a series level."""

    def __init__(self, feature_type_list: List[FeatureType], series: pd.Series) -> None:
        """Initializes ADS series validator.

        Parameters
        ----------
        feature_type_list : List[FeatureType]
            The list of feature types.
        series : `pd.Series`
            The pandas series.
        """
        self._feature_type_list = feature_type_list
        self._series = series

    def __getattr__(self, attr):
        """Makes it possible to invoke registered validators as a regular method."""
        for feature_type in self._feature_type_list:
            if hasattr(feature_type.validator, attr):
                feature_type.validator._bind_data(self._series)
                return getattr(feature_type.validator, attr)
        raise AttributeError(attr)


@pd.api.extensions.register_series_accessor("ads")
class ADSSeriesAccessor(ADSFeatureTypesMixin, EDAMixinSeries):
    """ADS accessor for Pandas Series.

    Attributes
    ----------
    name: str
        The name of Series.
    tags: List[str]
        The list of tags for the Series.

    Methods
    -------
    help(self, prop: str = None) -> None
        Provids docstring for affordable methods and properties.
    sync(self, src: Union[pd.DataFrame, pd.Series]) -> None
        Syncs feature types of current series with that from src.

    Attributes
    ----------
    default_type(self) -> str
        Gets the name of default feature type for the series.
    feature_type(self) -> List[str]
        Gets the list of registered feature types for the series.
    feature_type_description(self) -> pd.DataFrame
        Gets the list of registered feature types in a DataFrame format.

    Examples
    --------
    >>> from ads.feature_engineering.accessor.series_accessor import ADSSeriesAccessor
    >>> from ads.feature_engineering.feature_type.string import String
    >>> from ads.feature_engineering.feature_type.ordinal import Ordinal
    >>> from ads.feature_engineering.feature_type.base import Tag
    >>> series = pd.Series(['name1', 'name2', 'name3'])
    >>> series.ads.default_type
    'string'
    >>> series.ads.feature_type
    ['string']
    >>> series.ads.feature_type_description
        Feature Type                         Description
    ----------------------------------------------------
    0         string    Type representing string values.
    >>> series.ads.feature_type = ['string', Ordinal, Tag('abc')]
    >>> series.ads.feature_type
    ['string', 'ordinal', 'abc']
    >>> series1 = series.dropna()
    >>> series1.ads.sync(series)
    >>> series1.ads.feature_type
    ['string', 'ordinal', 'abc']
    """

    def __init__(self, pandas_obj: pd.Series) -> None:
        """Initializes ADS Pandas Series Accessor.

        Parameters
        ----------
        pandas_obj : `pd.Series`
            The pandas series
        """
        self._obj = pandas_obj
        super().__init__()
        self._feature_type = [self._default_type]
        self.tags = []
        self.name = self._obj.name

    @property
    def _default_type(self) -> FeatureType:
        """Gets default feature type for the series.

        Returns
        -------
        FeatureType
            The default feature type for the series.
        """
        return self._feature_type_by_dtype(self._obj.dtype)

    @property
    def default_type(self) -> str:
        """Gets the name of default feature type for the series.

        Returns
        -------
        str
            The name of default feature type.
        """
        return self._default_type.name

    @property
    def feature_type(self) -> List[str]:
        """Gets the list of registered feature types for the series.

        Returns
        -------
        List[str]
            Names of feature types.

        Examples
        --------
        >>> series = pd.Series(['name1'])
        >>> series.ads.feature_type = ['name', 'string', Tag('tag for name')]
        >>> series.ads.feature_type
        ['name', 'string', 'tag for name']
        """
        types = []
        for feature_type in self._feature_type:
            types.append(feature_type.name)
        return types + self.tags

    @property
    def feature_type_description(self) -> pd.DataFrame:
        """Gets the list of registered feature types in a DataFrame format.

        Returns
        -------
        pd.DataFrame
            The DataFrame with feature types for this series.

        Examples
        --------
        >>> series = pd.Series(['name1'])
        >>> series.ads.feature_type = ['name', 'string', Tag('Name tag')]
        >>> series.ads.feature_type_description
                Feature Type                               Description
            ----------------------------------------------------------
            0           name            Type representing name values.
            1         string          Type representing string values.
            2        Name tag                                     Tag.
        """
        feature_types = (
            (feature_type.name, feature_type.description)
            for feature_type in self._feature_type
            if self._is_type_registered(feature_type)
        )
        tags = ((tag, "Tag") for tag in self.tags)
        return pd.DataFrame(
            tuple(feature_types) + tuple(tags), columns=["Feature Type", "Description"]
        )

    @feature_type.setter
    def feature_type(self, feature_types: List[Union[FeatureType, str, Tag]]) -> None:
        """Sets feature types for the series.

        Parameters
        ----------
        feature_types : List[Union[FeatureType, str, Tag]]
            The list of feature types.

        Return
        ------
        None
            Nothing.

        Raises
        ------
        TypeError: If input data has wrong format.

        Examples
        --------
        >>> series = pd.Series(['name1', 'name2', 'name3'])
        >>> series.ads.feature_type = ['name']
        >>> series.feature_type
        ['name', 'string']
        >>> series.ads.feature_type = ['string', 'name']
        >>> series.feature_type
        ['string', 'name']
        >>> series.ads.feature_type = []
        >>> series.feature_type
        ['string']
        """
        if feature_types is None or not isinstance(feature_types, list):
            raise TypeError("Argument must be a list of feature types.")

        self._feature_type = []
        self.tags = []
        for feature_type in feature_types:
            self._add_feature_type(feature_type)

        default_feature_type = self._default_type
        if default_feature_type not in self._feature_type:
            self._add_feature_type(default_feature_type)

    def sync(self, src: Union[pd.DataFrame, pd.Series]) -> None:
        """Syncs feature types of current series with that from src.

        The src could be a dataframe or a series. In either case, only columns
        with matched names are synced.

        Parameters
        ----------
        src: (`pd.DataFrame` | `pd.Series`)
            The source to sync from.

        Returns
        -------
        None
            Nothing.

        Examples
        --------
        >>> series = pd.Series(['name1', 'name2', 'name3', None])
        >>> series.ads.feature_type = ['name']
        >>> series.ads.feature_type
        ['name', string]
        >>> series.dropna().ads.feature_type
        ['string']
        >>> series1 = series.dropna()
        >>> series1.ads.sync(series)
        >>> series1.ads.feature_type
        ['name', 'string']
        """
        if isinstance(src, pd.DataFrame):
            if self._obj.name not in src.columns:
                logger.warning(
                    "The source DataFrame doesn't have a clumn %s.", self._obj.name
                )
                return
            self._sync(src[self._obj.name])
        elif isinstance(src, pd.Series):
            self._sync(src)

    def _sync(self, src: pd.Series) -> None:
        """Copies all feature types from src series to the current."""
        new_feature_type = [ft for ft in src.ads._feature_type]
        new_tags = [Tag(tag) for tag in src.ads.tags]
        self.feature_type = new_feature_type + new_tags

    def _add_feature_type(self, feature_type: Union[FeatureType, str, Tag]) -> None:
        """Adds a feature type to the series.

        Parameters
        ----------
        feature_type : Union[FeatureType, str, Tag]
            The feature type to add.

        Returns
        -------
        None
            Nothing.
        """
        if isinstance(feature_type, Tag):
            if feature_type.name in self.feature_type:
                logger.warning(
                    "The tag '%s' is already added to the series '%s'.",
                    feature_type.name,
                    self.name,
                )
            self.tags.append(feature_type.name)
        else:
            feature_type_cls = self._get_type(feature_type)
            if feature_type_cls.name in self.feature_type:
                logger.warning(
                    "The type '%s' is already added to the series '%s'.",
                    feature_type_cls.name,
                    self.name,
                )
            self._feature_type.append(feature_type_cls)

    def _remove_feature_type(self, feature_type: Union[FeatureType, str, Tag]) -> None:
        """Removes a feature type.

        Parameters
        ----------
        feature_type : Union[FeatureType, str, Tag]
            feature type

        Raises
        ------
        TypeNotFound
            Type does not exist for this series
        """
        tag = self._get_tag(feature_type)
        if tag:
            if tag in self.tags:
                self.tags.remove(tag)
                return
            else:
                raise TypeNotFound(tag)

        feature_type_cls = self._get_type(feature_type)
        if feature_type_cls in self._feature_type:
            self._feature_type.remove(feature_type_cls)
            return

        raise TypeNotFound(feature_type_cls.__name__)

    def _get_tag(self, feature_type: Union[str, Tag]) -> str:
        if isinstance(feature_type, str) and feature_type in self.tags:
            return feature_type
        if isinstance(feature_type, Tag):
            return feature_type.name

    def __getattr__(self, attr):
        if attr == "validator":
            return ADSSeriesValidator(self._feature_type, self._obj)

        for feature_type in self._feature_type:
            if feature_type == ADSString:
                # We need to initialize first to use the plugins.
                if hasattr(feature_type("default"), attr):
                    methods = inspect.classify_class_attrs(feature_type)
                    for method in methods:
                        if method.name == attr:
                            if method.kind == "method":
                                return lambda *args, **kwargs: [
                                    getattr(ADSString(v), attr)(*args, **kwargs)
                                    for v in self._obj.values
                                ]

                            elif method.kind == "property":
                                attr_objects = []
                                for v in self._obj.values:
                                    attr_objects.append(getattr(feature_type(v), attr))
                                return attr_objects

            if hasattr(feature_type, attr):
                # non-instance methods, aka class method or static method
                non_ins_methods = [
                    method.name
                    for method in inspect.classify_class_attrs(feature_type)
                    if method.kind == "class method" or method.kind == "static method"
                ]
                # instance method
                ins_methods = [
                    method.name
                    for method in inspect.classify_class_attrs(feature_type)
                    if method.kind == "method"
                ]
                attr_object = getattr(feature_type, attr)

                # if isinstance(attr_object, FeatureValidator):
                #     attr_object._bind_data(self._obj)

                # there was one case that inspect could not track one of the function.
                # hence replace with __dict__. however, problem with __dict__ is that
                # it does not track back its ancestors functions. hence, use a union
                # to safeguard different scenarios.
                if (attr in non_ins_methods) or (
                    attr in feature_type.__dict__
                    and isinstance(feature_type.__dict__[attr], staticmethod)
                    or isinstance(feature_type.__dict__[attr], classmethod)
                ):
                    return lambda *args, **kwargs: attr_object(
                        self._obj, *args, **kwargs
                    )
                elif attr in ins_methods:
                    return lambda *args, **kwargs: getattr(feature_type(), attr)(
                        self._obj, *args, **kwargs
                    )
                return attr_object

        raise AttributeError(
            f"'{self.__class__.__name__}' does not have an attribute {attr}."
        )
