#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that helps to manage feature types.
Provides functionalities to register, unregister, list feature types.

Classes
--------
    FeatureTypeManager
        Feature Types Manager class that manages feature types.

Examples
--------
    >>> from ads.feature_engineering.feature_type.base import FeatureType
    >>> class NewType(FeatureType):
    ...    description="My personal type."
    ...    pass
    >>> FeatureTypeManager.feature_type_register(NewType)
    >>> FeatureTypeManager.feature_type_registered()
                Name        Feature Type                                  Description
    ---------------------------------------------------------------------------------
    0	  Continuous          continuous          Type representing continuous values.
    1	    DateTime           date_time           Type representing date and/or time.
    2	    Category            category  Type representing discrete unordered values.
    3	     Ordinal             ordinal             Type representing ordered values.
    4        NewType            new_type                             My personal type.

    >>> FeatureTypeManager.warning_registered()
        Feature Type             Warning                    Handler
    ----------------------------------------------------------------------
    0     continuous               zeros              zeros_handler
    1     continuous    high_cardinality   high_cardinality_handler

    >>> FeatureTypeManager.validator_registered()
        Feature Type            Validator                 Condition                     Handler
    -------------------------------------------------------------------------------------------
    0   phone_number      is_phone_number                        ()             default_handler
    1   phone_number      is_phone_number    {'country_code': '+7'}    specific_country_handler
    2    credit_card       is_credit_card                        ()             default_handler

    >>> FeatureTypeManager.feature_type_unregister(NewType)
    >>> FeatureTypeManager.feature_type_reset()
    >>> FeatureTypeManager.feature_type_object('continuous')
    Continuous
"""
from typing import Union
import logging
import pandas as pd
import pandas.api.types as pdtypes
from ads.feature_engineering.feature_type.base import FeatureType
from ads.feature_engineering.feature_type.object import Object
from ads.feature_engineering.feature_type.integer import Integer
from ads.feature_engineering.feature_type.category import Category
from ads.feature_engineering.feature_type.ordinal import Ordinal
from ads.feature_engineering.feature_type.boolean import Boolean
from ads.feature_engineering.feature_type.string import String
from ads.feature_engineering.feature_type.lat_long import LatLong
from ads.feature_engineering.feature_type.creditcard import CreditCard
from ads.feature_engineering.feature_type.zip_code import ZipCode
from ads.feature_engineering.feature_type.phone_number import PhoneNumber
from ads.feature_engineering.feature_type.datetime import DateTime
from ads.feature_engineering.feature_type.continuous import Continuous
from ads.feature_engineering.feature_type.address import Address
from ads.feature_engineering.feature_type.constant import Constant
from ads.feature_engineering.feature_type.document import Document
from ads.feature_engineering.feature_type.gis import GIS
from ads.feature_engineering.feature_type.ip_address import IpAddress
from ads.feature_engineering.feature_type.ip_address_v4 import IpAddressV4
from ads.feature_engineering.feature_type.ip_address_v6 import IpAddressV6
from ads.feature_engineering.feature_type.text import Text
from ads.feature_engineering.feature_type.unknown import Unknown
from ads.feature_engineering.feature_type.discrete import Discrete
from ads.feature_engineering import exceptions
from ads.feature_engineering.feature_type.adsstring.string import ADSString

logger = logging.getLogger(__name__)


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
    if pdtypes.is_bool_dtype(dtype):
        return Boolean
    if pdtypes.is_datetime64_any_dtype(dtype):
        return DateTime
    if pdtypes.is_categorical_dtype(dtype):
        return Category
    if pdtypes.is_string_dtype(dtype):
        return String
    if pdtypes.is_float_dtype(dtype):
        return Continuous
    if pdtypes.is_integer_dtype(dtype):
        return Integer
    return Object


class FeatureTypeManager:
    """Feature Types Manager class that manages feature types.

    Provides functionalities to register, unregister, list feature types.

    Methods
    -------
    feature_type_object(cls, feature_type: Union[FeatureType, str]) -> FeatureType
        Gets a feature type by class object or name.
    feature_type_register(cls, feature_type_cls: FeatureType) -> None
        Registers a feature type.
    feature_type_unregister(cls, feature_type_cls: Union[FeatureType, str]) -> None
        Unregisters a feature type.
    feature_type_reset(cls) -> None
        Resets feature types to be default.
    feature_type_registered(cls) -> pd.DataFrame
        Lists all registered feature types as a DataFrame.
    warning_registered(cls) -> pd.DataFrame
        Lists registered warnings for all registered feature types.
    validator_registered(cls) -> pd.DataFrame
        Lists registered validators for all registered feature types.

    Examples
    --------
    >>> from ads.feature_engineering.feature_type.base import FeatureType
    >>> class NewType(FeatureType):
    ...    pass
    >>> FeatureTypeManager.register_feature_type(NewType)
    >>> FeatureTypeManager.feature_type_registered()
                Name      Feature Type                                  Description
    -------------------------------------------------------------------------------
    0	  Continuous        continuous          Type representing continuous values.
    1	    DateTime         date_time           Type representing date and/or time.
    2	    Category          category  Type representing discrete unordered values.
    3	     Ordinal           ordinal             Type representing ordered values.

    >>> FeatureTypeManager.warning_registered()
        Feature Type             Warning                    Handler
    ----------------------------------------------------------------------
    0     continuous               zeros              zeros_handler
    1     continuous    high_cardinality   high_cardinality_handler

    >>> FeatureTypeManager.validator_registered()
        Feature Type            Validator                 Condition                     Handler
    -------------------------------------------------------------------------------------------
    0   phone_number      is_phone_number                        ()             default_handler
    1   phone_number      is_phone_number    {'country_code': '+7'}    specific_country_handler
    2    credit_card       is_credit_card                        ()             default_handler

    >>> FeatureTypeManager.feature_type_unregister(NewType)
    >>> FeatureTypeManager.feature_type_reset()
    >>> FeatureTypeManager.feature_type_object('continuous')
    Continuous
    """

    _default_registered_type = [
        Continuous,
        DateTime,
        Category,
        Ordinal,
        Boolean,
        String,
        LatLong,
        PhoneNumber,
        ZipCode,
        CreditCard,
        Object,
        Integer,
        Address,
        Constant,
        Document,
        GIS,
        IpAddressV4,
        IpAddressV6,
        IpAddress,
        Text,
        Unknown,
        Discrete,
        ADSString,
    ]
    _name_to_type_map = {tp.name: tp for tp in _default_registered_type}

    @classmethod
    def feature_type_register(cls, feature_type_cls: FeatureType) -> None:
        """Registers new feature type.

        Parameters
        ----------
        feature_type : FeatureType
            Subclass of FeatureType to be registered.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        TypeError
            Type is not a subclass of FeatureType.
        TypeError
            Type has already been registered.
        NameError
            Name has already been used.
        """
        if not issubclass(feature_type_cls, FeatureType):
            raise exceptions.InvalidFeatureType(feature_type_cls.__name__)
        if feature_type_cls in cls._name_to_type_map.values():
            raise exceptions.TypeAlreadyRegistered(feature_type_cls.__name__)
        if feature_type_cls.name in cls._name_to_type_map:
            raise exceptions.NameAlreadyRegistered(feature_type_cls.name)

        cls._name_to_type_map[feature_type_cls.name] = feature_type_cls

    @classmethod
    def feature_type_unregister(cls, feature_type: Union[FeatureType, str]) -> None:
        """Unregisters a feature type.

        Parameters
        ----------
        feature_type: (FeatureType | str)
            The FeatureType subclass or a str indicating feature type.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        TypeError
            In attempt to unregister a default feature type.
        """
        feature_type_cls = cls.feature_type_object(feature_type)
        if feature_type_cls in cls._default_registered_type:
            raise TypeError(
                f"Default type {feature_type_cls.__name__} cannot be removed."
            )

        del cls._name_to_type_map[feature_type_cls.name]

    @classmethod
    def feature_type_reset(cls) -> None:
        """Resets feature types to be default.

        Returns
        -------
        None
            Nothing.
        """
        cls._name_to_type_map = {tp.name: tp for tp in cls._default_registered_type}

    @classmethod
    def feature_type_registered(cls) -> pd.DataFrame:
        """Lists all registered feature types as a DataFrame.

        Returns
        -------
        pd.DataFrame:
            The list of feature types in a DataFrame format.
        """
        return (
            pd.DataFrame(
                [
                    (ft.__name__, name, ft.description)
                    for name, ft in cls._name_to_type_map.items()
                ],
                columns=["Class", "Name", "Description"],
            )
            .sort_values(by=["Class", "Name"])
            .reset_index(drop=True)
        )

    @classmethod
    def feature_type_object(cls, feature_type: Union[FeatureType, str]) -> FeatureType:
        """Gets a feature type by class object or name.

        Parameters
        ----------
        feature_type: Union[FeatureType, str]
            The FeatureType subclass or a str indicating feature type.

        Returns
        -------
        FeatureType
            Found feature type.

        Raises
        ------
        TypeNotFound
            If provided feature type not registered.
        TypeError
            If provided feature type not a subclass of FeatureType.
        """

        if isinstance(feature_type, str):
            if feature_type not in cls._name_to_type_map:
                raise exceptions.TypeNotFound(feature_type)
            return cls._name_to_type_map[feature_type]

        if not isinstance(feature_type, FeatureType) and not issubclass(
            feature_type, FeatureType
        ):
            raise TypeError(
                f"{feature_type} must be an instance or subclass of FeatureType."
            )

        if isinstance(feature_type, FeatureType):
            feature_type_cls = feature_type.__class__
        else:
            feature_type_cls = feature_type

        if feature_type_cls not in cls._name_to_type_map.values():
            raise exceptions.TypeNotFound(feature_type_cls.__name__)

        return feature_type_cls

    @classmethod
    def is_type_registered(cls, feature_type: Union[FeatureType, str]) -> bool:
        """Checks if provided feature type registered in the system.

        Parameters
        ----------
        feature_type: Union[FeatureType, str]
            The FeatureType subclass or a str indicating feature type.

        Returns
        -------
        bool
            True if provided feature type registered, False otherwise.
        """
        result = False
        try:
            cls.feature_type_object(feature_type)
        except Exception as e:  # pylint: disable=broad-except
            logger.info("Error %s occured. Arguments %s", str(e), e.args)
        else:
            result = True
        return result

    @classmethod
    def warning_registered(cls) -> pd.DataFrame:
        """Lists registered warnings for all registered feature types.

        Returns
        -------
        pd.DataFrame:
            The list of registered warnings for registered feature types
            in a DataFrame format.

        Examples
        --------
        >>> FeatureTypeManager.warning_registered()
            Feature Type             Warning                    Handler
        ----------------------------------------------------------------------
        0     continuous               zeros              zeros_handler
        1     continuous    high_cardinality   high_cardinality_handler
        """
        result_df = pd.DataFrame((), columns=["Feature Type", "Warning", "Handler"])
        for feature_type in cls._name_to_type_map.values():
            feature_type_df = feature_type.warning.registered()
            feature_type_df.insert(0, "Feature Type", feature_type.name)
            feature_type_df = feature_type_df.rename(columns={"Name": "Warning"})
            result_df = pd.concat([result_df, feature_type_df])
        result_df.reset_index(drop=True, inplace=True)
        return result_df

    @classmethod
    def validator_registered(cls) -> pd.DataFrame:
        """Lists registered validators for registered feature types.

        Returns
        -------
        pd.DataFrame:
            The list of registered validators for registered feature types
            in a DataFrame format.

        Examples
        --------
        >>> FeatureTypeManager.validator_registered()
            Feature Type            Validator                 Condition                     Handler
        -------------------------------------------------------------------------------------------
        0   phone_number      is_phone_number                        ()             default_handler
        1   phone_number      is_phone_number    {'country_code': '+7'}    specific_country_handler
        2    credit_card       is_credit_card                        ()             default_handler
        """
        result_df = pd.DataFrame(
            [], columns=["Feature Type", "Validator", "Condition", "Handler"]
        )
        for feature_type in cls._name_to_type_map.values():
            feature_type_df = feature_type.validator.registered()
            feature_type_df.insert(0, "Feature Type", feature_type.name)
            feature_type_df = feature_type_df.rename(columns={"Name": "Validator"})
            result_df = pd.concat([result_df, feature_type_df])
        result_df.reset_index(drop=True, inplace=True)
        return result_df
