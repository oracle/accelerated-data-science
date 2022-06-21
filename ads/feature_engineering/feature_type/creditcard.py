#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that represents a CreditCard feature type.

Classes:
    CreditCard
        The CreditCard feature type.

Functions:
    default_handler(data: pd.Series) -> pd.Series
        Processes given data and indicates if the data matches requirements.
    _luhn_checksum(card_number: str) -> float
        Implements Luhn algorithm to validate a credit card number.
"""
import matplotlib.pyplot as plt
import pandas as pd
import re
from ads.feature_engineering.feature_type.string import String
from ads.feature_engineering.utils import (
    assign_issuer,
    _count_unique_missing,
    _set_seaborn_theme,
    SchemeTeal,
)
from ads.feature_engineering import schema
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


_max_sample_size_to_luhn_check = 1000
_pattern_string = r"""
        ^(?:4[0-9]{12}(?:[0-9]{3})?         # Visa
        |  (?:5[1-5][0-9]{2}                # MasterCard
        | 222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12}
        |  3[47][0-9]{13}                   # American Express
        |  3(?:0[0-5]|[68][0-9])[0-9]{11}   # Diners Club
        |  6(?:011|5[0-9]{2})[0-9]{12}      # Discover
        |  (?:2131|1800|35\d{3})\d{11}      # JCB
        |  (5018|5020|5038|5612|5893|6304|6759|6761|6762|6763|0604|6390)\d+$   # Maestro
        |  ^(5[06789]|6)[0-9]{0,}$          # Maestro
        |  ^4[0-9]{12}(?:[0-9]{6})?$        #Visa 19 digit
        )$
    """
PATTERN = re.compile(_pattern_string, re.VERBOSE)


def default_handler(data: pd.Series, *args, **kwargs) -> pd.Series:
    """
    Processes given data and indicates if the data matches requirements.

    Parameters
    ----------
    data: :class:`pandas.Series`
        The data to process.

    Returns
    -------
    :class:`pandas.Series`
        The logical list indicating if the data matches requirements.
    """

    def _is_credit_card(x: pd.Series):
        return (
            not pd.isnull(x)
            and PATTERN.match(str(x)) is not None
            and _luhn_checksum(str(x)) == 0
        )

    return data.apply(lambda x: True if _is_credit_card(x) else False)


def _luhn_checksum(card_number: str) -> float:
    """
    Implements Luhn algorithm to validate a credit card number.

    Parameters
    ----------
    card_number : str
        The credit card number.

    Returns
    -------
    float
        The checksum of the card number
    """

    def digits_of(n):
        return [int(d) for d in str(n)]

    digits = digits_of(card_number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = 0
    checksum += sum(odd_digits)
    for d in even_digits:
        checksum += sum(digits_of(d * 2))
    return checksum % 10


class CreditCard(String):
    """
    Type representing credit card numbers.

    Attributes
    ----------
    description: str
        The feature type description.
    name: str
        The feature type name.
    warning: FeatureWarning
        Provides functionality to register warnings and invoke them.
    validator
        Provides functionality to register validators and invoke them.

    Methods
    --------
    feature_stat(x: pd.Series) -> pd.DataFrame
        Generates feature statistics.
    feature_plot(x: pd.Series) -> plt.Axes
        Shows the counts of observations in each credit card type using bar chart.


    Examples
    --------
    >>> from ads.feature_engineering.feature_type.creditcard import CreditCard
    >>> import pandas as pd
    >>> s = pd.Series(["4532640527811543", None, "4556929308150929", "4539944650919740", "4485348152450846", "4556593717607190"], name='credit_card')
    >>> s.ads.feature_type = ['credit_card']
    >>> CreditCard.validator.is_credit_card(s)
    0     True
    1    False
    2     True
    3     True
    4     True
    5     True
    Name: credit_card, dtype: bool
    """

    description = "Type representing credit card numbers."

    @staticmethod
    def feature_stat(x: pd.Series):
        """Generates feature statistics.

        Feature statistics include (total)count, unique(count), missing(count) and
            count of each credit card type.

        Examples
        --------
        >>> visa = [
            "4532640527811543",
            None,
            "4556929308150929",
            "4539944650919740",
            "4485348152450846",
            "4556593717607190",
            ]
        >>> mastercard = [
            "5334180299390324",
            "5111466404826446",
            "5273114895302717",
            "5430972152222336",
            "5536426859893306",
            ]
        >>> amex = [
            "371025944923273",
            "374745112042294",
            "340984902710890",
            "375767928645325",
            "370720852891659",
            ]
        >>> creditcard_list = visa + mastercard + amex
        >>> creditcard_series = pd.Series(creditcard_list,name='card')
        >>> creditcard_series.ads.feature_type = ['credit_card']
        >>> creditcard_series.ads.feature_stat()
            Metric              Value
        0	count	            16
        1	unique	            15
        2	missing	            1
        3	count_Amex	        5
        4	count_Visa	        5
        5	count_MasterCard	3
        6	count_Diners Club	2
        7	count_missing	    1

        Returns
        -------
        :class:`pandas.DataFrame`
            Summary statistics of the Series or Dataframe provided.
        """
        df_stat = _count_unique_missing(x)
        card_types = x.apply(assign_issuer)
        value_counts = card_types.value_counts()
        value_counts.index = [
            "count_" + cardtype for cardtype in list(value_counts.index)
        ]
        return pd.concat([df_stat, value_counts.to_frame()])

    @staticmethod
    @runtime_dependency(module="seaborn", install_from=OptionalDependency.VIZ)
    def feature_plot(x: pd.Series) -> plt.Axes:
        """
        Shows the counts of observations in each credit card type using bar chart.

        Examples
        --------
        >>> visa = [
            "4532640527811543",
            None,
            "4556929308150929",
            "4539944650919740",
            "4485348152450846",
            "4556593717607190",
            ]
        >>> mastercard = [
            "5334180299390324",
            "5111466404826446",
            "5273114895302717",
            "5430972152222336",
            "5536426859893306",
            ]
        >>> amex = [
            "371025944923273",
            "374745112042294",
            "340984902710890",
            "375767928645325",
            "370720852891659",
            ]
        >>> creditcard_list = visa + mastercard + amex
        >>> creditcard_series = pd.Series(creditcard_list,name='card')
        >>> creditcard_series.ads.feature_type = ['credit_card']
        >>> creditcard_series.ads.feature_plot()

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Plot object for the series based on the CreditCard feature type.
        """
        card_types = x.apply(assign_issuer)
        df = card_types.value_counts().to_frame()
        if len(df.index):
            _set_seaborn_theme()
            ax = seaborn.barplot(
                y=df.index, x=list(df.iloc[:, 0]), color=SchemeTeal.AREA_DARK
            )
            ax.set(xlabel="Count")
            return ax

    @classmethod
    def feature_domain(cls, x: pd.Series) -> schema.Domain:
        """
        Generate the domain of the data of this feature type.

        Examples
        --------
        >>> visa = [
            "4532640527811543",
            None,
            "4556929308150929",
            "4539944650919740",
            "4485348152450846",
            "4556593717607190",
            ]
        >>> mastercard = [
            "5334180299390324",
            "5111466404826446",
            "5273114895302717",
            "5430972152222336",
            "5536426859893306",
            ]
        >>> amex = [
            "371025944923273",
            "374745112042294",
            "340984902710890",
            "375767928645325",
            "370720852891659",
            ]
        >>> creditcard_list = visa + mastercard + amex
        >>> creditcard_series = pd.Series(creditcard_list,name='card')
        >>> creditcard_series.ads.feature_type = ['credit_card']
        >>> creditcard_series.ads.feature_domain()
        constraints: []
        stats:
            count: 16
            count_Amex: 5
            count_Diners Club: 2
            count_MasterCard: 3
            count_Visa: 5
            count_missing: 1
            missing: 1
            unique: 15
        values: CreditCard

        Returns
        -------
        ads.feature_engineering.schema.Domain
            Domain based on the CreditCard feature type.
        """
        return schema.Domain(
            cls.__name__,
            cls.feature_stat(x).to_dict()[x.name],
            [],
        )


CreditCard.validator.register("is_credit_card", default_handler)
