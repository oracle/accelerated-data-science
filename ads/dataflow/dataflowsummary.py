#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, absolute_import

import abc
import ads.common.utils as utils
from oci.util import to_dict
from pandas import DataFrame
import pandas as pd
from abc import ABCMeta
from ads.common.decorator.deprecate import deprecated


class SummaryList(list, metaclass=ABCMeta):
    @deprecated("2.6.3")
    def __init__(self, entity_list, datetime_format=utils.date_format):
        if isinstance(entity_list, filter):
            entity_list = list(entity_list)
        super(SummaryList, self).__init__(entity_list)

        df = DataFrame.from_records([to_dict(entity) for entity in entity_list])

        # edge case handling: handle edge case where dataframe is empty
        if not df.empty:
            # handle collision protection
            nuniques = df["id"].nunique()
            minLen = -6
            shortened_ocids = df["id"].str[minLen:]
            while shortened_ocids.nunique() != nuniques:
                minLen -= 1
                shortened_ocids = df["id"].str[minLen:]
            df.index = shortened_ocids
            self.short_id_index = df["id"].to_dict()
            ordered_columns = [
                "display_name",
                "time_created",
                "lifecycle_state",
            ]

            ordered_columns.extend(
                [column for column in df.columns if column not in ordered_columns]
            )

            for column in ordered_columns:
                if column not in df.columns:
                    ordered_columns.remove(column)
            self.df = df[ordered_columns].drop(columns="id")
            self.df["time_created"] = pd.to_datetime(
                self.df["time_created"]
            ).dt.strftime(datetime_format)
            self.datetime_format = datetime_format
        else:
            self.df = df

    @abc.abstractmethod
    def sort_by(self, columns, reverse=False):  # pragma: no cover
        """
        Abstract sort method for dataflow summary.
        """
        pass

    @abc.abstractmethod
    def filter(self, selection, instance=None):  # pragma: no cover
        """
        Abstract filter method for dataflow summary.
        """
        pass

    @abc.abstractmethod
    def __add__(self, rhs):  # pragma: no cover
        pass

    def to_dataframe(self, datetime_format=None):
        """
        Abstract to_dataframe method for dataflow summary.
        """
        df = self.df.copy()
        if datetime_format is not None:
            df["time_created"] = pd.to_datetime(df["time_created"]).dt.strftime(
                datetime_format
            )
        return df
