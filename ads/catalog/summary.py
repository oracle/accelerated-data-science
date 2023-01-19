#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, absolute_import
import warnings

warnings.warn(
    (
        "The `ads.catalog.summary` is deprecated in `oracle-ads 2.6.9` and will be removed in `oracle-ads 3.0`."
    ),
    DeprecationWarning,
    stacklevel=2,
)
import abc
import ads.common.utils as utils
from oci.util import to_dict
from pandas import DataFrame
import pandas as pd
from abc import ABCMeta
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


class SummaryList(list, metaclass=ABCMeta):
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
                "description",
                "time_created",
                "lifecycle_state",
                "user_name",
                "compartment_id",
                "project_id",
            ]
            remove_list = ["created_by"]

            ordered_columns.extend(
                [column for column in df.columns if column not in ordered_columns]
            )

            for column in ordered_columns:
                if column not in df.columns:
                    ordered_columns.remove(column)
            for column in remove_list:
                ordered_columns.remove(column)
            self.df = df[ordered_columns].drop(columns="id")
            self.df["compartment_id"] = "..." + self.df["compartment_id"].str[-6:]
            if "project_id" in ordered_columns:
                self.df["project_id"] = "..." + self.df["project_id"].str[-6:]
            if "model_version_set_id" in ordered_columns:
                self.df["model_version_set_id"] = (
                    "..." + self.df["model_version_set_id"].str[-6:]
                )
            self.df["time_created"] = pd.to_datetime(
                self.df["time_created"]
            ).dt.strftime(datetime_format)
            self.datetime_format = datetime_format
        else:
            self.df = df

    @abc.abstractmethod
    def sort_by(self, columns, reverse=False):  # pragma: no cover
        """
        Abstract method for sorting, implemented by the derived class
        """
        pass

    @abc.abstractmethod
    def filter(self, selection, instance=None):  # pragma: no cover
        """
        Abstract method for filtering, implemented by the derived class
        """
        pass

    @abc.abstractmethod
    def __add__(self, rhs):  # pragma: no cover
        pass

    def to_dataframe(self, datetime_format=None):

        """
        Returns the model catalog summary as a pandas dataframe

        Parameters
        ----------
        datatime_format: date_format
          A datetime format, like utils.date_format. Defaults to none.

        Returns
        -------
        Dataframe: The pandas DataFrame repersentation of the model catalog summary
        """

        df = self.df.copy()
        if datetime_format is not None:
            df["time_created"] = pd.to_datetime(df["time_created"]).dt.strftime(
                datetime_format
            )
        return df

    @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
    def show_in_notebook(self, datetime_format=None):

        """
        Displays the model catalog summary in a Jupyter Notebook cell

        Parameters
        ----------
        date_format: like utils.date_format. Defaults to none.

        Returns
        -------
        None
        """
        from IPython.core.display import display

        display(
            self.to_dataframe(datetime_format=datetime_format).style.applymap(
                self._color_lifecycle_state, subset=["lifecycle_state"]
            )
        )

    def _repr_html_(self):
        return self.df.style.applymap(
            self._color_lifecycle_state, subset=["lifecycle_state"]
        ).render()

    def _sort_by(self, cols, reverse=False):
        return sorted(
            self,
            key=lambda x: (
                [
                    getattr(x, col).lower()
                    if isinstance(getattr(x, col), str)
                    else getattr(x, col)
                    for col in cols
                ]
            ),
            reverse=reverse,
        )

    def _color_lifecycle_state(self, lifecycle_state):
        """
        Takes a scalar and returns a string with
        the css property
        """
        if lifecycle_state == "INACTIVE":
            color = "grey"
        elif lifecycle_state == "ACTIVE":
            color = "green"
        elif lifecycle_state == "DELETING":
            color = "black"
        elif lifecycle_state == "CREATING":
            color = "blue"
        elif lifecycle_state == "DELETED":
            color = "black"
        elif lifecycle_state == "FAILED":
            color = "red"
        return "color: %s" % color
