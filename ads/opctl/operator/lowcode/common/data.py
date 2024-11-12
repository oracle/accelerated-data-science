#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import time
from .transformations import Transformations
from ads.opctl import logger
from ads.opctl.operator.lowcode.common.const import DataColumns
from ads.opctl.operator.lowcode.common.utils import load_data
from ads.opctl.operator.lowcode.common.errors import (
    InputDataError,
    InvalidParameterError,
    PermissionsError,
    DataMismatchError,
)
from abc import ABC
import pandas as pd


class AbstractData(ABC):
    def __init__(self, spec: dict, name="input_data"):
        self.Transformations = Transformations
        self.data = None
        self._data_dict = dict()
        self.name = name
        self.spec = spec
        self.load_transform_ingest_data(spec)

    def get_raw_data_by_cat(self, category):
        mapping = self._data_transformer.get_target_category_columns_map()
        # For given category, mapping gives the target_category_columns and it's values.
        # condition filters raw_data based on the values of target_category_columns for the given category
        condition = pd.Series(True, index=self.raw_data.index)
        if category in mapping:
            for col, val in mapping[category].items():
                condition &= (self.raw_data[col] == val)
        data_by_cat = self.raw_data[condition].reset_index(drop=True)
        data_by_cat = self._data_transformer._format_datetime_col(data_by_cat) if self.spec.datetime_column else data_by_cat
        return data_by_cat


    def get_dict_by_series(self):
        if not self._data_dict:
            for s_id in self.list_series_ids():
                try:
                    self._data_dict[s_id] = self.data.xs(
                        s_id, level=DataColumns.Series
                    ).reset_index()
                except KeyError as ke:
                    logger.debug(
                        f"Unable to extract series: {s_id} from data: {self.data}. This may occur due to significant missing data. Error message: {ke.args}"
                    )
                    pass
        return self._data_dict

    def get_data_for_series(self, series_id):
        data_dict = self.get_dict_by_series()
        try:
            return data_dict[series_id]
        except:
            raise InvalidParameterError(
                f"Unable to retrieve series {series_id} from {self.name}. Available series ids are: {self.list_series_ids()}"
            )

    def _load_data(self, data_spec, **kwargs):
        loading_start_time = time.time()
        try:
            raw_data = load_data(data_spec)
        except InvalidParameterError as e:
            e.args = e.args + (f"Invalid Parameter: {self.name}",)
            raise e
        loading_end_time = time.time()
        logger.info(
            f"{self.name} loaded in {loading_end_time - loading_start_time} seconds",
        )
        return raw_data

    def _transform_data(self, spec, raw_data, **kwargs):
        transformation_start_time = time.time()
        self._data_transformer = self.Transformations(spec, name=self.name)
        data = self._data_transformer.run(raw_data)
        transformation_end_time = time.time()
        logger.info(
            f"{self.name} transformations completed in {transformation_end_time - transformation_start_time} seconds"
        )
        return data

    def load_transform_ingest_data(self, spec):
        self.raw_data = self._load_data(getattr(spec, self.name))
        self.data = self._transform_data(spec, self.raw_data)
        self._ingest_data(spec)

    def _ingest_data(self, spec):
        pass

    def get_data_long(self):
        return self.data.reset_index(drop=False)

    def get_min_time(self):
        return self.data.index.get_level_values(0).min()

    def get_max_time(self):
        return self.data.index.get_level_values(0).max()

    def list_series_ids(self):
        return self.data.index.get_level_values(1).unique().tolist()

    def get_num_rows(self):
        return self.data.shape[0]
