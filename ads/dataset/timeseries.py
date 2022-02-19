#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class Timeseries:
    def __init__(self, col_name, df, date_range=None, min=None, max=None):
        self.col_name = col_name
        self.df = df
        self.date_range = date_range
        self.min = min
        self.max = max

    def plot(self, **kwargs):
        # this could be improved :)
        pass
