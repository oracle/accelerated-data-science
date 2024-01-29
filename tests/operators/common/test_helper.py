#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import yaml
import random
import tempfile
import pandas as pd
from pandas.tseries.frequencies import to_offset
import numpy as np
import pytest
import datetime
from ads.opctl.operator.lowcode.common.errors import (
    InputDataError,
    InvalidParameterError,
    PermissionsError,
    DataMismatchError,
)
from ads.opctl.operator.cmd import run
from ads.opctl.operator.lowcode.common.utils import (
    get_frequency_of_datetime,
    get_frequency_in_seconds,
)


DATETIME_FORMATS_TO_TEST = [
    ["%Y"],
    ["%y"],
    ["%b-%d-%Y"],
    ["%d-%m-%y"],
    ["%d/%m/%y %H:%M:%S"],
]

timedelta_units = [
    "days",
    "seconds",
    "hours",
    "minutes",
    "microseconds",
    "milliseconds",
]  # 'weeks',


def test_frequency_heplers():
    step_size = random.randint(1, 59)
    dt_now = datetime.datetime.now()

    for unit in timedelta_units:
        delta = datetime.timedelta(**{unit: step_size})
        dt_col = pd.Series(np.arange(100) * delta + dt_now)

        # pd_unit = 'W' if unit == 'weeks' else unit
        assert (
            get_frequency_of_datetime(dt_col)
            == to_offset(pd.Timedelta(f"{step_size}{unit}")).freqstr
        )
        assert get_frequency_in_seconds(dt_col) == delta.total_seconds()
