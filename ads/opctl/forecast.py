#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.operator.lowcode.forecast.__main__ import operate
from ads.opctl.operator.lowcode.forecast.operator_config import ForecastOperatorConfig

if __name__ == "__main__":
    config = ForecastOperatorConfig()
    operate(config)
