#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.operator.lowcode.anomaly.__main__ import operate
from ads.opctl.operator.lowcode.anomaly.operator_config import AnomalyOperatorConfig

if __name__ == "__main__":
    config = AnomalyOperatorConfig()
    operate(config)
