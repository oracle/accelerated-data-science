#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os

from ads.aqua import set_log_level
from ads.aqua.deployment import AquaDeploymentApp
from ads.aqua.evaluation import AquaEvaluationApp
from ads.aqua.finetune import AquaFineTuningApp
from ads.aqua.model import AquaModelApp


class AquaCommand:
    """Contains the command groups for project Aqua."""

    model = AquaModelApp
    fine_tuning = AquaFineTuningApp
    deployment = AquaDeploymentApp
    evaluation = AquaEvaluationApp

    def __init__(self, log_level=os.environ.get("LOG_LEVEL", "INFO").upper()):
        set_log_level(log_level)
