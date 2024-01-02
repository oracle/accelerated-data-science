#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.aqua.model import AquaModel
from ads.aqua.fine_tune import AquaFineTuning
from ads.aqua.deployment import AquaDeployment


class AquaCommand:
    """Contains the command groups for project Aqua."""

    model = AquaModel
    fine_tuning = AquaFineTuning
    deployment = AquaDeployment
