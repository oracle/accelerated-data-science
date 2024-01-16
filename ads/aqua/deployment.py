#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from typing import List, Dict
from ads.model.deployment import ModelDeployment

logger = logging.getLogger(__name__)


class AquaDeployment(ModelDeployment):
    """Represents an Aqua Model Deployment.

    Attributes
    ----------

    Methods
    -------
    create(self, **kwargs) -> "AquaDeployment"
        Creates an instance of model deployment via Aqua
    list(self, ..., **kwargs) -> List["AquaDeployment"]
        List existing model deployments created via Aqua
    clone()
        Clone an existing model deployment
    suggest()
        Provide suggestions for model deployment via Aqua
    stats()
        Get model deployment statistics
    """

    def __init__(self, **kwargs):
        """Initializes an Aqua model deployment."""
        super().__init__(**kwargs)

    def create(self, **kwargs) -> "AquaDeployment":
        pass

    def list(self, **kwargs) -> List["AquaDeployment"]:
        pass

    def clone(self, **kwargs) -> "AquaDeployment":
        pass

    def suggest(self, **kwargs) -> Dict:
        pass

    def stats(self, **kwargs) -> Dict:
        pass
