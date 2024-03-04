#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import logging
from dataclasses import dataclass, field
from ads.common.serializer import DataClassSerializable
from ads.aqua.data import AquaResourceIdentifier

logger = logging.getLogger(__name__)


@dataclass(repr=False)
class AquaJobSummary(DataClassSerializable):
    """Represents an Aqua job summary."""

    id: str
    name: str
    console_url: str
    lifecycle_state: str
    lifecycle_details: str
    time_created: str
    tags: dict
    experiment: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    source: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    job: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
