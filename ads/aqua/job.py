#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import logging
from dataclasses import dataclass
from typing import List
from ads.aqua.base import AquaApp


logger = logging.getLogger(__name__)


@dataclass
class AquaJobSummary:
    """Represents an Aqua job summary."""

    id: str
    compartment_id: str
    project_id: str

    model_id: str
    task: str


@dataclass
class AquaJob(AquaJobSummary):
    """Represents an Aqua job."""

    dataset: str


@dataclass
class AquaFineTuningJob(AquaJob):
    """Represents an Aqua fine-tuning job."""

    epoch: int


class AquaJobApp(AquaApp):
    """Contains APIs for Aqua jobs."""


class AquaFineTuningApp(AquaApp):
    """Contains APIs for Aqua fine-tuning jobs."""

    def get(self, job_id) -> AquaFineTuningJob:
        """Gets the information of an Aqua model."""
        return AquaFineTuningJob(
            id=job_id,
            compartment_id="ocid.compartment.xxx",
            project_id="ocid.project.xxx",
            model_id="ocid.model.xxx",
            task="fine-tuning",
            dataset="dummy",
            epoch=2,
        )

    def list(self, compartment_id, project_id=None, **kwargs) -> List[AquaJobSummary]:
        """Lists Aqua models."""
        return [
            AquaJobSummary(
                id=f"ocid{i}.xxx",
                compartment_id=compartment_id,
                project_id=project_id,
                model_id="ocid.model.xxx",
                task="fine-tuning",
            )
            for i in range(5)
        ]
