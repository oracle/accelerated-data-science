#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass, field
from ads.common.serializer import DataClassSerializable
from ads.aqua.base import AquaApp
from ads.aqua.job import AquaJobSummary, Resource, AquaResourceIdentifier, JobTypeTags
from ads.common.utils import get_console_link
from ads.aqua.utils import UNKNOWN


@dataclass(repr=False)
class AquaFineTuningParams(DataClassSerializable):
    epochs: int = None
    learning_rate: float = None


@dataclass(repr=False)
class AquaFineTuningSummary(AquaJobSummary, DataClassSerializable):
    parameters: AquaFineTuningParams = field(default_factory=AquaFineTuningParams)


class AquaFineTuningApp(AquaApp):
    """Contains APIs for Aqua fine-tuning jobs."""

    def create(self, **kwargs) -> "AquaFineTuningSummary":
        """Creates a aqua fine tuned model."""

        # todo : parse kwargs and convert to CreateAquaFineTuneDetails object
        #   with CreateAquaFineTuneDetails(**kwargs)

        return AquaFineTuningSummary(
            id="ocid1.datasciencemodel.oc1.iad.xxxx",
            name="Fine Tuned Model Name",
            console_url=get_console_link(
                resource=Resource.MODEL.value,
                ocid="ocid1.datasciencemodel.oc1.iad.xxxx",
                region=self.region,
            ),
            time_created="2024-02-15 20:18:34.225000+00:00",
            lifecycle_state=UNKNOWN,
            lifecycle_details=UNKNOWN,
            experiment=AquaResourceIdentifier(
                id="ocid1.datasciencemodelversionset.oc1.iad.xxxx",
                name="Model Version Set Name",
                url=get_console_link(
                    resource=Resource.MODEL_VERSION_SET.value,
                    ocid="ocid1.datasciencemodelversionset.oc1.iad.xxxx",
                    region=self.region,
                ),
            ),
            source=AquaResourceIdentifier(
                id="ocid1.datasciencemodel.oc1.iad.xxxx",
                name="Base Model Name",
                url=get_console_link(
                    resource=Resource.MODEL.value,
                    ocid="ocid1.datasciencemodel.oc1.iad.xxxx",
                    region=self.region,
                ),
            ),
            job=AquaResourceIdentifier(
                id="ocid1.datasciencejob.oc1.iad.xxxx",
                name="Fine Tuning Job Name",
                url=get_console_link(
                    resource=Resource.JOB.value,
                    ocid="ocid1.datasciencejob.oc1.iad.xxxx",
                    region=self.region,
                ),
            ),
            tags=dict(
                aqua_finetuning=JobTypeTags.AQUA_FINE_TUNING.value,
                finetuning_job_id="ocid1.datasciencejob.oc1.iad.xxxx",
                finetuning_source="ocid1.datasciencemodel.oc1.iad.xxxx",
                finetuning_experiment_id="ocid1.datasciencemodelversionset.oc1.iad.xxxx",
            ),
            parameters=AquaFineTuningParams(epochs=0, learning_rate=0.0),
        )
