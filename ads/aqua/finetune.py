#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass, field
from typing import Dict

from ads.common.serializer import DataClassSerializable
from ads.common.utils import get_console_link
from ads.config import AQUA_MODEL_FINETUNING_CONFIG

from ads.aqua.base import AquaApp
from ads.aqua.job import AquaJobSummary
from ads.aqua.data import Resource, AquaResourceIdentifier, Tags
from ads.aqua.utils import UNKNOWN


@dataclass(repr=False)
class AquaFineTuningParams(DataClassSerializable):
    epochs: int = None
    learning_rate: float = None


@dataclass(repr=False)
class AquaFineTuningSummary(AquaJobSummary, DataClassSerializable):
    parameters: AquaFineTuningParams = field(default_factory=AquaFineTuningParams)


class AquaFineTuningApp(AquaApp):
    """Provides a suite of APIs to interact with Aqua fine-tuned models within the Oracle
    Cloud Infrastructure Data Science service, serving as an interface for creating fine-tuned models.

    Methods
    -------
    create(...) -> AquaFineTuningSummary
        Creates a fine-tuned Aqua model.
    get_finetuning_config(self, model_id: str) -> Dict:
        Gets the finetuning config for given Aqua model.

    Note:
        Use `ads aqua finetuning <method_name> --help` to get more details on the parameters available.
        This class is designed to work within the Oracle Cloud Infrastructure
        and requires proper configuration and authentication set up to interact
        with OCI services.
    """

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
                aqua_finetuning=Tags.AQUA_FINE_TUNING.value,
                finetuning_job_id="ocid1.datasciencejob.oc1.iad.xxxx",
                finetuning_source="ocid1.datasciencemodel.oc1.iad.xxxx",
                finetuning_experiment_id="ocid1.datasciencemodelversionset.oc1.iad.xxxx",
            ),
            parameters=AquaFineTuningParams(epochs=0, learning_rate=0.0),
        )

    def get_finetuning_config(self, model_id: str) -> Dict:
        """Gets the finetuning config for given Aqua model.

        Parameters
        ----------
        model_id: str
            The OCID of the Aqua model.

        Returns
        -------
        Dict:
            A dict of allowed finetuning configs.
        """

        return self.get_config(model_id, AQUA_MODEL_FINETUNING_CONFIG)
