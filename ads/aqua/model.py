#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import datetime
from dateutil.tz import tzutc
import logging
import fsspec
from dataclasses import dataclass
from typing import List
from enum import Enum
from ads.config import COMPARTMENT_OCID
from ads.aqua.base import AquaApp

logger = logging.getLogger(__name__)

ICON_FILE_NAME = "icon.txt"
UNKNOWN = "Unknown"


class Tags(Enum):
    TASK = "task"
    LICENSE = "license"
    ORGANIZATION = "organization"
    AQUA_TAG = "OCI_AQUA"
    AQUA_SERVICE_MODEL_TAG = "aqua_service_model"
    AQUA_FINE_TUNED_MODEL_TAG = "aqua_fine_tuned_model"


@dataclass
class AquaModelSummary:
    """Represents a summary of Aqua model."""

    name: str
    id: str
    compartment_id: str
    project_id: str
    time_created: int
    icon: str
    task: str
    license: str
    organization: str
    is_fine_tuned_model: bool


@dataclass
class AquaModel(AquaModelSummary):
    """Represents an Aqua model."""

    model_card: str


class AquaModelApp(AquaApp):
    """Contains APIs for Aqua model.

    Attributes
    ----------

    Methods
    -------
    create(self, **kwargs) -> "AquaModel"
        Creates an instance of Aqua model.
    deploy(..., **kwargs)
        Deploys an Aqua model.
    list(self, ..., **kwargs) -> List["AquaModel"]
        List existing models created via Aqua

    """

    def __init__(self, **kwargs):
        """Initializes an Aqua model."""
        super().__init__(**kwargs)

    def create(self, **kwargs) -> "AquaModel":
        pass

    def get(self, model_id) -> "AquaModel":
        """Gets the information of an Aqua model."""
        model_card = """
# Model Card: Dummy Text Generator
## Description
This is a simple dummy text generator model developed using Hugging Face's Transformers library. It generates random text based on a pre-trained language model.
## Model Details
- Model Name: DummyTextGenerator
- Model Architecture: GPT-2
- Model Size: 125M parameters
- Training Data: Random text from the internet
## Usage
You can use this model to generate dummy text for various purposes, such as testing text processing pipelines or generating placeholder text for design mockups.
Here's an example of how to use it in Python:
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model_name = "dummy-text-generator"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```
## Evaluation
The model does not perform any meaningful text generation but can be used for basic testing purposes.
## License
This model is released under the MIT License.
"""
        return AquaModel(
            **{
                "compartment_id": "ocid1.compartment.oc1..xxxx",
                "project_id": "ocid1.datascienceproject.oc1.eu-frankfurt-1.xxxx",
                "name": "codellama/CodeLlama-7b-Instruct-hf",
                "id": "ocid1.datasciencemodel.oc1.eu-frankfurt-1.xxxx",
                "time_created": "2024-01-08T22:45:42.443000+00:00",
                "icon": "The icon of the model",
                "task": "text_generation",
                "license": "Apache 2.0",
                "organization": "Meta AI",
                "is_fine_tuned_model": False,
                "model_card": model_card,
            }
        )

    def list(
        self, compartment_id: str = None, project_id: str = None, **kwargs
    ) -> List["AquaModelSummary"]:
        """List Aqua models in a given compartment and under certain project.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        project_id: (str, optional). Defaults to `None`.
            The project OCID.
        kwargs
            Additional keyword arguments for `list_call_get_all_results <https://docs.oracle.com/en-us/iaas/tools/python/2.118.1/api/pagination.html#oci.pagination.list_call_get_all_results>`_

        Returns
        -------
        List[dict]:
            The list of the Aqua models.
        """
        compartment_id = compartment_id or COMPARTMENT_OCID
        kwargs.update({"compartment_id": compartment_id, "project_id": project_id})

        models = self.list_resource(self.client.list_models, **kwargs)

        aqua_models = []
        for model in models:  # ModelSummary
            if self._if_show(model):
                # TODO: need to update after model by reference release
                artifact_path = ""
                try:
                    custom_metadata_list = self.client.get_model(
                        model.id
                    ).data.custom_metadata_list
                except Exception as e:
                    # show opc-request-id and status code
                    logger.error(f"Failing to retreive model information. {e}")
                    return []

                for custom_metadata in custom_metadata_list:
                    if custom_metadata.key == "Object Storage Path":
                        artifact_path = custom_metadata.value
                        break

                if not artifact_path:
                    raise FileNotFoundError("Failed to retrieve model artifact path.")

                with fsspec.open(
                    f"{artifact_path}/{ICON_FILE_NAME}", "rb", **self._auth
                ) as f:
                    icon = f.read()
                    aqua_models.append(
                        AquaModelSummary(
                            name=model.display_name,
                            id=model.id,
                            compartment_id=model.compartment_id,
                            project_id=model.project_id,
                            time_created=model.time_created,
                            icon=icon,
                            task=model.freeform_tags.get(Tags.TASK.value, UNKNOWN),
                            license=model.freeform_tags.get(
                                Tags.LICENSE.value, UNKNOWN
                            ),
                            organization=model.freeform_tags.get(
                                Tags.ORGANIZATION.value, UNKNOWN
                            ),
                            is_fine_tuned_model=True
                            if model.freeform_tags.get(
                                Tags.AQUA_FINE_TUNED_MODEL_TAG.value
                            )
                            else False,
                        )
                    )
        return aqua_models

    def _if_show(self, model: "ModelSummary") -> bool:
        """Determine if the given model should be return by `list`."""
        TARGET_TAGS = model.freeform_tags.keys()
        if not Tags.AQUA_TAG.value in TARGET_TAGS:
            return False

        return (
            True
            if (
                Tags.AQUA_SERVICE_MODEL_TAG.value in TARGET_TAGS
                or Tags.AQUA_FINE_TUNED_MODEL_TAG.value in TARGET_TAGS
            )
            else False
        )
