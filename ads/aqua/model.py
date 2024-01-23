#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import logging
from dataclasses import dataclass
from typing import List
from ads.aqua.base import AquaApp

logger = logging.getLogger(__name__)

@dataclass
class AquaModelSummary:
    """Represents a summary of Aqua model."""
    id: str
    compartment_id: str
    project_id: str
    created_by: str
    display_name: str
    lifecycle_state: str
    time_created: str
    task: str
    license: str
    organization: str
    is_fine_tuned: bool
    model_card: str

@dataclass
class AquaModel(AquaModelSummary):
    """Represents an Aqua model."""
    icon: str = None

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
                "created_by": "ocid1.user.oc1..xxxx",
                "display_name": "codellama/CodeLlama-7b-Instruct-hf",
                "id": "ocid1.datasciencemodel.oc1.eu-frankfurt-1.xxxx",
                "lifecycle_state": "ACTIVE",
                "time_created": "2024-01-08T22:45:42.443000+00:00",
                "icon": "The icon of the model",
                "task": "text_generation",
                "license": "Apache 2.0",
                "organization": "Meta AI",
                "is_fine_tuned": False,
                "model_card": model_card
            }
        )

    def list(self, compartment_id, project_id=None, **kwargs) -> List["AquaModelSummary"]:
        """Lists Aqua models."""
        return [
            AquaModel(id=f"ocid{i}", compartment_id=compartment_id, project_id=project_id)
            for i in range(5)
        ]
