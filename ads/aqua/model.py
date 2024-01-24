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
        return [
            AquaModelSummary(
                name="Mock fine tuned model 1",
                id="ocid1.datasciencemodel.oc1.iad.xxxx",
                compartment_id="ocid1.compartment.oc1..xxxx",
                project_id="ocid1.datascienceproject.oc1.iad.xxxx",
                time_created=datetime.datetime(
                    2024, 1, 19, 19, 33, 58, 78000, tzinfo=tzutc()
                ),
                icon=b"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MiA1MiI+PHBhdGggZD0iTTQ2Ljk0IDUySDUuMDZDMi4yNyA1MiAwIDQ5LjczIDAgNDYuOTRWNS4wNkMwIDIuMjcgMi4yNyAwIDUuMDYgMGg0MS44N0M0OS43MyAwIDUyIDIuMjcgNTIgNS4wNnY0MS44N2MwIDIuOC0yLjI3IDUuMDctNS4wNiA1LjA3eiIgZmlsbD0iI2I5ZGFjNCIvPjxwYXRoIGQ9Ik00NC4yNSAxOC4yM3YtMy41Yy4xNi0uMDQuMzItLjA4LjQ3LS4xNWEyLjY5IDIuNjkgMCAwMC0xLjA0LTUuMTdjLTEuNDggMC0yLjY5IDEuMjEtMi42OSAyLjY5IDAgLjUzLjE2IDEuMDIuNDIgMS40NGwtNS4xMiA1LjA3Yy0uMjctLjE3LS41Ny0uMy0uODktLjM3di0zLjVjLjE2LS4wNC4zMi0uMDguNDctLjE1YTIuNjkgMi42OSAwIDEwLTMuNzMtMi40OGMwIC41My4xNiAxLjAyLjQyIDEuNDRsLTUuMTIgNS4wN2MtLjI3LS4xNy0uNTctLjMtLjg5LS4zN3YtMy41Yy4xNi0uMDQuMzItLjA4LjQ3LS4xNWEyLjY5IDIuNjkgMCAwMC0xLjA0LTUuMTdjLTEuNDggMC0yLjY5IDEuMjEtMi42OSAyLjY5YTIuNyAyLjcgMCAwMDIuMTIgMi42M3YzLjVhMi42OSAyLjY5IDAgMDAuNTcgNS4zMiAyLjY5MyAyLjY5MyAwIDAwMi43LTIuNjljMC0uNTMtLjE2LTEuMDItLjQyLTEuNDRsNS4xMi01LjA3Yy4yNy4xNy41Ny4zLjg5LjM3djMuNWEyLjY5IDIuNjkgMCAwMC41NyA1LjMyIDIuNjkzIDIuNjkzIDAgMDAyLjctMi42OWMwLS41My0uMTYtMS4wMi0uNDItMS40NGw1LjEyLTUuMDdjLjI3LjE3LjU3LjMuODkuMzd2My41YTIuNjkgMi42OSAwIDAwLjU3IDUuMzIgMi42ODQgMi42ODQgMCAwMDIuNjktMi42OSAyLjczIDIuNzMgMCAwMC0yLjE0LTIuNjN6bS0uNTgtNy42OGExLjU0IDEuNTQgMCAxMS0uMDAxIDMuMDgxIDEuNTQgMS41NCAwIDAxLjAwMS0zLjA4MXptLTguODUgMGExLjU0IDEuNTQgMCAxMS0uMDAxIDMuMDgxIDEuNTQgMS41NCAwIDAxLjAwMS0zLjA4MXpNMjQuNDEgMTIuMWExLjU0IDEuNTQgMCAxMTEuNTQgMS41NGMtLjg0IDAtMS41NC0uNjktMS41NC0xLjU0em0yLjE1IDEwLjE4YTEuNTQgMS41NCAwIDExLTEuMTk5LTIuODQxIDEuNTQgMS41NCAwIDAxMS4xOTkgMi44NDF6bTguODYgMGExLjU0IDEuNTQgMCAxMS0xLjE5OS0yLjg0MSAxLjU0IDEuNTQgMCAwMTEuMTk5IDIuODQxem04Ljg1IDBhMS41NCAxLjU0IDAgMTEuOTQtMS40MmMuMDEuNjItLjM2IDEuMTgtLjk0IDEuNDJ6TTI1LjMzIDM3LjE2Yy0uODUgMC0xLjUxLS4yNi0xLjk3LS43Ny0uNDYtLjUyLS42OS0xLjI1LS42OS0yLjE5cy4yMy0xLjY2LjctMi4xOGMuNDctLjUyIDEuMTItLjc4IDEuOTctLjc4Ljg1IDAgMS41LjI2IDEuOTcuNzguNDYuNTIuNyAxLjI1LjcgMi4xOCAwIC45NS0uMjMgMS42OC0uNjkgMi4xOS0uNDguNTEtMS4xNC43Ny0xLjk5Ljc3em0wLTEuMDRjLjQ4IDAgLjgzLS4xNiAxLjA2LS40OS4yMy0uMzMuMzUtLjguMzUtMS40MyAwLS42Mi0uMTItMS4xLS4zNS0xLjQyLS4yMy0uMzMtLjU5LS40OS0xLjA2LS40OS0uNDggMC0uODMuMTYtMS4wNi40OS0uMjMuMzMtLjM1LjgtLjM1IDEuNDIgMCAuNjIuMTIgMS4xLjM1IDEuNDMuMjIuMzIuNTguNDkgMS4wNi40OXptNC43My45M2gtMS4xOVYzMi45aC43OGwuMTcuNjFjLjE0LS4yLjMxLS4zNS41MS0uNDdzLjQ0LS4xNy43MS0uMTdjLjEyIDAgLjIxLjAxLjMuMDJ2MS4wOGgtLjI1Yy0uMjcgMC0uNDkuMDctLjY3LjIxLS4xOC4xNC0uMjkuMzItLjM0LjU1djIuMzJ6bTQuNDcgMGwtLjE2LS41OGMtLjE0LjIxLS4zMy4zNy0uNTUuNDlzLS40OC4xOC0uNzYuMThjLS40MyAwLS43Ny0uMTItMS4wMi0uMzUtLjI1LS4yNC0uMzctLjU2LS4zNy0uOTcgMC0uNDMuMTctLjc4LjQ5LTEuMDMuMzMtLjI1Ljc3LS4zOCAxLjMzLS4zOGguNjh2LS4xNWMwLS4yMy0uMDYtLjM4LS4xNy0uNDUtLjEyLS4wNy0uMy0uMS0uNTQtLjEtLjQ0IDAtLjkxLjEtMS40Mi4zdi0uOWMuMjEtLjA5LjQ1LS4xNi43My0uMjIuMjgtLjA2LjU2LS4wOS44NS0uMDkuNTQgMCAuOTYuMTIgMS4yNC4zNy4yOS4yNS40My42MS40MyAxLjA3djIuOGgtLjc2em0tMS4xNi0uNzhjLjE3IDAgLjMzLS4wNC40OC0uMTMuMTUtLjA5LjI3LS4yMS4zNi0uMzZ2LS42OGgtLjU2Yy0uMjcgMC0uNDguMDUtLjYzLjE2LS4xNS4xLS4yMi4yNS0uMjIuNDUgMCAuMzguMTkuNTYuNTcuNTZ6bTYuMDMtLjI4di45M2MtLjE2LjA2LS4zNS4xMi0uNTcuMTYtLjIyLjA0LS40NS4wNi0uNjcuMDYtMS4zOCAwLTIuMDYtLjc0LTIuMDYtMi4yMSAwLS42Ni4xOS0xLjE4LjU3LTEuNTYuMzgtLjM4LjktLjU3IDEuNTYtLjU3LjM2IDAgLjcxLjA2IDEuMDYuMTh2LjkyYy0uMjYtLjEzLS41NC0uMi0uODUtLjItLjM3IDAtLjY1LjEtLjg1LjMtLjIuMi0uMy41MS0uMy45MiAwIC40LjA5LjcxLjI2Ljk1LjE4LjIzLjQzLjM1Ljc3LjM1LjE3IDAgLjM0LS4wMi41Mi0uMDZzLjM3LS4xLjU2LS4xN3ptLjg0IDEuMDZWMzJsLS40My0uMzZ2LS40OGgxLjYydjUuODloLTEuMTl6bTUuNjctLjJjLS40LjE5LS44OC4yOS0xLjQ0LjI5LS43MSAwLTEuMjYtLjE5LTEuNjQtLjU2LS4zOC0uMzctLjU3LS45MS0uNTctMS42MiAwLS42Ny4xOC0xLjIuNTUtMS41OC4zNy0uMzguODgtLjU3IDEuNTMtLjU3LjU4IDAgMS4wMS4xNyAxLjMyLjUuMy4zMy40Ni44LjQ2IDEuMzl2LjZoLTIuNjdjLjA1LjMyLjE2LjU2LjM1LjcxLjE5LjE1LjQ1LjIyLjguMjIuMjIgMCAuNDMtLjAyLjYzLS4wNi4yLS4wNC40My0uMTEuNjktLjIxdi44OXptLTEuNi0zLjE4Yy0uNTYgMC0uODYuMjktLjkuODZoMS42M2MtLjAxLS4yNy0uMDgtLjQ4LS4yMS0uNjNhLjY1My42NTMgMCAwMC0uNTItLjIzek0xMi4xOCA0NS4wNWwyLjE3LTUuN2gxLjE4bDIuMTggNS43aC0xLjM3bC0uMzgtMS4xN2gtMi4xbC0uMzggMS4xN2gtMS4zem0yLjAxLTIuMmgxLjQ0bC0uNzItMi4yNC0uNzIgMi4yNHptMy45My42VjQwLjloMS4xOXYyLjUxYzAgLjUxLjIuNzYuNi43Ni4xOCAwIC4zNC0uMDYuNDktLjE3LjE1LS4xMS4yNy0uMjcuMzUtLjQ2VjQwLjloMS4xOXY0LjE1aC0uNzhsLS4xNi0uNjNjLS4zOC40OS0uODcuNzMtMS40NS43My0uNDYgMC0uODEtLjE1LTEuMDYtLjQ0LS4yNC0uMy0uMzctLjcyLS4zNy0xLjI2em01LjA3LjE3VjQxLjhoLS43M3YtLjZsLjc4LS4zMi4zNS0xLjE3aC43OHYxLjE5aC44OXYuODloLS44OXYxLjc2YzAgLjIuMDUuMzUuMTUuNDQuMS4wOS4yNS4xMy40NC4xM2guMTdjLjA2IDAgLjEyLS4wMS4xNy0uMDF2Ljk1Yy0uMTMuMDItLjI0LjAzLS4zNC4wNC0uMS4wMS0uMjEuMDEtLjM0LjAxLS40OCAwLS44My0uMTItMS4wOC0uMzZzLS4zNS0uNjItLjM1LTEuMTN6bTQuNjYgMS41MmMtLjY1IDAtMS4xNS0uMTktMS41Mi0uNTctLjM2LS4zOC0uNTUtLjkyLS41NS0xLjYgMC0uNjkuMTgtMS4yMi41NS0xLjYuMzYtLjM4Ljg3LS41NyAxLjUyLS41N3MxLjE2LjE5IDEuNTIuNTdjLjM2LjM4LjU0LjkxLjU0IDEuNiAwIC42OC0uMTggMS4yMi0uNTQgMS42LS4zNi4zOC0uODcuNTctMS41Mi41N3ptMC0uOWMuNTggMCAuODYtLjQyLjg2LTEuMjcgMC0uODQtLjI5LTEuMjYtLjg2LTEuMjZzLS44Ni40Mi0uODYgMS4yNmMwIC44NS4yOSAxLjI3Ljg2IDEuMjd6bTIuOTMuODF2LTUuN2gxLjI4bDEuODEgMy43NiAxLjc5LTMuNzZoMS4yOHY1LjdoLTEuMjF2LTMuNjlsLTEuNTEgMy4xNWgtLjc1bC0xLjUxLTMuMXYzLjY1aC0xLjE4em03LjI3IDB2LTUuN2gxLjIydjQuNjFoMi41MXYxLjA5aC0zLjczem00LjAzIDBsMS40NC0yLjEzLTEuMzgtMi4wMmgxLjM0bC43NyAxLjI0Ljc5LTEuMjRoMS4yMmwtMS4zOCAyLjAzIDEuNDQgMi4xMmgtMS4zNGwtLjgyLTEuMzYtLjg2IDEuMzZoLTEuMjJ6IiBmaWxsPSIjMzEyZTJjIi8+PHBhdGggZmlsbD0iI2M5NDYzNSIgZD0iTTAgNS44aDE4LjMzdjE4LjMzSDB6Ii8+PHBhdGggZD0iTTYuMjEgMTYuOTZ2Ljc5Yy0uMjEuMTEtLjQ1LjItLjcxLjI2LS4yNi4wNi0uNTQuMDktLjgzLjA5LS44NCAwLTEuNDgtLjI1LTEuOTMtLjc0LS40NS0uNS0uNjctMS4yMS0uNjctMi4xNSAwLS41OC4xMS0xLjA3LjMyLTEuNDguMjEtLjQxLjUxLS43My45MS0uOTUuNC0uMjIuODctLjMzIDEuNDEtLjMzLjI0IDAgLjQ4LjAzLjcxLjA4LjI0LjA2LjQ0LjEzLjYxLjIzdi43OWMtLjI2LS4xMS0uNDktLjE4LS42OC0uMjNzLS4zOS0uMDctLjU4LS4wN2MtLjU2IDAtLjk5LjE3LTEuMy41MS0uMy4zNC0uNDYuODItLjQ2IDEuNDQgMCAuNjcuMTUgMS4xOC40NSAxLjUzcy43NC41NCAxLjMuNTRjLjIyIDAgLjQ0LS4wMy42Ny0uMDhzLjUtLjEyLjc4LS4yM3ptLjgxIDEuMDN2LTUuNDVoMS44N2MxLjMxIDAgMS45Ni42IDEuOTYgMS43OSAwIDEuMi0uNjYgMS43OS0xLjk2IDEuNzloLS45NHYxLjg3aC0uOTN6TTguNyAxMy4zaC0uNzV2Mi4wN2guNzVjLjQzIDAgLjc0LS4wOC45My0uMjQuMTktLjE2LjI5LS40My4yOS0uNzkgMC0uMzctLjEtLjYzLS4yOS0uNzktLjE5LS4xNy0uNS0uMjUtLjkzLS4yNXptMi44NiAyLjE3di0yLjkzaC45NHYyLjkyYzAgLjYzLjEgMS4wOS4zIDEuMzhzLjUyLjQ0Ljk2LjQ0Yy40NCAwIC43Ni0uMTUuOTYtLjQ0LjItLjI5LjMtLjc1LjMtMS4zOHYtMi45MmguOTR2Mi45M2MwIC44OS0uMTggMS41NS0uNTQgMS45OHMtLjkxLjY0LTEuNjUuNjRjLS43NCAwLTEuMjktLjIxLTEuNjUtLjY0LS4zOC0uNDItLjU2LTEuMDgtLjU2LTEuOTh6IiBmaWxsPSIjZmZmIi8+PC9zdmc+",
                task="text_generation",
                license="Apache",
                organization="Meta AI",
                is_fine_tuned_model=True,
            ),
            AquaModelSummary(
                name="Mock fine tuned model 2",
                id="ocid1.datasciencemodel.oc1.iad.aaaaa",
                compartment_id="ocid1.compartment.oc1..xxxx",
                project_id="ocid1.datascienceproject.oc1.iad.xxxx",
                time_created=datetime.datetime(
                    2024, 1, 19, 19, 20, 57, 856000, tzinfo=tzutc()
                ),
                icon=b"data:image/svg+xml;base64,PHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJwcmVmaXhfX3ByZWZpeF9fTGF5ZXJfMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4PSIwIiB5PSIwIiB2aWV3Qm94PSIwIDAgNTIgNTIiIHhtbDpzcGFjZT0icHJlc2VydmUiPjxzdHlsZT4ucHJlZml4X19wcmVmaXhfX3N0MXtmaWxsOiNmZmZ9PC9zdHlsZT48cGF0aCBpZD0icHJlZml4X19wcmVmaXhfX0JhY2tncm91bmQiIGQ9Ik00Ni45NCA1Mkg1LjA2QzIuMjcgNTIgMCA0OS43MyAwIDQ2Ljk0VjUuMDZDMCAyLjI3IDIuMjcgMCA1LjA2IDBoNDEuODdDNDkuNzMgMCA1MiAyLjI3IDUyIDUuMDZ2NDEuODdjMCAyLjgtMi4yNyA1LjA3LTUuMDYgNS4wN3oiIGZpbGw9IiM2ZjU4N2IiLz48cGF0aCBpZD0icHJlZml4X19wcmVmaXhfX0dyYXBoaWNzIiBjbGFzcz0icHJlZml4X19wcmVmaXhfX3N0MSIgZD0iTTE2LjIyIDM2LjQ0Yy4xOC0uNTEuMi0xLjA1LjA4LTEuNTdsMi4xNS0uOTNjLS4xNy0uMzYtLjMyLS43NC0uNDQtMS4xMmwtMi4yMS45NmMtLjMzLS40My0uNzgtLjc2LTEuMy0uOTUtLjcxLS4yNS0xLjQ4LS4yMS0yLjE2LjEyLTEuNC42Ny0yIDIuMzYtMS4zMyAzLjc3YTIuODIgMi44MiAwIDAwMy43NyAxLjMyYy42Ny0uMzEgMS4xOS0uODggMS40NC0xLjZ6bS0xLjk2LjUzYy0uODEuMzktMS43OC4wNC0yLjE3LS43Ni0uMzktLjgxLS4wNC0xLjc4Ljc2LTIuMTcuMzktLjE5LjgzLS4yMSAxLjI0LS4wNy40MS4xNC43NC40NC45My44My4zOC44MS4wNCAxLjc4LS43NiAyLjE3em00LjQ4IDQuMTJsMS4zNS00LjY4Yy0uMzQtLjM3LS42My0uNzctLjkxLTEuMThsLTEuNTcgNS40M2MtLjU2LS4wOS0xLjEzLS4wMS0xLjY1LjI0LTEuNC42Ny0yIDIuMzYtMS4zMyAzLjc3YTIuODIgMi44MiAwIDAwNS4yMS0uMjljLjI1LS43MS4yMS0xLjQ4LS4xMi0yLjE2LS4yMS0uNDctLjU2LS44NS0uOTgtMS4xM3ptLS4wMiAyLjg5Yy0uMTQuNDEtLjQ0Ljc0LS44My45My0uODEuMzktMS43OC4wNC0yLjE3LS43Ni0uMzktLjgxLS4wNC0xLjc4Ljc2LTIuMTcuMzktLjE5LjgzLS4yMSAxLjI0LS4wNy40MS4xNC43NC40NC45My44My4xOS4zOS4yMi44My4wNyAxLjI0em0tNi40Mi0xNC45bDUuMzggMi41M2MtLjA5LS40Ni0uMTYtLjkzLS4xOS0xLjQxbC00Ljc4LTIuMjVjLjA3LS41NC0uMDEtMS4wOS0uMjUtMS41OS0uNjctMS40LTIuMzctMi0zLjc3LTEuMzMtMS40LjY3LTIgMi4zNi0xLjMzIDMuNzdhMi44MiAyLjgyIDAgMDAzLjc3IDEuMzJjLjQ5LS4yMy44OS0uNiAxLjE3LTEuMDR6bS0xLjY4LS4wNGMtLjgxLjM5LTEuNzguMDUtMi4xNy0uNzYtLjM5LS44MS0uMDUtMS43OC43Ni0yLjE3LjgxLS4zOSAxLjc4LS4wNCAyLjE3Ljc2LjE5LjM5LjIxLjgzLjA3IDEuMjRzLS40NC43NC0uODMuOTN6bTM1LjE0LTE4LjFjMC0yLjEzLTUuNzMtMi43LTkuMTEtMi43LTMuMzggMC05LjExLjU3LTkuMTEgMi43IDAgLjEyLjAyLjIzLjA2LjM0aC0uMDZ2OC4xOGMuNDEgMCAuODEuMDIgMS4yLjA2VjE3LjRjMS44Ni45NSA1LjQ4IDEuMjUgNy45MSAxLjI1IDIuNDMgMCA2LjA1LS4zIDcuOTEtMS4yNXYzLjU0Yy0uMTcuNTMtMy4wNCAxLjUyLTcuOTEgMS41Mi0uNjcgMC0xLjMtLjAyLTEuODktLjA1LjM5LjM5Ljc1LjggMS4wNyAxLjI0LjI4LjAxLjU2LjAxLjgxLjAxIDIuNDMgMCA2LjA1LS4zIDcuOTEtMS4yNXYzLjU0Yy0uMTYuNS0yLjY5IDEuNC03LjAyIDEuNTEuMDguMzkuMTQuNzkuMTggMS4xOSAzLjQtLjExIDguMDQtLjc2IDguMDQtMi42OFYxMS4yOGgtLjA2Yy4wNS0uMTEuMDctLjIyLjA3LS4zNHptLTkuMTEgNi41MWMtNC44OCAwLTcuNzQtLjk5LTcuOTEtMS41VjEyLjRjMS44Ni45NSA1LjQ4IDEuMjUgNy45MSAxLjI1IDIuNDMgMCA2LjA1LS4zIDcuOTEtMS4yNXYzLjU0Yy0uMTcuNTItMy4wMyAxLjUxLTcuOTEgMS41MXptMC01LjAxYy00LjgzIDAtNy42OS0uOTctNy45MS0xLjUuMjItLjUzIDMuMDctMS41IDcuOTEtMS41IDQuODggMCA3Ljc0Ljk5IDcuOTEgMS40OS0uMTYuNTItMy4wMyAxLjUxLTcuOTEgMS41MXpNMzQuMDYgMjkuM2MtLjA4LTEuNzItLjgyLTMuMy0yLjA5LTQuNDZhNi4zNjkgNi4zNjkgMCAwMC00LjYzLTEuNjdjLTEuNzIuMDgtMy4zLjgyLTQuNDYgMi4wOWE2LjM4NCA2LjM4NCAwIDAwLTEuNjcgNC42M2MuMDggMS43Mi44MiAzLjMgMi4wOSA0LjQ2YTYuMzkzIDYuMzkzIDAgMDA0LjMzIDEuNjhjLjEgMCAuMiAwIC4zMS0uMDEgMy41My0uMTYgNi4yOC0zLjE3IDYuMTItNi43MnptLTIuNTYgMy44M2E1LjE5IDUuMTkgMCAwMS0zLjYzIDEuNyA1LjE4IDUuMTggMCAwMS0zLjc3LTEuMzYgNS4xOSA1LjE5IDAgMDEtMS43LTMuNjNjLS4wNi0xLjQuNDItMi43NCAxLjM2LTMuNzdhNS4xOSA1LjE5IDAgMDEzLjYzLTEuN2MuMDggMCAuMTctLjAxLjI1LS4wMSAxLjMxIDAgMi41NS40OCAzLjUyIDEuMzdhNS4xOSA1LjE5IDAgMDExLjcgMy42MyA1LjIxIDUuMjEgMCAwMS0xLjM2IDMuNzd6bS0uMzYtNC44N2MuMjEuNDMuMzMuOS4zNSAxLjM5bC0xLjIuMDVjLS4wMi0uMzItLjA5LS42My0uMjMtLjkybDEuMDgtLjUyem0tLjM2LS42bC0uOTYuNzJjLS4wOS0uMTItLjItLjI0LS4zMS0uMzQtLjQ3LS40My0xLjA4LS42NS0xLjczLS42MmwtLjA1LTEuMmMuOTUtLjA1IDEuODguMjkgMi41OS45NC4xNi4xNS4zMi4zMi40Ni41em02LjUzIDkuMDZsLTIuMjEtMS42NmE5LjE1NiA5LjE1NiAwIDAwMS43OC01Ljg5IDkuMTQ5IDkuMTQ5IDAgMDAtMy4wMS02LjQxIDkuMTggOS4xOCAwIDAwLTYuNjYtMi40MWMtNS4xLjI0LTkuMDUgNC41Ny04LjgyIDkuNjdhOS4xNDkgOS4xNDkgMCAwMDMuMDEgNi40MSA5LjE3NCA5LjE3NCAwIDAwNi4yMyAyLjQyYy4xNSAwIC4yOSAwIC40NC0uMDEgMS41Mi0uMDcgMi45Ni0uNSA0LjIzLTEuMjVsMS42IDIuMjcgMy41OCA0LjAyIDMuNTUtMy42Ny0zLjcyLTMuNDl6bS05LjMxLjkyYy0yLjE1LjA5LTQuMjEtLjY0LTUuOC0yLjA5YTguMDAzIDguMDAzIDAgMDEtMi42Mi01LjU4Yy0uMS0yLjE1LjY0LTQuMjEgMi4wOS01LjhhOC4wMDMgOC4wMDMgMCAwMTUuNTgtMi42MmMuMTMtLjAxLjI1LS4wMS4zOC0uMDEgNC4yNyAwIDcuODQgMy4zNyA4LjA0IDcuNjguMSAyLjE1LS42NCA0LjIxLTIuMDkgNS44QTcuOTYzIDcuOTYzIDAgMDEyOCAzNy42NHptNi44MyAxLjQ4bC0xLjU1LTIuMmMuMzctLjI4LjcyLS42IDEuMDQtLjk0bDIuMjEgMS42NiAyLjc4IDIuNjEtMS44MSAxLjg3LTIuNjctM3ptNy45OCAyLjc2bC0zLjc0IDMuNzgtLjg1LS44NCAzLjc0LTMuNzguODUuODR6Ii8+PHBhdGggaWQ9InByZWZpeF9fcHJlZml4X19DUFVfdGFnX2JjZyIgZmlsbD0iI2M4NDUzNCIgZD0iTTUuNiAwaDE4LjMzdjE4LjMzSDUuNnoiLz48cGF0aCBpZD0icHJlZml4X19wcmVmaXhfX0NQVV90YWciIGNsYXNzPSJwcmVmaXhfX3ByZWZpeF9fc3QxIiBkPSJNMTEuODEgMTEuMTZ2Ljc5Yy0uMjEuMTEtLjQ1LjE5LS43MS4yNS0uMjYuMDYtLjU0LjA5LS44My4wOS0uODQgMC0xLjQ4LS4yNS0xLjkzLS43NC0uNDUtLjUtLjY3LTEuMjEtLjY3LTIuMTUgMC0uNTguMTEtMS4wNy4zMi0xLjQ4LjIxLS40LjUyLS43Mi45MS0uOTQuNC0uMjIuODctLjMzIDEuNDEtLjMzLjI0IDAgLjQ4LjAzLjcxLjA4LjI0LjA2LjQ0LjEzLjYxLjIzdi43OWMtLjI2LS4xMS0uNDktLjE4LS42OC0uMjNzLS4zOS0uMDctLjU4LS4wN2MtLjU2IDAtLjk5LjE3LTEuMy41MS0uMy4zNC0uNDYuODItLjQ2IDEuNDQgMCAuNjcuMTUgMS4xOC40NSAxLjUzcy43NC41NCAxLjMuNTRjLjIyIDAgLjQ0LS4wMy42Ny0uMDhzLjUtLjEyLjc4LS4yM3ptLjgyIDEuMDRWNi43NWgxLjg3YzEuMzEgMCAxLjk2LjYgMS45NiAxLjc5IDAgMS4yLS42NiAxLjc5LTEuOTYgMS43OWgtLjk0djEuODdoLS45M3ptMS42Ny00LjdoLS43NHYyLjA3aC43NGMuNDMgMCAuNzQtLjA4LjkzLS4yNC4xOS0uMTYuMjktLjQzLjI5LS43OSAwLS4zNy0uMS0uNjMtLjI5LS43OS0uMTktLjE3LS41LS4yNS0uOTMtLjI1em0yLjg2IDIuMThWNi43NWguOTR2Mi45MmMwIC42My4xIDEuMDkuMyAxLjM4cy41Mi40NC45Ni40NGMuNDQgMCAuNzYtLjE1Ljk2LS40NC4yLS4yOS4zLS43NS4zLTEuMzhWNi43NWguOTR2Mi45M2MwIC44OS0uMTggMS41NS0uNTQgMS45OHMtLjkxLjY0LTEuNjUuNjRjLS43NCAwLTEuMjktLjIxLTEuNjUtLjY0LS4zOC0uNDMtLjU2LTEuMDktLjU2LTEuOTh6Ii8+PC9zdmc+",
                task="text_generation",
                license="MIT",
                organization="OpenAI",
                is_fine_tuned_model=True,
            ),
            AquaModelSummary(
                name="Mock service model 2",
                id="ocid1.datasciencemodel.oc1.iad.bbbb",
                compartment_id="ocid1.compartment.oc1..xxxx",
                project_id="ocid1.datascienceproject.oc1.iad.xxxx",
                time_created=datetime.datetime(
                    2024, 1, 19, 17, 57, 39, 158000, tzinfo=tzutc()
                ),
                icon=b"data:image/svg+xml;base64,PHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJwcmVmaXhfX3ByZWZpeF9fTGF5ZXJfMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4PSIwIiB5PSIwIiB2aWV3Qm94PSIwIDAgNTIgNTIiIHhtbDpzcGFjZT0icHJlc2VydmUiPjxzdHlsZT4ucHJlZml4X19wcmVmaXhfX3N0MXtmaWxsOiNmZmZ9PC9zdHlsZT48cGF0aCBpZD0icHJlZml4X19wcmVmaXhfX0JhY2tncm91bmQiIGQ9Ik00Ni45NCA1Mkg1LjA2QzIuMjcgNTIgMCA0OS43MyAwIDQ2Ljk0VjUuMDZDMCAyLjI3IDIuMjcgMCA1LjA2IDBoNDEuODdDNDkuNzMgMCA1MiAyLjI3IDUyIDUuMDZ2NDEuODdjMCAyLjgtMi4yNyA1LjA3LTUuMDYgNS4wN3oiIGZpbGw9IiM2ZjU4N2IiLz48cGF0aCBpZD0icHJlZml4X19wcmVmaXhfX0dyYXBoaWNzIiBjbGFzcz0icHJlZml4X19wcmVmaXhfX3N0MSIgZD0iTTE2LjIyIDM2LjQ0Yy4xOC0uNTEuMi0xLjA1LjA4LTEuNTdsMi4xNS0uOTNjLS4xNy0uMzYtLjMyLS43NC0uNDQtMS4xMmwtMi4yMS45NmMtLjMzLS40My0uNzgtLjc2LTEuMy0uOTUtLjcxLS4yNS0xLjQ4LS4yMS0yLjE2LjEyLTEuNC42Ny0yIDIuMzYtMS4zMyAzLjc3YTIuODIgMi44MiAwIDAwMy43NyAxLjMyYy42Ny0uMzEgMS4xOS0uODggMS40NC0xLjZ6bS0xLjk2LjUzYy0uODEuMzktMS43OC4wNC0yLjE3LS43Ni0uMzktLjgxLS4wNC0xLjc4Ljc2LTIuMTcuMzktLjE5LjgzLS4yMSAxLjI0LS4wNy40MS4xNC43NC40NC45My44My4zOC44MS4wNCAxLjc4LS43NiAyLjE3em00LjQ4IDQuMTJsMS4zNS00LjY4Yy0uMzQtLjM3LS42My0uNzctLjkxLTEuMThsLTEuNTcgNS40M2MtLjU2LS4wOS0xLjEzLS4wMS0xLjY1LjI0LTEuNC42Ny0yIDIuMzYtMS4zMyAzLjc3YTIuODIgMi44MiAwIDAwNS4yMS0uMjljLjI1LS43MS4yMS0xLjQ4LS4xMi0yLjE2LS4yMS0uNDctLjU2LS44NS0uOTgtMS4xM3ptLS4wMiAyLjg5Yy0uMTQuNDEtLjQ0Ljc0LS44My45My0uODEuMzktMS43OC4wNC0yLjE3LS43Ni0uMzktLjgxLS4wNC0xLjc4Ljc2LTIuMTcuMzktLjE5LjgzLS4yMSAxLjI0LS4wNy40MS4xNC43NC40NC45My44My4xOS4zOS4yMi44My4wNyAxLjI0em0tNi40Mi0xNC45bDUuMzggMi41M2MtLjA5LS40Ni0uMTYtLjkzLS4xOS0xLjQxbC00Ljc4LTIuMjVjLjA3LS41NC0uMDEtMS4wOS0uMjUtMS41OS0uNjctMS40LTIuMzctMi0zLjc3LTEuMzMtMS40LjY3LTIgMi4zNi0xLjMzIDMuNzdhMi44MiAyLjgyIDAgMDAzLjc3IDEuMzJjLjQ5LS4yMy44OS0uNiAxLjE3LTEuMDR6bS0xLjY4LS4wNGMtLjgxLjM5LTEuNzguMDUtMi4xNy0uNzYtLjM5LS44MS0uMDUtMS43OC43Ni0yLjE3LjgxLS4zOSAxLjc4LS4wNCAyLjE3Ljc2LjE5LjM5LjIxLjgzLjA3IDEuMjRzLS40NC43NC0uODMuOTN6bTM1LjE0LTE4LjFjMC0yLjEzLTUuNzMtMi43LTkuMTEtMi43LTMuMzggMC05LjExLjU3LTkuMTEgMi43IDAgLjEyLjAyLjIzLjA2LjM0aC0uMDZ2OC4xOGMuNDEgMCAuODEuMDIgMS4yLjA2VjE3LjRjMS44Ni45NSA1LjQ4IDEuMjUgNy45MSAxLjI1IDIuNDMgMCA2LjA1LS4zIDcuOTEtMS4yNXYzLjU0Yy0uMTcuNTMtMy4wNCAxLjUyLTcuOTEgMS41Mi0uNjcgMC0xLjMtLjAyLTEuODktLjA1LjM5LjM5Ljc1LjggMS4wNyAxLjI0LjI4LjAxLjU2LjAxLjgxLjAxIDIuNDMgMCA2LjA1LS4zIDcuOTEtMS4yNXYzLjU0Yy0uMTYuNS0yLjY5IDEuNC03LjAyIDEuNTEuMDguMzkuMTQuNzkuMTggMS4xOSAzLjQtLjExIDguMDQtLjc2IDguMDQtMi42OFYxMS4yOGgtLjA2Yy4wNS0uMTEuMDctLjIyLjA3LS4zNHptLTkuMTEgNi41MWMtNC44OCAwLTcuNzQtLjk5LTcuOTEtMS41VjEyLjRjMS44Ni45NSA1LjQ4IDEuMjUgNy45MSAxLjI1IDIuNDMgMCA2LjA1LS4zIDcuOTEtMS4yNXYzLjU0Yy0uMTcuNTItMy4wMyAxLjUxLTcuOTEgMS41MXptMC01LjAxYy00LjgzIDAtNy42OS0uOTctNy45MS0xLjUuMjItLjUzIDMuMDctMS41IDcuOTEtMS41IDQuODggMCA3Ljc0Ljk5IDcuOTEgMS40OS0uMTYuNTItMy4wMyAxLjUxLTcuOTEgMS41MXpNMzQuMDYgMjkuM2MtLjA4LTEuNzItLjgyLTMuMy0yLjA5LTQuNDZhNi4zNjkgNi4zNjkgMCAwMC00LjYzLTEuNjdjLTEuNzIuMDgtMy4zLjgyLTQuNDYgMi4wOWE2LjM4NCA2LjM4NCAwIDAwLTEuNjcgNC42M2MuMDggMS43Mi44MiAzLjMgMi4wOSA0LjQ2YTYuMzkzIDYuMzkzIDAgMDA0LjMzIDEuNjhjLjEgMCAuMiAwIC4zMS0uMDEgMy41My0uMTYgNi4yOC0zLjE3IDYuMTItNi43MnptLTIuNTYgMy44M2E1LjE5IDUuMTkgMCAwMS0zLjYzIDEuNyA1LjE4IDUuMTggMCAwMS0zLjc3LTEuMzYgNS4xOSA1LjE5IDAgMDEtMS43LTMuNjNjLS4wNi0xLjQuNDItMi43NCAxLjM2LTMuNzdhNS4xOSA1LjE5IDAgMDEzLjYzLTEuN2MuMDggMCAuMTctLjAxLjI1LS4wMSAxLjMxIDAgMi41NS40OCAzLjUyIDEuMzdhNS4xOSA1LjE5IDAgMDExLjcgMy42MyA1LjIxIDUuMjEgMCAwMS0xLjM2IDMuNzd6bS0uMzYtNC44N2MuMjEuNDMuMzMuOS4zNSAxLjM5bC0xLjIuMDVjLS4wMi0uMzItLjA5LS42My0uMjMtLjkybDEuMDgtLjUyem0tLjM2LS42bC0uOTYuNzJjLS4wOS0uMTItLjItLjI0LS4zMS0uMzQtLjQ3LS40My0xLjA4LS42NS0xLjczLS42MmwtLjA1LTEuMmMuOTUtLjA1IDEuODguMjkgMi41OS45NC4xNi4xNS4zMi4zMi40Ni41em02LjUzIDkuMDZsLTIuMjEtMS42NmE5LjE1NiA5LjE1NiAwIDAwMS43OC01Ljg5IDkuMTQ5IDkuMTQ5IDAgMDAtMy4wMS02LjQxIDkuMTggOS4xOCAwIDAwLTYuNjYtMi40MWMtNS4xLjI0LTkuMDUgNC41Ny04LjgyIDkuNjdhOS4xNDkgOS4xNDkgMCAwMDMuMDEgNi40MSA5LjE3NCA5LjE3NCAwIDAwNi4yMyAyLjQyYy4xNSAwIC4yOSAwIC40NC0uMDEgMS41Mi0uMDcgMi45Ni0uNSA0LjIzLTEuMjVsMS42IDIuMjcgMy41OCA0LjAyIDMuNTUtMy42Ny0zLjcyLTMuNDl6bS05LjMxLjkyYy0yLjE1LjA5LTQuMjEtLjY0LTUuOC0yLjA5YTguMDAzIDguMDAzIDAgMDEtMi42Mi01LjU4Yy0uMS0yLjE1LjY0LTQuMjEgMi4wOS01LjhhOC4wMDMgOC4wMDMgMCAwMTUuNTgtMi42MmMuMTMtLjAxLjI1LS4wMS4zOC0uMDEgNC4yNyAwIDcuODQgMy4zNyA4LjA0IDcuNjguMSAyLjE1LS42NCA0LjIxLTIuMDkgNS44QTcuOTYzIDcuOTYzIDAgMDEyOCAzNy42NHptNi44MyAxLjQ4bC0xLjU1LTIuMmMuMzctLjI4LjcyLS42IDEuMDQtLjk0bDIuMjEgMS42NiAyLjc4IDIuNjEtMS44MSAxLjg3LTIuNjctM3ptNy45OCAyLjc2bC0zLjc0IDMuNzgtLjg1LS44NCAzLjc0LTMuNzguODUuODR6Ii8+PHBhdGggaWQ9InByZWZpeF9fcHJlZml4X19DUFVfdGFnX2JjZyIgZmlsbD0iI2M4NDUzNCIgZD0iTTUuNiAwaDE4LjMzdjE4LjMzSDUuNnoiLz48cGF0aCBpZD0icHJlZml4X19wcmVmaXhfX0NQVV90YWciIGNsYXNzPSJwcmVmaXhfX3ByZWZpeF9fc3QxIiBkPSJNMTEuODEgMTEuMTZ2Ljc5Yy0uMjEuMTEtLjQ1LjE5LS43MS4yNS0uMjYuMDYtLjU0LjA5LS44My4wOS0uODQgMC0xLjQ4LS4yNS0xLjkzLS43NC0uNDUtLjUtLjY3LTEuMjEtLjY3LTIuMTUgMC0uNTguMTEtMS4wNy4zMi0xLjQ4LjIxLS40LjUyLS43Mi45MS0uOTQuNC0uMjIuODctLjMzIDEuNDEtLjMzLjI0IDAgLjQ4LjAzLjcxLjA4LjI0LjA2LjQ0LjEzLjYxLjIzdi43OWMtLjI2LS4xMS0uNDktLjE4LS42OC0uMjNzLS4zOS0uMDctLjU4LS4wN2MtLjU2IDAtLjk5LjE3LTEuMy41MS0uMy4zNC0uNDYuODItLjQ2IDEuNDQgMCAuNjcuMTUgMS4xOC40NSAxLjUzcy43NC41NCAxLjMuNTRjLjIyIDAgLjQ0LS4wMy42Ny0uMDhzLjUtLjEyLjc4LS4yM3ptLjgyIDEuMDRWNi43NWgxLjg3YzEuMzEgMCAxLjk2LjYgMS45NiAxLjc5IDAgMS4yLS42NiAxLjc5LTEuOTYgMS43OWgtLjk0djEuODdoLS45M3ptMS42Ny00LjdoLS43NHYyLjA3aC43NGMuNDMgMCAuNzQtLjA4LjkzLS4yNC4xOS0uMTYuMjktLjQzLjI5LS43OSAwLS4zNy0uMS0uNjMtLjI5LS43OS0uMTktLjE3LS41LS4yNS0uOTMtLjI1em0yLjg2IDIuMThWNi43NWguOTR2Mi45MmMwIC42My4xIDEuMDkuMyAxLjM4cy41Mi40NC45Ni40NGMuNDQgMCAuNzYtLjE1Ljk2LS40NC4yLS4yOS4zLS43NS4zLTEuMzhWNi43NWguOTR2Mi45M2MwIC44OS0uMTggMS41NS0uNTQgMS45OHMtLjkxLjY0LTEuNjUuNjRjLS43NCAwLTEuMjktLjIxLTEuNjUtLjY0LS4zOC0uNDMtLjU2LTEuMDktLjU2LTEuOTh6Ii8+PC9zdmc+",
                task="text_generation",
                license="MIT",
                organization="OpenAI",
                is_fine_tuned_model=False,
            ),
            AquaModelSummary(
                name="Mock service model 1",
                id="ocid1.datasciencemodel.oc1.iad.cccc",
                compartment_id="ocid1.compartment.oc1..xxxx",
                project_id="ocid1.datascienceproject.oc1.iad.xxxx",
                time_created=datetime.datetime(
                    2024, 1, 19, 17, 47, 19, 488000, tzinfo=tzutc()
                ),
                icon=b"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MiA1MiI+PHBhdGggZD0iTTQ2Ljk0IDUySDUuMDZDMi4yNyA1MiAwIDQ5LjczIDAgNDYuOTRWNS4wNkMwIDIuMjcgMi4yNyAwIDUuMDYgMGg0MS44N0M0OS43MyAwIDUyIDIuMjcgNTIgNS4wNnY0MS44N2MwIDIuOC0yLjI3IDUuMDctNS4wNiA1LjA3eiIgZmlsbD0iI2I5ZGFjNCIvPjxwYXRoIGQ9Ik00NC4yNSAxOC4yM3YtMy41Yy4xNi0uMDQuMzItLjA4LjQ3LS4xNWEyLjY5IDIuNjkgMCAwMC0xLjA0LTUuMTdjLTEuNDggMC0yLjY5IDEuMjEtMi42OSAyLjY5IDAgLjUzLjE2IDEuMDIuNDIgMS40NGwtNS4xMiA1LjA3Yy0uMjctLjE3LS41Ny0uMy0uODktLjM3di0zLjVjLjE2LS4wNC4zMi0uMDguNDctLjE1YTIuNjkgMi42OSAwIDEwLTMuNzMtMi40OGMwIC41My4xNiAxLjAyLjQyIDEuNDRsLTUuMTIgNS4wN2MtLjI3LS4xNy0uNTctLjMtLjg5LS4zN3YtMy41Yy4xNi0uMDQuMzItLjA4LjQ3LS4xNWEyLjY5IDIuNjkgMCAwMC0xLjA0LTUuMTdjLTEuNDggMC0yLjY5IDEuMjEtMi42OSAyLjY5YTIuNyAyLjcgMCAwMDIuMTIgMi42M3YzLjVhMi42OSAyLjY5IDAgMDAuNTcgNS4zMiAyLjY5MyAyLjY5MyAwIDAwMi43LTIuNjljMC0uNTMtLjE2LTEuMDItLjQyLTEuNDRsNS4xMi01LjA3Yy4yNy4xNy41Ny4zLjg5LjM3djMuNWEyLjY5IDIuNjkgMCAwMC41NyA1LjMyIDIuNjkzIDIuNjkzIDAgMDAyLjctMi42OWMwLS41My0uMTYtMS4wMi0uNDItMS40NGw1LjEyLTUuMDdjLjI3LjE3LjU3LjMuODkuMzd2My41YTIuNjkgMi42OSAwIDAwLjU3IDUuMzIgMi42ODQgMi42ODQgMCAwMDIuNjktMi42OSAyLjczIDIuNzMgMCAwMC0yLjE0LTIuNjN6bS0uNTgtNy42OGExLjU0IDEuNTQgMCAxMS0uMDAxIDMuMDgxIDEuNTQgMS41NCAwIDAxLjAwMS0zLjA4MXptLTguODUgMGExLjU0IDEuNTQgMCAxMS0uMDAxIDMuMDgxIDEuNTQgMS41NCAwIDAxLjAwMS0zLjA4MXpNMjQuNDEgMTIuMWExLjU0IDEuNTQgMCAxMTEuNTQgMS41NGMtLjg0IDAtMS41NC0uNjktMS41NC0xLjU0em0yLjE1IDEwLjE4YTEuNTQgMS41NCAwIDExLTEuMTk5LTIuODQxIDEuNTQgMS41NCAwIDAxMS4xOTkgMi44NDF6bTguODYgMGExLjU0IDEuNTQgMCAxMS0xLjE5OS0yLjg0MSAxLjU0IDEuNTQgMCAwMTEuMTk5IDIuODQxem04Ljg1IDBhMS41NCAxLjU0IDAgMTEuOTQtMS40MmMuMDEuNjItLjM2IDEuMTgtLjk0IDEuNDJ6TTI1LjMzIDM3LjE2Yy0uODUgMC0xLjUxLS4yNi0xLjk3LS43Ny0uNDYtLjUyLS42OS0xLjI1LS42OS0yLjE5cy4yMy0xLjY2LjctMi4xOGMuNDctLjUyIDEuMTItLjc4IDEuOTctLjc4Ljg1IDAgMS41LjI2IDEuOTcuNzguNDYuNTIuNyAxLjI1LjcgMi4xOCAwIC45NS0uMjMgMS42OC0uNjkgMi4xOS0uNDguNTEtMS4xNC43Ny0xLjk5Ljc3em0wLTEuMDRjLjQ4IDAgLjgzLS4xNiAxLjA2LS40OS4yMy0uMzMuMzUtLjguMzUtMS40MyAwLS42Mi0uMTItMS4xLS4zNS0xLjQyLS4yMy0uMzMtLjU5LS40OS0xLjA2LS40OS0uNDggMC0uODMuMTYtMS4wNi40OS0uMjMuMzMtLjM1LjgtLjM1IDEuNDIgMCAuNjIuMTIgMS4xLjM1IDEuNDMuMjIuMzIuNTguNDkgMS4wNi40OXptNC43My45M2gtMS4xOVYzMi45aC43OGwuMTcuNjFjLjE0LS4yLjMxLS4zNS41MS0uNDdzLjQ0LS4xNy43MS0uMTdjLjEyIDAgLjIxLjAxLjMuMDJ2MS4wOGgtLjI1Yy0uMjcgMC0uNDkuMDctLjY3LjIxLS4xOC4xNC0uMjkuMzItLjM0LjU1djIuMzJ6bTQuNDcgMGwtLjE2LS41OGMtLjE0LjIxLS4zMy4zNy0uNTUuNDlzLS40OC4xOC0uNzYuMThjLS40MyAwLS43Ny0uMTItMS4wMi0uMzUtLjI1LS4yNC0uMzctLjU2LS4zNy0uOTcgMC0uNDMuMTctLjc4LjQ5LTEuMDMuMzMtLjI1Ljc3LS4zOCAxLjMzLS4zOGguNjh2LS4xNWMwLS4yMy0uMDYtLjM4LS4xNy0uNDUtLjEyLS4wNy0uMy0uMS0uNTQtLjEtLjQ0IDAtLjkxLjEtMS40Mi4zdi0uOWMuMjEtLjA5LjQ1LS4xNi43My0uMjIuMjgtLjA2LjU2LS4wOS44NS0uMDkuNTQgMCAuOTYuMTIgMS4yNC4zNy4yOS4yNS40My42MS40MyAxLjA3djIuOGgtLjc2em0tMS4xNi0uNzhjLjE3IDAgLjMzLS4wNC40OC0uMTMuMTUtLjA5LjI3LS4yMS4zNi0uMzZ2LS42OGgtLjU2Yy0uMjcgMC0uNDguMDUtLjYzLjE2LS4xNS4xLS4yMi4yNS0uMjIuNDUgMCAuMzguMTkuNTYuNTcuNTZ6bTYuMDMtLjI4di45M2MtLjE2LjA2LS4zNS4xMi0uNTcuMTYtLjIyLjA0LS40NS4wNi0uNjcuMDYtMS4zOCAwLTIuMDYtLjc0LTIuMDYtMi4yMSAwLS42Ni4xOS0xLjE4LjU3LTEuNTYuMzgtLjM4LjktLjU3IDEuNTYtLjU3LjM2IDAgLjcxLjA2IDEuMDYuMTh2LjkyYy0uMjYtLjEzLS41NC0uMi0uODUtLjItLjM3IDAtLjY1LjEtLjg1LjMtLjIuMi0uMy41MS0uMy45MiAwIC40LjA5LjcxLjI2Ljk1LjE4LjIzLjQzLjM1Ljc3LjM1LjE3IDAgLjM0LS4wMi41Mi0uMDZzLjM3LS4xLjU2LS4xN3ptLjg0IDEuMDZWMzJsLS40My0uMzZ2LS40OGgxLjYydjUuODloLTEuMTl6bTUuNjctLjJjLS40LjE5LS44OC4yOS0xLjQ0LjI5LS43MSAwLTEuMjYtLjE5LTEuNjQtLjU2LS4zOC0uMzctLjU3LS45MS0uNTctMS42MiAwLS42Ny4xOC0xLjIuNTUtMS41OC4zNy0uMzguODgtLjU3IDEuNTMtLjU3LjU4IDAgMS4wMS4xNyAxLjMyLjUuMy4zMy40Ni44LjQ2IDEuMzl2LjZoLTIuNjdjLjA1LjMyLjE2LjU2LjM1LjcxLjE5LjE1LjQ1LjIyLjguMjIuMjIgMCAuNDMtLjAyLjYzLS4wNi4yLS4wNC40My0uMTEuNjktLjIxdi44OXptLTEuNi0zLjE4Yy0uNTYgMC0uODYuMjktLjkuODZoMS42M2MtLjAxLS4yNy0uMDgtLjQ4LS4yMS0uNjNhLjY1My42NTMgMCAwMC0uNTItLjIzek0xMi4xOCA0NS4wNWwyLjE3LTUuN2gxLjE4bDIuMTggNS43aC0xLjM3bC0uMzgtMS4xN2gtMi4xbC0uMzggMS4xN2gtMS4zem0yLjAxLTIuMmgxLjQ0bC0uNzItMi4yNC0uNzIgMi4yNHptMy45My42VjQwLjloMS4xOXYyLjUxYzAgLjUxLjIuNzYuNi43Ni4xOCAwIC4zNC0uMDYuNDktLjE3LjE1LS4xMS4yNy0uMjcuMzUtLjQ2VjQwLjloMS4xOXY0LjE1aC0uNzhsLS4xNi0uNjNjLS4zOC40OS0uODcuNzMtMS40NS43My0uNDYgMC0uODEtLjE1LTEuMDYtLjQ0LS4yNC0uMy0uMzctLjcyLS4zNy0xLjI2em01LjA3LjE3VjQxLjhoLS43M3YtLjZsLjc4LS4zMi4zNS0xLjE3aC43OHYxLjE5aC44OXYuODloLS44OXYxLjc2YzAgLjIuMDUuMzUuMTUuNDQuMS4wOS4yNS4xMy40NC4xM2guMTdjLjA2IDAgLjEyLS4wMS4xNy0uMDF2Ljk1Yy0uMTMuMDItLjI0LjAzLS4zNC4wNC0uMS4wMS0uMjEuMDEtLjM0LjAxLS40OCAwLS44My0uMTItMS4wOC0uMzZzLS4zNS0uNjItLjM1LTEuMTN6bTQuNjYgMS41MmMtLjY1IDAtMS4xNS0uMTktMS41Mi0uNTctLjM2LS4zOC0uNTUtLjkyLS41NS0xLjYgMC0uNjkuMTgtMS4yMi41NS0xLjYuMzYtLjM4Ljg3LS41NyAxLjUyLS41N3MxLjE2LjE5IDEuNTIuNTdjLjM2LjM4LjU0LjkxLjU0IDEuNiAwIC42OC0uMTggMS4yMi0uNTQgMS42LS4zNi4zOC0uODcuNTctMS41Mi41N3ptMC0uOWMuNTggMCAuODYtLjQyLjg2LTEuMjcgMC0uODQtLjI5LTEuMjYtLjg2LTEuMjZzLS44Ni40Mi0uODYgMS4yNmMwIC44NS4yOSAxLjI3Ljg2IDEuMjd6bTIuOTMuODF2LTUuN2gxLjI4bDEuODEgMy43NiAxLjc5LTMuNzZoMS4yOHY1LjdoLTEuMjF2LTMuNjlsLTEuNTEgMy4xNWgtLjc1bC0xLjUxLTMuMXYzLjY1aC0xLjE4em03LjI3IDB2LTUuN2gxLjIydjQuNjFoMi41MXYxLjA5aC0zLjczem00LjAzIDBsMS40NC0yLjEzLTEuMzgtMi4wMmgxLjM0bC43NyAxLjI0Ljc5LTEuMjRoMS4yMmwtMS4zOCAyLjAzIDEuNDQgMi4xMmgtMS4zNGwtLjgyLTEuMzYtLjg2IDEuMzZoLTEuMjJ6IiBmaWxsPSIjMzEyZTJjIi8+PHBhdGggZmlsbD0iI2M5NDYzNSIgZD0iTTAgNS44aDE4LjMzdjE4LjMzSDB6Ii8+PHBhdGggZD0iTTYuMjEgMTYuOTZ2Ljc5Yy0uMjEuMTEtLjQ1LjItLjcxLjI2LS4yNi4wNi0uNTQuMDktLjgzLjA5LS44NCAwLTEuNDgtLjI1LTEuOTMtLjc0LS40NS0uNS0uNjctMS4yMS0uNjctMi4xNSAwLS41OC4xMS0xLjA3LjMyLTEuNDguMjEtLjQxLjUxLS43My45MS0uOTUuNC0uMjIuODctLjMzIDEuNDEtLjMzLjI0IDAgLjQ4LjAzLjcxLjA4LjI0LjA2LjQ0LjEzLjYxLjIzdi43OWMtLjI2LS4xMS0uNDktLjE4LS42OC0uMjNzLS4zOS0uMDctLjU4LS4wN2MtLjU2IDAtLjk5LjE3LTEuMy41MS0uMy4zNC0uNDYuODItLjQ2IDEuNDQgMCAuNjcuMTUgMS4xOC40NSAxLjUzcy43NC41NCAxLjMuNTRjLjIyIDAgLjQ0LS4wMy42Ny0uMDhzLjUtLjEyLjc4LS4yM3ptLjgxIDEuMDN2LTUuNDVoMS44N2MxLjMxIDAgMS45Ni42IDEuOTYgMS43OSAwIDEuMi0uNjYgMS43OS0xLjk2IDEuNzloLS45NHYxLjg3aC0uOTN6TTguNyAxMy4zaC0uNzV2Mi4wN2guNzVjLjQzIDAgLjc0LS4wOC45My0uMjQuMTktLjE2LjI5LS40My4yOS0uNzkgMC0uMzctLjEtLjYzLS4yOS0uNzktLjE5LS4xNy0uNS0uMjUtLjkzLS4yNXptMi44NiAyLjE3di0yLjkzaC45NHYyLjkyYzAgLjYzLjEgMS4wOS4zIDEuMzhzLjUyLjQ0Ljk2LjQ0Yy40NCAwIC43Ni0uMTUuOTYtLjQ0LjItLjI5LjMtLjc1LjMtMS4zOHYtMi45MmguOTR2Mi45M2MwIC44OS0uMTggMS41NS0uNTQgMS45OHMtLjkxLjY0LTEuNjUuNjRjLS43NCAwLTEuMjktLjIxLTEuNjUtLjY0LS4zOC0uNDItLjU2LTEuMDgtLjU2LTEuOTh6IiBmaWxsPSIjZmZmIi8+PC9zdmc+",
                task="text_generation",
                license="Apache",
                organization="Meta AI",
                is_fine_tuned_model=False,
            ),
        ]

    # def list(
    #     self, compartment_id: str = None, project_id: str = None, **kwargs
    # ) -> List["AquaModelSummary"]:
    #     """List Aqua models in a given compartment and under certain project.

    #     Parameters
    #     ----------
    #     compartment_id: (str, optional). Defaults to `None`.
    #         The compartment OCID.
    #     project_id: (str, optional). Defaults to `None`.
    #         The project OCID.
    #     kwargs
    #         Additional keyword arguments for `list_call_get_all_results <https://docs.oracle.com/en-us/iaas/tools/python/2.118.1/api/pagination.html#oci.pagination.list_call_get_all_results>`_

    #     Returns
    #     -------
    #     List[dict]:
    #         The list of the Aqua models.
    #     """
    #     compartment_id = compartment_id or COMPARTMENT_OCID
    #     kwargs.update({"compartment_id": compartment_id, "project_id": project_id})

    #     models = self.list_resource(self.client.list_models, **kwargs)

    #     aqua_models = []
    #     for model in models:  # ModelSummary
    #         if self._if_show(model):
    #             # TODO: need to update after model by reference release
    #             artifact_path = ""
    #             try:
    #                 custom_metadata_list = self.client.get_model(
    #                     model.id
    #                 ).data.custom_metadata_list
    #             except Exception as e:
    #                 # show opc-request-id and status code
    #                 logger.error(f"Failing to retreive model information. {e}")
    #                 return []

    #             for custom_metadata in custom_metadata_list:
    #                 if custom_metadata.key == "Object Storage Path":
    #                     artifact_path = custom_metadata.value
    #                     break

    #             if not artifact_path:
    #                 raise FileNotFoundError("Failed to retrieve model artifact path.")

    #             with fsspec.open(
    #                 f"{artifact_path}/{ICON_FILE_NAME}", "rb", **self._auth
    #             ) as f:
    #                 icon = f.read()
    #                 aqua_models.append(
    #                     AquaModelSummary(
    #                         name=model.display_name,
    #                         id=model.id,
    #                         compartment_id=model.compartment_id,
    #                         project_id=model.project_id,
    #                         time_created=model.time_created,
    #                         icon=icon,
    #                         task=model.freeform_tags.get(Tags.TASK.value, UNKNOWN),
    #                         license=model.freeform_tags.get(
    #                             Tags.LICENSE.value, UNKNOWN
    #                         ),
    #                         organization=model.freeform_tags.get(
    #                             Tags.ORGANIZATION.value, UNKNOWN
    #                         ),
    #                         is_fine_tuned_model=True
    #                         if model.freeform_tags.get(
    #                             Tags.AQUA_FINE_TUNED_MODEL_TAG.value
    #                         )
    #                         else False,
    #                     )
    #                 )
    #     return aqua_models

    # def _if_show(self, model: "ModelSummary") -> bool:
    #     """Determine if the given model should be return by `list`."""
    #     TARGET_TAGS = model.freeform_tags.keys()
    #     if not Tags.AQUA_TAG.value in TARGET_TAGS:
    #         return False

    #     return (
    #         True
    #         if (
    #             Tags.AQUA_SERVICE_MODEL_TAG.value in TARGET_TAGS
    #             or Tags.AQUA_FINE_TUNED_MODEL_TAG.value in TARGET_TAGS
    #         )
    #         else False
    #     )
