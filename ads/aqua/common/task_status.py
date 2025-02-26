#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass

from ads.common.extended_enum import ExtendedEnum
from ads.common.serializer import DataClassSerializable


class TaskStatusEnum(ExtendedEnum):
    MODEL_VALIDATION_SUCCESSFUL = "MODEL_VALIDATION_SUCCESSFUL"
    MODEL_DOWNLOAD_STARTED = "MODEL_DOWNLOAD_STARTED"
    MODEL_DOWNLOAD_SUCCESSFUL = "MODEL_DOWNLOAD_SUCCESSFUL"
    MODEL_UPLOAD_STARTED = "MODEL_UPLOAD_STARTED"
    MODEL_UPLOAD_SUCCESSFUL = "MODEL_UPLOAD_SUCCESSFUL"
    DATASCIENCE_MODEL_CREATED = "DATASCIENCE_MODEL_CREATED"
    MODEL_REGISTRATION_SUCCESSFUL = "MODEL_REGISTRATION_SUCCESSFUL"
    REGISTRATION_FAILED = "REGISTRATION_FAILED"
    MODEL_DOWNLOAD_INPROGRESS = "MODEL_DOWNLOAD_INPROGRESS"


@dataclass
class TaskStatus(DataClassSerializable):
    state: TaskStatusEnum
    message: str
