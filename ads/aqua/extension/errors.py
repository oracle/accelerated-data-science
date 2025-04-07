#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import uuid
from typing import Any, Dict, List, Optional

from pydantic import Field

from ads.aqua.config.utils.serializer import Serializable

from ads.aqua.constants import (
    AQUA_TROUBLESHOOTING_LINK
)

class Errors(str):
    INVALID_INPUT_DATA_FORMAT = "Invalid format of input data."
    NO_INPUT_DATA = "No input data provided."
    MISSING_REQUIRED_PARAMETER = "Missing required parameter: '{}'"
    MISSING_ONEOF_REQUIRED_PARAMETER = "Either '{}' or '{}' is required."
    INVALID_VALUE_OF_PARAMETER = "Invalid value of parameter: '{}'"

class ReplyDetails(Serializable):
    """Structured reply to be returned to the client."""
    status: int
    troubleshooting_tips: str = Field(f"For general tips on troubleshooting: {AQUA_TROUBLESHOOTING_LINK}",
                                      description="GitHub Link for troubleshooting documentation")
    message: str = Field("Unknown HTTP Error.", description="GitHub Link for troubleshooting documentation")
    service_payload: Optional[Dict[str, Any]] = Field(default_factory=dict)
    reason: str = Field("Unknown error", description="Reason for Error")
    request_id: str = Field(str(uuid.uuid4()), description="Unique ID for tracking the error.")
