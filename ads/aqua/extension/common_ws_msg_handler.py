#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from importlib import metadata
from typing import List, Union

from ads.aqua import ODSC_MODEL_COMPARTMENT_OCID, fetch_service_compartment
from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.common.errors import AquaResourceAccessError
from ads.aqua.common.utils import known_realm
from ads.aqua.extension.aqua_ws_msg_handler import AquaWSMsgHandler
from ads.aqua.extension.models.ws_models import (
    AdsVersionResponse,
    CompatibilityCheckResponse,
    RequestResponseType,
)


class AquaCommonWsMsgHandler(AquaWSMsgHandler):
    @staticmethod
    def get_message_types() -> List[RequestResponseType]:
        return [RequestResponseType.AdsVersion, RequestResponseType.CompatibilityCheck]

    def __init__(self, message: Union[str, bytes]):
        super().__init__(message)

    @handle_exceptions
    def process(self) -> Union[AdsVersionResponse, CompatibilityCheckResponse]:
        request = json.loads(self.message)
        if request.get("kind") == "AdsVersion":
            version = metadata.version("oracle_ads")
            response = AdsVersionResponse(
                message_id=request.get("message_id"),
                kind=RequestResponseType.AdsVersion,
                data=version,
            )
            return response
        if request.get("kind") == "CompatibilityCheck":
            if ODSC_MODEL_COMPARTMENT_OCID or fetch_service_compartment():
                return CompatibilityCheckResponse(
                    message_id=request.get("message_id"),
                    kind=RequestResponseType.CompatibilityCheck,
                    data={"status": "ok"},
                )
            elif known_realm():
                return CompatibilityCheckResponse(
                    message_id=request.get("message_id"),
                    kind=RequestResponseType.CompatibilityCheck,
                    data={"status": "compatible"},
                )
            else:
                raise AquaResourceAccessError(
                    "The AI Quick actions extension is not compatible in the given region."
                )
