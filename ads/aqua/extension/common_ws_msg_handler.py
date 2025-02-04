#!/usr/bin/env python

# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import concurrent.futures
import json
import sys
import traceback
from importlib import metadata
from typing import List, Union

import oci

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.common.errors import AquaResourceAccessError
from ads.aqua.common.utils import known_realm
from ads.aqua.extension.aqua_ws_msg_handler import AquaWSMsgHandler
from ads.aqua.extension.models.ws_models import (
    AdsVersionResponse,
    CompatibilityCheckResponse,
    CompatibilityCheckResponseData,
    RequestResponseType,
)
from ads.aqua.extension.utils import ui_compatability_check


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
            service_compartment = None
            response = None
            extension_status = "compatible" if known_realm() else "incompatible"
            try:
                service_compartment = ui_compatability_check()
            except (
                concurrent.futures.TimeoutError,
                oci.exceptions.ConnectTimeout,
            ) as ex:
                response = CompatibilityCheckResponseData(
                    status=extension_status,
                    msg="If you are using custom networking in your notebook session, "
                    "please check if the subnet has service gateway configured.",
                    payload={
                        "status_code": 408,
                        "reason": f"{type(ex).__name__}: {str(ex)}",
                        "exc_info": "".join(
                            traceback.format_exception(*sys.exc_info())
                        ),
                    },
                ).to_dict()
            except Exception as ex:
                response = CompatibilityCheckResponseData(
                    status=extension_status,
                    msg="Unable to load AI Quick Actions configuration. "
                    "Please check if you have set up the policies to enable the extension.",
                    payload={
                        "status_code": 404,
                        "reason": f"{type(ex).__name__}: {str(ex)}",
                        "exc_info": "".join(
                            traceback.format_exception(*sys.exc_info())
                        ),
                    },
                ).to_dict()
            if service_compartment:
                response = CompatibilityCheckResponseData(
                    status="ok",
                    msg="Successfully retrieved service compartment id.",
                    payload={"ODSC_MODEL_COMPARTMENT_OCID": service_compartment},
                ).to_dict()
                return CompatibilityCheckResponse(
                    message_id=request.get("message_id"),
                    kind=RequestResponseType.CompatibilityCheck,
                    data=response,
                )
            elif extension_status == "compatible" and response is not None:
                return CompatibilityCheckResponse(
                    message_id=request.get("message_id"),
                    kind=RequestResponseType.CompatibilityCheck,
                    data=response,
                )
            else:
                raise AquaResourceAccessError(
                    "The AI Quick actions extension is not compatible in the given region."
                )
