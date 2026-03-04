#!/usr/bin/env python

# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from importlib import metadata
from typing import List, Optional, Union

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.extension.aqua_ws_msg_handler import AquaWSMsgHandler
from ads.aqua.extension.models.ws_models import (
    AdsVersionResponse,
    RequestResponseType,
)


class AquaCommonWsMsgHandler(AquaWSMsgHandler):
    @staticmethod
    def get_message_types() -> List[RequestResponseType]:
        return [RequestResponseType.AdsVersion, RequestResponseType.CompatibilityCheck]

    def __init__(self, message: Union[str, bytes]):
        super().__init__(message)

    @handle_exceptions
    def process(self) -> Optional[AdsVersionResponse]:
        request = json.loads(self.message)
        if request.get("kind") == "AdsVersion":
            version = metadata.version("oracle_ads")
            response = AdsVersionResponse(
                message_id=request.get("message_id"),
                kind=RequestResponseType.AdsVersion,
                data=version,
            )
            return response
