#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from typing import List, Union

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.extension.aqua_ws_msg_handler import AquaWSMsgHandler
from ads.aqua.extension.models.ws_models import (
    ListModelsResponse,
    ModelDetailsResponse,
    RequestResponseType,
)
from ads.aqua.model import AquaModelApp


class AquaModelWSMsgHandler(AquaWSMsgHandler):
    def __init__(self, message: Union[str, bytes]):
        super().__init__(message)

    @staticmethod
    def get_message_types() -> List[RequestResponseType]:
        return [RequestResponseType.ListModels, RequestResponseType.ModelDetails]

    @handle_exceptions
    def process(self) -> Union[ListModelsResponse, ModelDetailsResponse]:
        request = json.loads(self.message)
        if request.get("kind") == "ListModels":
            models_list = AquaModelApp().list(
                compartment_id=request.get("compartment_id"),
                project_id=request.get("project_id"),
                model_type=request.get("model_type"),
            )
            response = ListModelsResponse(
                message_id=request.get("message_id"),
                kind=RequestResponseType.ListModels,
                data=models_list,
            )
            return response
        elif request.get("kind") == "ModelDetails":
            model_id = request.get("model_id")
            response = AquaModelApp().get(model_id)
            return ModelDetailsResponse(
                message_id=request.get("message_id"),
                kind=RequestResponseType.ModelDetails,
                data=response,
            )
