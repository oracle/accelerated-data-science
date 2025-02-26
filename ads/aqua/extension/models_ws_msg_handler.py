#!/usr/bin/env python

# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from logging import getLogger
from typing import List, Union

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.extension.aqua_ws_msg_handler import AquaWSMsgHandler
from ads.aqua.extension.models.ws_models import (
    ListModelsResponse,
    ModelDetailsResponse,
    ModelRegisterRequest,
    RequestResponseType,
)
from ads.aqua.extension.status_manager import (
    RegistrationSubscriber,
    StatusTracker,
    TaskNameEnum,
)
from ads.aqua.model import AquaModelApp

logger = getLogger(__name__)

REGISTRATION_STATUS = "registration_status"


class AquaModelWSMsgHandler(AquaWSMsgHandler):
    status_subscriber = {}
    register_status = {}  # Not threadsafe

    def __init__(self, message: Union[str, bytes]):
        super().__init__(message)

    @staticmethod
    def get_message_types() -> List[RequestResponseType]:
        return [
            RequestResponseType.ListModels,
            RequestResponseType.ModelDetails,
            RequestResponseType.RegisterModelStatus,
        ]

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
        elif request.get("kind") == "RegisterModelStatus":
            task_id = request.get("task_id")
            StatusTracker.add_subscriber(
                subscriber=RegistrationSubscriber(
                    task_id=task_id, subscriber=self.ws_connection
                ),
                notify_latest_status=False,
            )
            # if REGISTRATION_STATUS not in AquaModelWSMsgHandler.status_subscriber:
            #     AquaModelWSMsgHandler.status_subscriber = {
            #         REGISTRATION_STATUS: {job_id: {"subscriber": []}}
            #     }
            # if REGISTRATION_STATUS in AquaModelWSMsgHandler.status_subscriber:
            #     if (
            #         job_id
            #         in AquaModelWSMsgHandler.status_subscriber[REGISTRATION_STATUS]
            #     ):
            #         AquaModelWSMsgHandler.status_subscriber[REGISTRATION_STATUS][
            #             job_id
            #         ]["subscriber"].append(self.ws_connection)
            #     else:
            #         AquaModelWSMsgHandler.status_subscriber[REGISTRATION_STATUS][
            #             job_id
            #         ] = {"subscriber": [self.ws_connection]}

            latest_status = StatusTracker.get_latest_status(
                TaskNameEnum.REGISTRATION_STATUS, task_id=task_id
            )
            logger.info(latest_status)
            # if "state" in AquaModelWSMsgHandler.register_status.get(job_id, {}):
            #     return ModelRegisterRequest(
            #         status=AquaModelWSMsgHandler.register_status[job_id]["state"],
            #         message=AquaModelWSMsgHandler.register_status[job_id]["message"],
            #         job_id=job_id,
            #     )
            if latest_status:
                return ModelRegisterRequest(
                    status=latest_status.state,
                    message=latest_status.message,
                    task_id=task_id,
                )
            else:
                return ModelRegisterRequest(
                    status="SUBSCRIBED", task_id=task_id, message=""
                )
