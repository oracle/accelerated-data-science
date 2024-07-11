#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from typing import List, Union

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.evaluation import AquaEvaluationApp
from ads.aqua.extension.aqua_ws_msg_handler import AquaWSMsgHandler
from ads.aqua.extension.models.ws_models import (
    EvaluationDetailsResponse,
    ListEvaluationsResponse,
    RequestResponseType,
)
from ads.config import COMPARTMENT_OCID


class AquaEvaluationWSMsgHandler(AquaWSMsgHandler):
    @staticmethod
    def get_message_types() -> List[RequestResponseType]:
        return [
            RequestResponseType.ListEvaluations,
            RequestResponseType.EvaluationDetails,
        ]

    def __init__(self, message: Union[str, bytes]):
        super().__init__(message)

    @handle_exceptions
    def process(self) -> Union[ListEvaluationsResponse, EvaluationDetailsResponse]:
        request = json.loads(self.message)
        if request["kind"] == "ListEvaluations":
            return self.list_evaluations(request)
        if request["kind"] == "EvaluationDetails":
            return self.evaluation_details(request)

    @staticmethod
    def list_evaluations(request) -> ListEvaluationsResponse:
        eval_list = AquaEvaluationApp().list(
            request.get("compartment_id") or COMPARTMENT_OCID
        )
        response = ListEvaluationsResponse(
            message_id=request["message_id"],
            kind=RequestResponseType.ListEvaluations,
            data=eval_list,
        )
        return response

    @staticmethod
    def evaluation_details(request) -> EvaluationDetailsResponse:
        evaluation_details = AquaEvaluationApp().get(
            eval_id=request.get("evaluation_id")
        )
        response = EvaluationDetailsResponse(
            message_id=request.get("message_id"),
            kind=RequestResponseType.EvaluationDetails,
            data=evaluation_details,
        )
        return response
