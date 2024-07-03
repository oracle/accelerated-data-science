#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List, Union

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.evaluation import AquaEvaluationApp
from ads.aqua.extension.aqua_ws_msg_handler import AquaWSMsgHandler
from ads.aqua.extension.models.ws_models import (
    ListEvaluationsRequest,
    ListEvaluationsResponse,
    RequestResponseType,
)
from ads.config import COMPARTMENT_OCID


class AquaEvaluationWSMsgHandler(AquaWSMsgHandler):
    @staticmethod
    def get_message_types() -> List[RequestResponseType]:
        return [RequestResponseType.ListEvaluations]

    def __init__(self, message: Union[str, bytes]):
        super().__init__(message)

    @handle_exceptions
    def process(self) -> ListEvaluationsResponse:
        list_eval_request = ListEvaluationsRequest.from_json(self.message)

        eval_list = AquaEvaluationApp().list(
            list_eval_request.compartment_id or COMPARTMENT_OCID,
        )
        response = ListEvaluationsResponse(
            message_id=list_eval_request.message_id,
            kind=RequestResponseType.ListEvaluations,
            data=eval_list,
        )
        return response
