from typing import Union, List
from ads.aqua.decorator import handle_exceptions
from ads.aqua.evaluation import AquaEvaluationApp
from ads.aqua.extension.aqua_ws_msg_handler import AquaWSMsgHandler
from ads.aqua.extension.models.ws_models import (
    RequestResponseType,
    ListEvaluationsResponse,
    ListEvaluationsRequest,
)
from ads.config import COMPARTMENT_OCID
from tornado.web import HTTPError


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
            list_eval_request.project_id,
        )
        response = ListEvaluationsResponse(
            message_id=list_eval_request.message_id,
            kind=RequestResponseType.ListEvaluations,
            data=eval_list,
        )
        return response
