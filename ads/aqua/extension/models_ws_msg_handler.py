from typing import List, Union

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.extension.aqua_ws_msg_handler import AquaWSMsgHandler
from ads.aqua.extension.models.ws_models import RequestResponseType,ListModelsResponse, ListModelsRequest
from ads.aqua.model import AquaModelApp
from ads.config import COMPARTMENT_OCID


class AquaModelWSMsgHandler(AquaWSMsgHandler):

    def __init__(self, message: Union[str, bytes]):
        super().__init__(message)

    @staticmethod
    def get_message_types() -> List[RequestResponseType]:
        return [RequestResponseType.ListModels]

    @handle_exceptions
    def process(self) -> ListModelsResponse:
        list_models_request = ListModelsRequest.from_json(self.message)
        print(list_models_request)
        models_list = AquaModelApp().list(
            compartment_id=list_models_request.compartment_id or COMPARTMENT_OCID,
            project_id=list_models_request.project_id,
            model_type=list_models_request.model_type
        )
        response = ListModelsResponse(
            message_id=list_models_request.message_id,
            kind=RequestResponseType.ListModels,
            data=models_list,
        )
        return response
