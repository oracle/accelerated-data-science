from typing import List, Union

from ads import logger
from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.extension.aqua_ws_msg_handler import AquaWSMsgHandler
from ads.aqua.extension.models.ws_models import RequestResponseType, ListDeploymentResponse, ListDeploymentRequest
from ads.aqua.modeldeployment import AquaDeploymentApp
from ads.config import COMPARTMENT_OCID


class AquaDeploymentWSMsgHandler(AquaWSMsgHandler):

    def __init__(self, message: Union[str, bytes]):
        super().__init__(message)

    @staticmethod
    def get_message_types() -> List[RequestResponseType]:
        return [RequestResponseType.ListDeployments]

    @handle_exceptions
    def process(self) -> ListDeploymentResponse:
        list_deployment_request = ListDeploymentRequest.from_json(self.message)
        deployment_list = AquaDeploymentApp().list(
            compartment_id=list_deployment_request.compartment_id or COMPARTMENT_OCID,
            project_id=list_deployment_request.project_id,
        )
        response = ListDeploymentResponse(
            message_id=list_deployment_request.message_id,
            kind=RequestResponseType.ListDeployments,
            data=deployment_list,
        )
        return response
