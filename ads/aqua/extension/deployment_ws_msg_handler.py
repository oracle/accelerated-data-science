import json
from typing import List, Union

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.extension.aqua_ws_msg_handler import AquaWSMsgHandler
from ads.aqua.extension.models.ws_models import RequestResponseType, ListDeploymentResponse, ListDeploymentRequest, \
    ModelDeploymentDetailsResponse
from ads.aqua.modeldeployment import AquaDeploymentApp
from ads.config import COMPARTMENT_OCID


class AquaDeploymentWSMsgHandler(AquaWSMsgHandler):

    def __init__(self, message: Union[str, bytes]):
        super().__init__(message)

    @staticmethod
    def get_message_types() -> List[RequestResponseType]:
        return [RequestResponseType.ListDeployments, RequestResponseType.DeploymentDetails]

    @handle_exceptions
    def process(self) -> ListDeploymentResponse | ModelDeploymentDetailsResponse:
        request = json.loads(self.message)
        if request.get("kind") == "ListDeployments":
            deployment_list = AquaDeploymentApp().list(
                compartment_id=request.get("compartment_id") or COMPARTMENT_OCID,
                project_id=request.get("project_id"),
            )
            response = ListDeploymentResponse(
                message_id=request.get("message_id"),
                kind=RequestResponseType.ListDeployments,
                data=deployment_list,
            )
            return response
        elif request.get("kind") == "DeploymentDetails":
            deployment_details = AquaDeploymentApp().get(request.get("model_deployment_id"))
            response = ModelDeploymentDetailsResponse(message_id=request.get("message_id"),
                                                      kind=RequestResponseType.DeploymentDetails,
                                                      data=deployment_details)
            return response
