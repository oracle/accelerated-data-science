from ads.aqua.evaluation import AquaEvaluationSummary
from ads.aqua.model import AquaModelSummary
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

from ads.common.serializer import DataClassSerializable


class RequestResponseType(str, Enum):
    ListEvaluations = "ListEvaluations"
    ListModels = "ListModels"
    Error = "Error"


@dataclass
class BaseResponse(DataClassSerializable):
    message_id: str
    kind: RequestResponseType
    data: object


@dataclass
class BaseRequest(DataClassSerializable):
    message_id: str
    kind: RequestResponseType


@dataclass
class ListEvaluationsRequest(BaseRequest):

    compartment_id: Optional[str] = None
    limit: Optional[int] = None
    project_id: Optional[str] = None
    kind = RequestResponseType.ListEvaluations


@dataclass
class ListModelsRequest(BaseRequest):
    compartment_id: Optional[str] = None


@dataclass
class ListEvaluationsResponse(BaseResponse):
    data: List[AquaEvaluationSummary]


@dataclass
class ListModelsResponse(BaseResponse):
    data: List[AquaModelSummary]


@dataclass
class AquaWsError(DataClassSerializable):
    status: str
    message: str
    service_payload: Optional[dict]
    reason: Optional[str]


@dataclass
class ErrorResponse(BaseResponse):
    data: AquaWsError
    kind = RequestResponseType.Error
