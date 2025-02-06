#!/usr/bin/env python

# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass
from typing import List, Optional

from ads.aqua.evaluation.entities import AquaEvaluationDetail, AquaEvaluationSummary
from ads.aqua.model.entities import AquaModel, AquaModelSummary
from ads.aqua.modeldeployment.entities import AquaDeployment, AquaDeploymentDetail
from ads.common.extended_enum import ExtendedEnum
from ads.common.serializer import DataClassSerializable


class RequestResponseType(ExtendedEnum):
    ListEvaluations = "ListEvaluations"
    EvaluationDetails = "EvaluationDetails"
    ListDeployments = "ListDeployments"
    DeploymentDetails = "DeploymentDetails"
    ListModels = "ListModels"
    ModelDetails = "ModelDetails"
    AdsVersion = "AdsVersion"
    CompatibilityCheck = "CompatibilityCheck"
    Error = "Error"


@dataclass
class BaseResponse(DataClassSerializable):
    message_id: str
    kind: RequestResponseType
    data: Optional[object]


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
class EvaluationDetailsRequest(BaseRequest):
    kind = RequestResponseType.EvaluationDetails
    evaluation_id: str


@dataclass
class ListModelsRequest(BaseRequest):
    compartment_id: Optional[str] = None
    project_id: Optional[str] = None
    model_type: Optional[str] = None
    kind = RequestResponseType.ListDeployments


@dataclass
class ModelDetailsRequest(BaseRequest):
    kind = RequestResponseType.ModelDetails
    model_id: str


@dataclass
class ListDeploymentRequest(BaseRequest):
    compartment_id: str
    project_id: Optional[str] = None
    kind = RequestResponseType.ListDeployments


@dataclass
class DeploymentDetailsRequest(BaseRequest):
    model_deployment_id: str
    kind = RequestResponseType.DeploymentDetails


@dataclass
class ListEvaluationsResponse(BaseResponse):
    data: List[AquaEvaluationSummary]


@dataclass
class EvaluationDetailsResponse(BaseResponse):
    data: AquaEvaluationDetail


@dataclass
class ListDeploymentResponse(BaseResponse):
    data: List[AquaDeployment]


@dataclass
class ModelDeploymentDetailsResponse(BaseResponse):
    data: AquaDeploymentDetail


@dataclass
class ListModelsResponse(BaseResponse):
    data: List[AquaModelSummary]


@dataclass
class ModelDetailsResponse(BaseResponse):
    data: AquaModel


@dataclass
class AdsVersionRequest(BaseRequest):
    kind: RequestResponseType.AdsVersion


@dataclass
class AdsVersionResponse(BaseResponse):
    data: str


@dataclass
class CompatibilityCheckRequest(BaseRequest):
    kind: RequestResponseType.CompatibilityCheck


@dataclass
class CompatibilityCheckResponse(BaseResponse):
    data: object


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
