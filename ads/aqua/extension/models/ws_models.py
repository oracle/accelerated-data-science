#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass
from typing import List, Optional

from ads.aqua.evaluation.entities import AquaEvaluationSummary
from ads.aqua.model.entities import AquaModelSummary
from ads.common.extended_enum import ExtendedEnumMeta
from ads.common.serializer import DataClassSerializable


class RequestResponseType(str, metaclass=ExtendedEnumMeta):
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
