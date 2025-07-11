from dataclasses import dataclass
from ads.common.extended_enum import ExtendedEnum
from ads.common.serializer import DataClassSerializable


class PolicyStatus(ExtendedEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNVERIFIED = "UNVERIFIED"


@dataclass(repr=False)
class OperationResultSuccess(DataClassSerializable):
    operation: str
    status: PolicyStatus = PolicyStatus.SUCCESS


@dataclass(repr=False)
class OperationResultFailure(DataClassSerializable):
    operation: str
    error: str
    policy_hint: str
    status: PolicyStatus = PolicyStatus.FAILURE


@dataclass(repr=False)
class CommonSettings(DataClassSerializable):
    compartment_id: str
    project_id: str
