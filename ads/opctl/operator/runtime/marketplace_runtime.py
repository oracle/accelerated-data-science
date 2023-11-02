from dataclasses import dataclass
from typing import ClassVar, Dict

from ads.common.serializer import DataClassSerializable

from ads.common.extended_enum import ExtendedEnum

from ads.opctl.operator.runtime.runtime import Runtime


class OPERATOR_MARKETPLACE_LOCAL_RUNTIME_TYPE(ExtendedEnum):
    PYTHON = "python"


MARKETPLACE_OPERATOR_LOCAL_KIND = "marketplace.local"


@dataclass(repr=True)
class MarketplacePythonRuntime(Runtime):
    """Represents a python operator runtime."""

    _schema: ClassVar[str] = "python_marketplace_runtime_schema.yaml"

    type: str = OPERATOR_MARKETPLACE_LOCAL_RUNTIME_TYPE.PYTHON.value
    version: str = "v1"

    @classmethod
    def init(cls, **kwargs: Dict) -> "MarketplacePythonRuntime":
        """Initializes a starter specification for the runtime.

        Returns
        -------
        PythonRuntime
            The runtime instance.
        """
        instance = cls()
        instance.kind = MARKETPLACE_OPERATOR_LOCAL_KIND
        return instance
